"""Shared simulation and analysis utilities for the UCNA notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numba as nb
import numpy as np
import scipy.linalg as la
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d


Array = np.ndarray
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
Drift = Callable[[Array | float], Array | float]


def width_binned(
    density: Callable[[float], float],
    tail_tolerance: float = 1e-8,
    grid_points: int = 8_193,
) -> float:
    """Return L such that ``density`` has ``tail_tolerance`` mass outside [-L, L].

    ``density`` may be normalized or unnormalized, but it must be a finite,
    non-negative, integrable scalar function of one scalar argument.
    """

    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie strictly between 0 and 1")
    if grid_points < 257 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 257")

    def checked_density(x: float) -> float:
        value = float(density(x))
        if not np.isfinite(value) or value < 0:
            raise ValueError("density must return finite, non-negative values")
        return value

    # Find a finite integration interval using inexpensive coarse grids. This
    # avoids nested adaptive quadrature when density itself is evaluated by
    # numerical integration (as for ucna_stationary_unnormalized).
    bound = 1.0
    for _ in range(30):
        coarse_x = np.linspace(-bound, bound, 257)
        coarse_density = np.array([checked_density(x) for x in coarse_x])
        peak = float(coarse_density.max())
        if peak <= 0:
            bound *= 2.0
            continue
        edge_ratio = max(coarse_density[0], coarse_density[-1]) / peak
        if edge_ratio <= tail_tolerance * 1e-2:
            break
        bound *= 2.0
    else:
        raise ValueError("could not find finite tails for density")

    grid = np.linspace(-bound, bound, grid_points)
    values = np.array([checked_density(x) for x in grid])
    cumulative = cumulative_trapezoid(values, grid, initial=0.0)
    total_mass = float(cumulative[-1])
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("density must have finite, positive total mass")

    positive = grid[grid_points // 2 :]
    cdf_positive = cumulative[grid_points // 2 :]
    cdf_negative = np.interp(-positive, grid, cumulative)
    outside_fraction = 1.0 - (cdf_positive - cdf_negative) / total_mass
    # outside_fraction decreases with L; interpolate on reversed arrays.
    return float(
        np.interp(tail_tolerance, outside_fraction[::-1], positive[::-1])
    )


def density_interval(
    density: Callable[[float], float],
    tail_tolerance: float = 1e-8,
    grid_points: int = 16_385,
) -> tuple[float, float]:
    """Return equal-tail quantiles containing ``1 - tail_tolerance`` mass.

    The density may be normalized or unnormalized. The returned interval has
    ``tail_tolerance / 2`` probability in each tail and need not be symmetric.
    """

    if not 0 < tail_tolerance < 1:
        raise ValueError("tail_tolerance must lie strictly between 0 and 1")
    if grid_points < 257 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 257")

    def checked_density(x: float) -> float:
        value = float(density(x))
        if not np.isfinite(value) or value < 0:
            raise ValueError("density must return finite, non-negative values")
        return value

    bound = 1.0
    for _ in range(30):
        coarse_x = np.linspace(-bound, bound, 257)
        coarse_density = np.array([checked_density(x) for x in coarse_x])
        peak = float(coarse_density.max())
        if peak > 0 and max(coarse_density[0], coarse_density[-1]) / peak <= tail_tolerance * 1e-2:
            break
        bound *= 2.0
    else:
        raise ValueError("could not find finite tails for density")

    grid = np.linspace(-bound, bound, grid_points)
    values = np.array([checked_density(x) for x in grid])
    cumulative = cumulative_trapezoid(values, grid, initial=0.0)
    total_mass = float(cumulative[-1])
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("density must have finite, positive total mass")

    cdf = cumulative / total_mass
    tail = tail_tolerance / 2.0
    return (
        float(np.interp(tail, cdf, grid)),
        float(np.interp(1.0 - tail, cdf, grid)),
    )


@dataclass(frozen=True)
class SimulationConfig:
    D: float
    tau: float
    dt: float = 0.01
    n_steps: int = 1_000
    seed: int = 0

    def __post_init__(self):
        if self.D < 0:
            raise ValueError("D must be nonnegative")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.n_steps < 0:
            raise ValueError("n_steps must be nonnegative")


@nb.njit
def simulate_trajectory(
    drift, D, tau, dt, n_steps, seed, x0=0.0, eta0=np.nan
):
    """Simulate one trajectory using exact OU updates and Heun for x.

    ``n_steps`` is the number of updates, so returned arrays have length
    ``n_steps + 1``. If eta0 is NaN, eta is initialized at stationarity.
    """

    np.random.seed(seed)
    rho = np.exp(-dt / tau)
    sigma = np.sqrt(D / tau)
    noise_scale = np.sqrt(1.0 - rho**2) * sigma

    x = np.empty(n_steps + 1, dtype=np.float64)
    eta = np.empty(n_steps + 1, dtype=np.float64)
    x[0] = x0
    eta[0] = sigma * np.random.normal() if np.isnan(eta0) else eta0

    for index in range(n_steps):
        force = drift(x[index]) + eta[index]
        predictor = x[index] + dt * force
        eta[index + 1] = rho * eta[index] + noise_scale * np.random.normal()
        corrected_force = drift(predictor) + eta[index + 1]
        x[index + 1] = x[index] + 0.5 * dt * (force + corrected_force)

    return dt * np.arange(n_steps + 1), x, eta


@nb.njit
def simulate_endpoints(
    drift,
    D,
    tau,
    dt,
    n_steps,
    n_realizations,
    seed,
    x0=0.0,
    initial_x=None,
    initial_eta=None,
):
    """Simulate ensemble endpoints and return the final (x, eta) state.

    Passing ``initial_x`` and ``initial_eta`` continues a joint state, as
    required for quenches. Otherwise x=x0 and eta is sampled at stationarity.
    """

    np.random.seed(seed)
    rho = np.exp(-dt / tau)
    sigma = np.sqrt(D / tau)
    noise_scale = np.sqrt(1.0 - rho**2) * sigma
    x = np.empty(n_realizations, dtype=np.float64)
    eta = np.empty(n_realizations, dtype=np.float64)

    for realization in range(n_realizations):
        x[realization] = x0 if initial_x is None else initial_x[realization]
        eta[realization] = (
            sigma * np.random.normal()
            if initial_eta is None
            else initial_eta[realization]
        )

    for _ in range(n_steps):
        for realization in range(n_realizations):
            force = drift(x[realization]) + eta[realization]
            predictor = x[realization] + dt * force
            eta[realization] = (
                rho * eta[realization] + noise_scale * np.random.normal()
            )
            corrected_force = drift(predictor) + eta[realization]
            x[realization] += 0.5 * dt * (force + corrected_force)

    return x, eta


@nb.njit
def simulate_snapshot_counts(
    drift,
    D,
    tau,
    dt,
    n_steps,
    snapshot_steps,
    bin_edges,
    n_realizations,
    seed,
    initial_mean=0.0,
    initial_std=1.0,
    initial_x=None,
    initial_eta=None,
):
    """Simulate an ensemble and histogram x at selected integration steps.

    The returned arrays contain counts for every requested snapshot, left/right
    outside counts, and the final joint ``(x, eta)`` state. Supplying both
    initial arrays preserves the hidden colored-noise state across a quench.
    Bin edges must be uniformly spaced and snapshot steps must be strictly
    increasing, include zero, and not exceed ``n_steps``.
    """

    np.random.seed(seed)
    n_snapshots = len(snapshot_steps)
    n_bins = len(bin_edges) - 1
    counts = np.zeros((n_snapshots, n_bins), dtype=np.int64)
    outside = np.zeros((n_snapshots, 2), dtype=np.int64)
    x = np.empty(n_realizations, dtype=np.float64)
    eta = np.empty(n_realizations, dtype=np.float64)
    rho = np.exp(-dt / tau)
    sigma = np.sqrt(D / tau)
    noise_scale = np.sqrt(1.0 - rho**2) * sigma
    lower = bin_edges[0]
    upper = bin_edges[-1]
    bin_width = (upper - lower) / n_bins

    for realization in range(n_realizations):
        x[realization] = (
            initial_mean + initial_std * np.random.normal()
            if initial_x is None
            else initial_x[realization]
        )
        eta[realization] = (
            sigma * np.random.normal()
            if initial_eta is None
            else initial_eta[realization]
        )

    snapshot_index = 0
    for step in range(n_steps + 1):
        if snapshot_index < n_snapshots and step == snapshot_steps[snapshot_index]:
            for realization in range(n_realizations):
                value = x[realization]
                if value < lower:
                    outside[snapshot_index, 0] += 1
                elif value > upper:
                    outside[snapshot_index, 1] += 1
                else:
                    bin_index = int((value - lower) / bin_width)
                    if bin_index == n_bins:
                        bin_index = n_bins - 1
                    counts[snapshot_index, bin_index] += 1
            snapshot_index += 1

        if step == n_steps:
            break
        for realization in range(n_realizations):
            force = drift(x[realization]) + eta[realization]
            predictor = x[realization] + dt * force
            eta[realization] = (
                rho * eta[realization] + noise_scale * np.random.normal()
            )
            corrected_force = drift(predictor) + eta[realization]
            x[realization] += 0.5 * dt * (force + corrected_force)

    return counts, outside, x, eta


def ucna_stationary_unnormalized(
    x: float,
    tau: float,
    D: float,
    drift: Drift,
    drift_prime: Drift,
    integration_step: float = 0.01,
    lower_bound: float = -5.0,
) -> float:
    """Unnormalized UCNA stationary density using the notebook formula."""

    if x < lower_bound:
        return 0.0
    if D <= 0:
        raise ValueError("D must be positive")
    n_points = int((x - lower_bound) / integration_step)
    ys = np.linspace(lower_bound, x, n_points)
    integral = _trapezoid(drift(ys), ys)
    gamma0 = 1.0 - tau * float(drift_prime(x))
    exponent = (-0.5 * tau * float(drift(x)) ** 2 + integral) / D
    return float(gamma0 * np.exp(exponent))


def generate_stationary_correction(
    drift_expression: Any,
    variable: Any,
    tau: float,
    D: float,
    *,
    reference: float = 0.0,
    integration_bounds: tuple[float, float] = (-np.inf, np.inf),
    quadrature_epsabs: float = 1e-10,
    quadrature_epsrel: float = 1e-10,
    quadrature_limit: int = 200,
) -> Callable[[Array | float], Array | float]:
    """Generate the corrected stationary-density callable from a SymPy drift.

    This is the symbolic translation of ``Correction.nb``. Derivatives entering
    the correction source ``h`` and its antiderivative are obtained exactly with
    SymPy. Only the two global normalization integrals are evaluated numerically,
    once, while constructing the callable.

    Parameters
    ----------
    drift_expression:
        SymPy expression for ``f(x)``.
    variable:
        The SymPy symbol used by ``drift_expression``.
    tau, D:
        Positive correlation time and noise-strength parameters.
    reference:
        Common lower reference of the exponent and correction primitives. Its
        value changes intermediate constants but not the normalized density.
    integration_bounds:
        Domain on which the density is normalized. Infinite bounds reproduce
        the Mathematica notebook.

    Returns
    -------
    callable
        A NumPy-compatible function evaluating the normalized corrected density.
        Diagnostic symbolic expressions and the UCNA callable are exposed as
        attributes on the returned function.

    Notes
    -----
    The construction assumes ``Gamma(x) = 1 - tau*f'(x) > 0`` throughout the
    normalization domain. A clear error is raised if SymPy cannot find the
    elementary antiderivatives required by this formulation.
    """

    if tau <= 0 or D <= 0:
        raise ValueError("tau and D must be positive")
    lower, upper = map(float, integration_bounds)
    if not lower < upper:
        raise ValueError("integration_bounds must be strictly increasing")
    if quadrature_epsabs <= 0 or quadrature_epsrel <= 0:
        raise ValueError("quadrature tolerances must be positive")
    if quadrature_limit < 1:
        raise ValueError("quadrature_limit must be positive")

    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "generate_stationary_correction requires SymPy"
        ) from exc

    x = variable
    if not isinstance(x, sp.Symbol):
        raise TypeError("variable must be a SymPy Symbol")
    drift = sp.sympify(drift_expression)
    unexpected_symbols = drift.free_symbols - {x}
    if unexpected_symbols:
        raise ValueError(
            "drift_expression contains unsubstituted symbols: "
            f"{sorted(map(str, unexpected_symbols))}"
        )

    # Decimal inputs are converted through strings so values such as 0.1 remain
    # exact rationals during differentiation and integration.
    tau_symbolic = sp.Rational(str(tau))
    D_symbolic = sp.Rational(str(D))
    reference_symbolic = sp.Rational(str(reference))
    drift_prime = sp.diff(drift, x)
    gamma = sp.simplify(1 - tau_symbolic * drift_prime)

    exponent_primitive = sp.integrate(drift * gamma / D_symbolic, x)
    if exponent_primitive.has(sp.Integral):
        raise ValueError(
            "SymPy could not integrate f(x)*Gamma(x)/D symbolically"
        )
    exponent = sp.simplify(
        exponent_primitive
        - exponent_primitive.subs(x, reference_symbolic)
    )
    unnormalized_ucna = gamma * sp.exp(exponent)

    gamma_prime = sp.diff(gamma, x)
    gamma_second = sp.diff(gamma, x, 2)
    gamma_third = sp.diff(gamma, x, 3)
    weighted_density = unnormalized_ucna * sp.sqrt(gamma)
    weighted_first = sp.diff(weighted_density, x)
    weighted_second = sp.diff(weighted_density, x, 2)

    correction_source = tau_symbolic / (4 * gamma ** sp.Rational(15, 2)) * (
        D_symbolic**2 * (
            138 * gamma_prime**3 * weighted_density
            - gamma * gamma_prime * (
                53 * gamma_second * weighted_density
                + 78 * gamma_prime * weighted_first
            )
            + 2 * gamma**2 * (
                gamma_third * weighted_density
                + 5 * gamma_second * weighted_first
                + 6 * gamma_prime * weighted_second
            )
        )
        + D_symbolic * (
            -16 * gamma**3 * drift_prime * gamma_prime * weighted_density
            + 4 * gamma**4 * drift_prime * weighted_first
            - 6 * drift * gamma**3 * gamma_second * weighted_density
            - 14 * drift * gamma**3 * gamma_prime * weighted_first
            + 39 * drift * gamma**2 * gamma_prime**2 * weighted_density
        )
        + 4 * drift**2 * gamma**4 * gamma_prime * weighted_density
        - 4 * drift * gamma**5 * drift_prime * weighted_density
    )

    # The exponential cancels analytically for the cubic and logistic models;
    # powsimp/cancel makes that cancellation explicit before integration.
    correction_integrand = sp.factor(sp.cancel(sp.powsimp(
        correction_source * sp.exp(-exponent), force=True
    )))
    correction_primitive = sp.integrate(
        correction_integrand, x, risch=True
    )
    if correction_primitive.has(sp.Integral):
        raise ValueError(
            "SymPy could not integrate the correction source symbolically"
        )
    # Do not call simplify here: for logarithmic primitives it may combine the
    # anchoring constant into an exact rational with thousands of digits.
    correction_primitive = (
        correction_primitive
        - correction_primitive.subs(x, reference_symbolic)
    )

    gamma_function = sp.lambdify(x, gamma, modules="numpy")
    ucna_unnormalized_function = sp.lambdify(
        x, unnormalized_ucna, modules="numpy"
    )
    primitive_function = sp.lambdify(
        x, correction_primitive, modules="numpy"
    )
    weighted_primitive_function = sp.lambdify(
        x, unnormalized_ucna * correction_primitive, modules="numpy"
    )

    def scalar_value(function, value: float, *, tail_zero: bool = False) -> float:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = float(function(value))
        if np.isfinite(result):
            return result
        if tail_zero:
            # Infinite-interval quadrature may probe extreme transformed points
            # where a rapidly vanishing expression overflows internally.
            return 0.0
        raise ValueError(f"symbolic expression is nonfinite at x={value:g}")

    def checked_unnormalized(value: float) -> float:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gamma_value = float(gamma_function(value))
        # Positive infinity can occur only because infinite-interval quadrature
        # probes very large x (e.g. Gamma contains exp(x)); the complete density
        # still tends to zero and is handled by tail_zero below.
        if np.isnan(gamma_value) or gamma_value <= 0:
            raise ValueError(f"Gamma(x) is nonpositive at x={value:g}")
        density_value = scalar_value(
            ucna_unnormalized_function, value, tail_zero=True
        )
        if density_value < 0:
            raise ValueError(f"UCNA density is negative at x={value:g}")
        return density_value

    quad_options = dict(
        epsabs=quadrature_epsabs,
        epsrel=quadrature_epsrel,
        limit=quadrature_limit,
    )
    normalization = quad(
        checked_unnormalized, lower, upper, **quad_options
    )[0]
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("UCNA normalization is nonpositive or nonfinite")

    def weighted_primitive(value: float) -> float:
        # Lambdify the product symbolically so its vanishing tail is evaluated
        # as one expression instead of producing the indeterminate 0*infinity.
        return scalar_value(
            weighted_primitive_function, value, tail_zero=True
        )

    primitive_mean = quad(
        weighted_primitive, lower, upper, **quad_options
    )[0] / normalization

    def ucna_density(values: Array | float) -> Array | float:
        array = np.asarray(values, dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gamma_values = np.asarray(gamma_function(array), dtype=float)
        if np.any(np.isnan(gamma_values)) or np.any(gamma_values <= 0):
            raise ValueError("Gamma(x) must be positive")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = (
                np.asarray(ucna_unnormalized_function(array), dtype=float)
                / normalization
            )
        # At extreme arguments a formula such as exp(-exp(2*x)) may overflow
        # internally although its limiting density is zero.
        result = np.where(np.isfinite(result), result, 0.0)
        if np.any(result < 0):
            raise ValueError("UCNA density evaluation produced invalid values")
        return float(result) if array.ndim == 0 else result

    def corrected_density(values: Array | float) -> Array | float:
        array = np.asarray(values, dtype=float)
        base = np.asarray(ucna_density(array), dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            primitive = np.asarray(primitive_function(array), dtype=float)
            result = base * (
                1.0 + (primitive - primitive_mean) / float(D_symbolic)
            )
        result = np.where((base == 0.0) & ~np.isfinite(result), 0.0, result)
        if np.any(~np.isfinite(result)):
            raise ValueError("corrected density evaluation produced nonfinite values")
        return float(result) if array.ndim == 0 else result

    # Keep the public result a plain callable while exposing derivation details
    # useful for validation and optional source-code generation.
    corrected_density.ucna_density = ucna_density
    corrected_density.drift_expression = drift
    corrected_density.gamma_expression = gamma
    corrected_density.exponent_expression = exponent
    corrected_density.source_expression = correction_source
    corrected_density.primitive_expression = correction_primitive
    corrected_density.normalization = float(normalization)
    corrected_density.primitive_mean = float(primitive_mean)
    corrected_density.integration_bounds = (lower, upper)
    return corrected_density


def normalize_density(values: Array, grid: Array) -> Array:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    mass = float(_trapezoid(values, grid))
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("density has nonpositive or nonfinite mass")
    return values / mass


def histogram_density(samples: Array, bin_edges: Array) -> tuple[Array, Array, Array]:
    """Return density, raw counts, and centers for fixed bin edges."""

    counts, edges = np.histogram(samples, bins=bin_edges, density=False)
    if counts.sum() == 0:
        raise ValueError("no samples fall inside the histogram range")
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return density, counts, centers


def kl_divergence(
    p: Array,
    q: Array,
    nonpositive_q: Literal["raise", "infinite", "skip"] = "raise",
) -> float:
    """Compute D_KL(p || q) after normalizing the supplied bin weights."""

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        raise ValueError("p and q must be finite")
    if np.any(p < 0):
        raise ValueError("p contains negative weights")
    if p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("p and q must have positive total weight")

    p = p / p.sum()
    q = q / q.sum()
    support = p > 0
    invalid = support & (q <= 0)
    if np.any(invalid):
        if nonpositive_q == "raise":
            raise ValueError("q is nonpositive where p is positive")
        if nonpositive_q == "infinite":
            return float("inf")
        support &= ~invalid
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def condition_on_positive_theory(
    counts_by_replica: Array, theory: Array
) -> tuple[Array, Array, float]:
    """Condition empirical counts and theory on bins where theory is positive.

    Returns the retained replica counts, normalized retained theory weights, and
    the fraction of empirical counts excluded by the conditioning.
    """

    counts = np.asarray(counts_by_replica)
    theory = np.asarray(theory, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != theory.size:
        raise ValueError("replica counts and theory bins are incompatible")
    if np.any(counts < 0) or np.any(~np.isfinite(theory)):
        raise ValueError("counts must be nonnegative and theory must be finite")
    retained = theory > 0
    if retained.sum() < 2 or theory[retained].sum() <= 0:
        raise ValueError("fewer than two positive-theory bins remain")
    total = counts.sum()
    if total <= 0:
        raise ValueError("empirical counts have zero total weight")
    excluded_fraction = float(counts[:, ~retained].sum() / total)
    conditioned_theory = theory[retained] / theory[retained].sum()
    return counts[:, retained], conditioned_theory, excluded_fraction


def rescaled_kl(p: Array, q: Array, n_samples: int) -> float:
    """Return N/(M-1) D_KL(p || q), as defined in the rebuttal."""

    n_bins = np.asarray(p).size
    if n_bins < 2 or n_samples <= 0:
        raise ValueError("at least two bins and one sample are required")
    return n_samples / (n_bins - 1) * kl_divergence(p, q)


def kl_from_raw(
    raw_kl: Array | float,
    n_samples: Array | int,
    n_bins: Array | int,
    measure: str = "bias_corrected",
) -> Array | float:
    """Select ordinary bias-corrected KL or the legacy rescaled statistic.

    Use the actual retained sample/bin counts on the declared support. The
    leading bias subtraction is a fixed-bin, well-sampled approximation for
    independently specified theory; it need not yield a nonnegative estimate.
    """

    if measure not in {"bias_corrected", "rescaled"}:
        raise ValueError("measure must be 'bias_corrected' or 'rescaled'")
    samples = np.asarray(n_samples, dtype=float)
    bins = np.asarray(n_bins, dtype=float)
    if (np.any(~np.isfinite(samples)) or np.any(samples <= 0)
            or np.any(~np.isfinite(bins)) or np.any(bins < 2)):
        raise ValueError("KL requires positive sample counts and at least two bins")
    raw = np.asarray(raw_kl, dtype=float)
    result = (raw - (bins - 1) / (2 * samples) if measure == "bias_corrected"
              else raw * samples / (bins - 1))
    return float(result) if result.ndim == 0 else result


def kl_uncertainty_from_raw(
    raw_sem: Array | float,
    n_samples: Array | int,
    n_bins: Array | int,
    measure: str = "bias_corrected",
) -> Array | float:
    """Transform the agreed raw-replica SEM without a bias subtraction."""

    # Validate the same support metadata as the central estimate.
    kl_from_raw(0.0, n_samples, n_bins, measure)
    result = np.asarray(raw_sem, dtype=float)
    if np.any(result < 0):
        raise ValueError("uncertainty cannot be negative")
    if measure == "rescaled":
        result = result * np.asarray(n_samples) / (np.asarray(n_bins) - 1)
    return float(result) if result.ndim == 0 else result


def summarize_kl(
    counts_by_replica: Array,
    theory: Array,
    measure: str = "bias_corrected",
) -> dict[str, Any]:
    """Pooled KL and raw-replica SEM, retaining both reporting conventions.

    Theory must be a genuine nonnegative bin distribution. If a signed
    approximation is conditioned on its positive bins, do that explicitly
    first and retain its excluded-mass/validity diagnostics in the caller.
    The SEM is the established replica variability estimate, not an exact
    confidence interval for the nonlinear pooled estimator.
    """

    counts = np.asarray(counts_by_replica, dtype=float)
    q = np.asarray(theory, dtype=float)
    if counts.ndim != 2 or counts.shape[0] < 2:
        raise ValueError("at least two replica histograms are required")
    if q.ndim != 1 or counts.shape[1] != len(q):
        raise ValueError("replica counts and theory bins are incompatible")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be finite and nonnegative")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or q.sum() <= 0:
        raise ValueError("theory must be finite, nonnegative and have positive mass")
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("each replica must retain at least one sample")
    pooled = counts.sum(axis=0)
    n_samples, n_bins = float(totals.sum()), len(q)
    raw = kl_divergence(pooled, q)
    raw_values = np.array([kl_divergence(row, q) for row in counts])
    raw_mean = float(raw_values.mean())
    raw_sem = float(raw_values.std(ddof=1) / np.sqrt(len(raw_values)))
    bias = (n_bins - 1) / (2 * n_samples)
    return {
        "KL": kl_from_raw(raw, n_samples, n_bins, measure),
        "KL_uncertainty": kl_uncertainty_from_raw(raw_sem, n_samples, n_bins, measure),
        "KL_raw": raw,
        "KL_bias": bias,
        "KL_bias_corrected": raw - bias,
        "KL_rescaled": kl_from_raw(raw, n_samples, n_bins, "rescaled"),
        "KL_raw_uncertainty": raw_sem,
        "KL_bias_corrected_uncertainty": raw_sem,
        "KL_rescaled_uncertainty": kl_uncertainty_from_raw(
            raw_sem, n_samples, n_bins, "rescaled"
        ),
        "replica_raw_KL_mean": raw_mean,
        "replica_raw_KL_sem": raw_sem,
        "N_used": int(n_samples),
        "M_used": n_bins,
        "KL_measure": measure,
        "empty_bins": int(np.sum(pooled == 0)),
        "bins_below_5_counts": int(np.sum(pooled < 5)),
        "replica_min_samples": int(totals.min()),
        "replica_max_samples": int(totals.max()),
    }


def replica_raw_kl_uncertainty(
    counts_by_replica: Array,
    theory: Array,
    pooled_sample_count: int,
    nonpositive_q: Literal["raise", "infinite", "skip"] = "raise",
) -> tuple[float, float, float]:
    """Return replica raw-KL mean, raw-KL SEM, and pooled-rescaled SEM."""

    counts_by_replica = np.asarray(counts_by_replica)
    if counts_by_replica.ndim != 2 or len(counts_by_replica) < 2:
        raise ValueError("at least two replica histograms are required")
    raw_values = np.array(
        [
            kl_divergence(counts, theory, nonpositive_q=nonpositive_q)
            for counts in counts_by_replica
        ]
    )
    raw_sem = float(raw_values.std(ddof=1) / np.sqrt(len(raw_values)))
    rescaled_sem = pooled_sample_count / (len(theory) - 1) * raw_sem
    return float(raw_values.mean()), raw_sem, float(rescaled_sem)


def map_pdf(pdf: Array, source_grid: Array, target_grid: Array) -> Array:
    interpolation = interp1d(
        source_grid, pdf, kind="linear", bounds_error=False, fill_value=0.0
    )
    return np.asarray(interpolation(target_grid))


def solve_fokker_planck_banded(
    x: Array,
    tau: float,
    D: float,
    drift_func,
    diffusion_func,
    p0: Array,
    dt: float = 0.01,
    tmax: float = 5.0,
) -> tuple[Array, Array]:
    """Final backward-Euler banded solver from dinamica.ipynb."""

    x = np.asarray(x, dtype=float)
    p = np.asarray(p0, dtype=float).copy()
    if x.ndim != 1 or p.shape != x.shape or len(x) < 3:
        raise ValueError("x and p0 must be matching one-dimensional arrays")
    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("x must be a uniform grid")
    dx = float(spacing[0])
    if dx <= 0 or dt <= 0 or tmax < 0:
        raise ValueError("grid, dt, and tmax must define forward evolution")

    n_steps = int(np.ceil(tmax / dt))
    times = dt * np.arange(n_steps + 1)
    drift = np.asarray(drift_func(x, tau, D), dtype=float)
    drift_mid = 0.5 * (drift[:-1] + drift[1:])
    p /= p.sum() * dx
    history = [p.copy()]

    for step in range(n_steps):
        diffusion = np.asarray(
            diffusion_func(x, times[step + 1], tau, D), dtype=float
        )
        lower = np.zeros(len(x) - 1)
        diagonal = np.zeros(len(x))
        upper = np.zeros(len(x) - 1)
        interior = np.arange(1, len(x) - 1)

        lower[interior - 1] = (
            drift_mid[interior - 1] / (2 * dx)
            + diffusion[interior - 1] / dx**2
        )
        diagonal[interior] = (
            -(drift_mid[interior] - drift_mid[interior - 1]) / (2 * dx)
            - 2 * diffusion[interior] / dx**2
        )
        upper[interior] = (
            -drift_mid[interior] / (2 * dx)
            + diffusion[interior + 1] / dx**2
        )
        diagonal[0] = -drift_mid[0] / (2 * dx) - diffusion[0] / dx**2
        upper[0] = -drift_mid[0] / (2 * dx) + diffusion[1] / dx**2
        lower[-1] = drift_mid[-1] / (2 * dx) + diffusion[-2] / dx**2
        diagonal[-1] = drift_mid[-1] / (2 * dx) - diffusion[-1] / dx**2

        banded = np.zeros((3, len(x)))
        banded[0, 1:] = -dt * upper
        banded[1] = 1.0 - dt * diagonal
        banded[2, :-1] = -dt * lower
        p = la.solve_banded((1, 1), banded, p)
        mass = p.sum() * dx
        if not np.isfinite(mass) or mass == 0:
            raise FloatingPointError("Fokker-Planck solution lost finite mass")
        p /= mass
        history.append(p.copy())

    return np.asarray(history), times


# Cubic B1/B3 stationary analysis.
# Model-specific globals are prefixed to keep both workflows independent.
from pathlib import Path
from functools import lru_cache
import hashlib
import inspect
import json
import pickle
import time
import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import simpson, cumulative_simpson, cumulative_trapezoid, solve_bvp
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import dawsn
cubic_ROOT = Path.cwd()
cubic_RAW_DIR = cubic_ROOT / 'results/cubic_stationary'
cubic_CACHE_DIR = cubic_RAW_DIR / 'b1_vs_b3'
cubic_KL_MEASURE = 'bias_corrected'
cubic_SOLVER_VERSION = 'cubic-repository-flux-v1'

cubic_ANALYSIS_VERSION = 'integrated-common-support-v1'

cubic_TAIL_EXPONENT = 30.0

cubic_PRODUCTION_N = 8001

cubic_PRODUCTION_DOMAIN_FACTOR = 1.5

cubic_NEGATIVE_MASS_THRESHOLD = 1e-8

cubic_EXCLUDED_EMPIRICAL_MASS_THRESHOLD = 1e-3

cubic_RESIDUAL_TOL = 1e-8

cubic_ABS_OBSERVABLE_TOL = 1e-4

cubic_SEM_FRACTION = 0.25

cubic_TAUS = (0.1, 0.2, 0.3, 1.0, 2.0, 3.0)

cubic_DS_BY_TAU = (
    (0.1, 0.2, 0.3, 1, 2, 3, 10, 20),
    (0.1, 0.2, 0.3, 1, 2, 3, 10),
    (0.1, 0.2, 0.3, 1, 2, 3),
    (0.1, 0.2, 0.3, 1, 2),
    (0.03, 0.1, 0.2, 0.3, 1),
    (0.02, 0.03, 0.1, 0.2, 0.3),
)

cubic_MAIN_PAIRS = [(float(t), float(d)) for t, ds in zip(cubic_TAUS, cubic_DS_BY_TAU) for d in ds]

def cubic_correction_coefficients(x, D, tau):
    G = 1 + 3*tau*x**2
    G1, G2 = 6*tau*x, 6*tau
    f, f1 = -x**3, -3*x**2
    alpha = 3*D**2*G1/G**6.5
    beta = (D*(2*G*f1-7*f*G1)/(2*G**5.5)
            + D**2*(5*G*G2-39*G1**2)/(2*G**7.5))
    chi = (-f*(G*f1-f*G1)/G**4.5
           + D*(39*f*G1**2-6*G*f*G2-16*G*G1*f1)/(4*G**6.5)
           + D**2*(-53*G*G1*G2+138*G1**3)/(4*G**8.5))
    root = np.sqrt(G)
    root1 = G1/(2*root)
    root2 = G2/(2*root)-G1**2/(4*G**1.5)
    return alpha*root, 2*alpha*root1+beta*root, alpha*root2+beta*root1+chi*root


def cubic_b1_coefficients(x, D, tau):
    G, G1 = 1+3*tau*x**2, 6*tau*x
    a, b, c = cubic_correction_coefficients(x, D, tau)
    return tau*a, -D/G**2+tau*b, -x**3/G+D*G1/G**3+tau*c


def cubic_choose_domain(tau, D, edges, factor=1.0):
    if tau <= 0 or D <= 0 or factor < 1:
        raise ValueError('Positive parameters and domain factor >= 1 required.')
    upper = (4*D*cubic_TAIL_EXPONENT)**0.25
    tail_L = brentq(lambda z: z**4/(4*D)+tau*z**6/(2*D)-cubic_TAIL_EXPONENT, 0, upper)
    # Always cover all saved bins, with a small guard outside their boundaries.
    return factor*max(tail_L, 1.05*float(np.max(np.abs(edges))))


def cubic_solve_b1_uniform(tau, D, L, n):
    if n < 5 or n % 2 != 1:
        raise ValueError('Use an odd number of at least five half-line nodes.')
    x = np.linspace(0, L, n)
    h = x[1]-x[0]
    A, B, C = cubic_b1_coefficients(x, D, tau)
    lo = A[1:-1]/h**2-B[1:-1]/(2*h)
    diag = -2*A[1:-1]/h**2+C[1:-1]
    hi = A[1:-1]/h**2+B[1:-1]/(2*h)
    matrix = np.zeros((3, n-2))
    matrix[0, 1:], matrix[1], matrix[2, :-1] = hi[:-1], diag, lo[1:]
    rhs = np.zeros(n-2)
    rhs[0] = -lo[0]
    p = np.r_[1.0, solve_banded((1, 1), matrix, rhs), 0.0]
    signed_mass = 2*simpson(p, x=x)
    if not np.isfinite(signed_mass) or abs(signed_mass) < 1e-12:
        raise ValueError('B1 has a nonfinite/nearly zero signed mass.')
    p /= signed_mass
    terms = np.array([lo*p[:-2], diag*p[1:-1], hi*p[2:]])
    residual = np.max(np.abs(terms.sum(axis=0)))/max(np.max(np.abs(terms).sum(axis=0)), 1e-300)
    # Independent gradient residual excludes the origin and boundary stencils.
    dp = np.gradient(p, x, edge_order=2)
    ddp = np.gradient(dp, x, edge_order=2)
    continuous_terms = np.array([A[3:-3]*ddp[3:-3], B[3:-3]*dp[3:-3], C[3:-3]*p[3:-3]])
    gradient_residual = np.max(np.abs(continuous_terms.sum(axis=0)))/max(np.max(np.abs(continuous_terms).sum(axis=0)), 1e-300)
    return x, p, dict(solver='uniform', n=n, raw_signed_mass=float(signed_mass),
                      solver_residual=float(residual), gradient_residual=float(gradient_residual),
                      boundary_abs=float(abs(p[-1])), origin=float(p[0]), success=True)


def cubic_solve_b1_log(tau, D, L, xmin=1e-8, n=1000):
    r = 1/6+1/(18*D*tau*tau)
    s = np.linspace(np.log(xmin), np.log(L), n)
    x = np.exp(s)
    G = 1+3*tau*x*x
    guess = G*np.exp(-x**4/(4*D)-tau*x**6/(2*D))
    guess_s = x*(6*tau*x/G-x**3*G/D)*guess
    def fun(ss, y):
        xx = np.exp(ss)
        A, B, C = cubic_b1_coefficients(xx, D, tau)
        return np.vstack((y[1], (1-B*xx/A)*y[1]-C*xx*xx/A*y[0]))
    def bc(ya, yb):
        return np.array([ya[0]-1, yb[0]])
    sol = solve_bvp(fun, bc, s, np.vstack((guess, guess_s)), tol=1e-8, max_nodes=30000)
    if not sol.success:
        raise RuntimeError(sol.message)
    ss = np.linspace(s[0], s[-1], 16001)
    xx, yy = np.exp(ss), sol.sol(ss)
    origin = sol.y[0, 0]-sol.y[1, 0]/r
    # Same leading central-mass approximation as Daniel; vary xmin explicitly.
    mass = 2*(simpson(xx*yy[0], x=ss)+origin*xmin)
    if not np.isfinite(mass) or abs(mass) < 1e-12:
        raise ValueError('Log-grid B1 has invalid signed mass.')
    return np.r_[0.0, xx], np.r_[origin, yy[0]]/mass, dict(
        solver='log', n=int(sol.x.size), xmin=xmin, origin=float(origin/mass),
        raw_signed_mass=float(mass), solver_residual=float(np.max(sol.rms_residuals)),
        gradient_residual=np.nan, boundary_abs=float(abs(yy[0, -1]/mass)), success=True)


@lru_cache(maxsize=16)
def cubic_generated_theory(tau, D, L):
    z = sp.Symbol('x', real=True)
    return generate_stationary_correction(-z**3, z, tau=tau, D=D,
                                         integration_bounds=(-L, L))


def cubic_daniel_b3(x, tau, D):
    G, G1, G2 = 1+3*tau*x**2, 6*tau*x, 6*tau
    f, f1 = -x**3, -3*x**2
    p = G*np.exp(-x**4/(4*D)-tau*x**6/(2*D))
    p /= 2*simpson(p, x=x)
    log1 = G1/G+f*G/D
    log2 = G2/G-(G1/G)**2+(f1*G+f*G1)/D
    a, b, c = cubic_correction_coefficients(x, D, tau)
    F = cumulative_simpson(tau*G**2/D*(a*(log2+log1**2)+b*log1+c), x=x, initial=0)
    return p*(1+F-2*simpson(p*F, x=x))


def cubic_check_operator_source(correction, tau, D, L):
    # The symbolic generator source is h=tau*G*L2[p_UCNA,unnormalized].
    # Probe the well-resolved core, so this check never divides tiny tail values.
    x = np.linspace(0, min(1.0, L/2), 17)
    G, G1, G2 = 1+3*tau*x**2, 6*tau*x, 6*tau
    f, f1 = -x**3, -3*x**2
    log1 = G1/G+f*G/D
    log2 = G2/G-(G1/G)**2+(f1*G+f*G1)/D
    a, b, c = cubic_correction_coefficients(x, D, tau)
    lhs = a*(log2+log1**2)+b*log1+c
    variable = next(iter(correction.gamma_expression.free_symbols))
    h = sp.lambdify(variable, correction.source_expression, modules='numpy')(x)
    unnormalized = G*np.exp(-x**4/(4*D)-tau*x**6/(2*D))
    rhs = h/(tau*G*unnormalized)
    error = float(np.max(np.abs(lhs-rhs)/(1+np.abs(rhs))))
    if not np.isfinite(error) or error > 1e-8:
        raise ValueError(f'Coefficient/source mismatch: {error:g}')
    return error


def cubic_raw_path(tau, D, output_dir=None):
    directory = cubic_RAW_DIR if output_dir is None else Path(output_dir)
    return directory / f'stationary_tau_{tau:g}_D_{D:g}.pkl'


def cubic_load_saved(tau, D, output_dir=None):
    path = cubic_raw_path(tau, D, output_dir=output_dir)
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved simulation: {path}; no simulation fallback exists.')
    with path.open('rb') as handle:
        saved = pickle.load(handle)  # Trusted, local repository result.
    cfg = saved['config']
    if not (np.isclose(cfg['tau'], tau) and np.isclose(cfg['D'], D)):
        raise ValueError('Saved parameter metadata do not match filename.')
    edges = np.asarray(saved['edges'], dtype=float)
    counts = np.asarray(saved['replica_counts'], dtype=np.int64)
    endpoints = np.asarray(saved['replica_endpoints'], dtype=float)
    if (counts.ndim != 2 or endpoints.ndim != 2 or counts.shape[0] < 2
            or counts.shape[0] != endpoints.shape[0]):
        raise ValueError('Expected rectangular replica counts and endpoints.')
    if counts.shape[1] != len(edges)-1 or np.any(np.diff(edges) <= 0) or np.any(counts < 0):
        raise ValueError('Invalid saved histogram.')
    finite = np.isfinite(endpoints)
    recomputed = np.asarray([np.histogram(row[np.isfinite(row)], bins=edges)[0] for row in endpoints])
    if not np.array_equal(recomputed, counts):
        raise ValueError('Saved histogram disagrees with saved finite endpoints.')
    n_total, n_finite = endpoints.size, int(finite.sum())
    rho_count = int(np.count_nonzero(finite & (np.sqrt(tau)/(1+3*tau*endpoints**2) >= 1)))
    stat = path.stat()
    provenance = dict(raw_file=str(path.relative_to(cubic_ROOT)), raw_size=stat.st_size,
        raw_mtime_ns=stat.st_mtime_ns, raw_config=cfg,
        N_total=int(n_total), N_finite=n_finite, N_nonfinite=int(n_total-n_finite),
        N_in_histogram=int(counts.sum()), N_outside_finite=int(n_finite-counts.sum()),
        rho_ge_1_fraction_finite=rho_count/n_finite if n_finite else np.nan,
        replica_in_histogram=counts.sum(axis=1).tolist())
    return edges, counts, provenance


def cubic_reflect(x, p):
    return np.r_[-x[:0:-1], x], np.r_[p[:0:-1], p]


def cubic_integrate_bins(x, p, edges):
    """Exact bin integrals of the piecewise-linear density; no clipping."""
    if edges[0] < x[0] or edges[-1] > x[-1]:
        raise ValueError('Solver domain must cover every analysis bin.')
    cdf = cumulative_trapezoid(p, x=x, initial=0)
    index = np.clip(np.searchsorted(x, edges, side='right')-1, 0, len(x)-2)
    dx = edges-x[index]
    slope = (p[index+1]-p[index])/(x[index+1]-x[index])
    primitive = cdf[index]+p[index]*dx+0.5*slope*dx**2
    return np.diff(primitive)


def cubic_source_fingerprint():
    functions = (cubic_correction_coefficients, cubic_b1_coefficients, cubic_solve_b1_uniform,
                 cubic_solve_b1_log, cubic_generated_theory, cubic_daniel_b3, cubic_check_operator_source,
                 generate_stationary_correction)
    # Jupyter records function source in linecache; fail explicitly if unavailable.
    payload = cubic_SOLVER_VERSION + '\n'.join(inspect.getsource(fun) for fun in functions)
    return hashlib.sha256(payload.encode()).hexdigest()


def cubic_bundle_spec(tau, D, edges, n, factor, solver='uniform', xmin=None):
    L = cubic_choose_domain(tau, D, edges, factor)
    return dict(version=cubic_SOLVER_VERSION, source_hash=cubic_source_fingerprint(),
                tau=float(tau), D=float(D), L=float(L), requested_n=int(n),
                domain_factor=float(factor), solver=solver,
                histogram_edges_hash=hashlib.sha256(np.asarray(edges, dtype=float).tobytes()).hexdigest(),
                xmin=None if xmin is None else float(xmin))


def cubic_bundle_path(spec, cache_dir=None):
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    pair_dir = (cubic_CACHE_DIR if cache_dir is None else Path(cache_dir)) / f"tau_{spec['tau']:g}_D_{spec['D']:g}"
    return pair_dir / f"{spec['solver']}_{digest}.npz"


def cubic_load_or_solve(tau, D, edges, n, factor, *, solver='uniform', xmin=None, allow_solve=False, cache_dir=None):
    spec = cubic_bundle_spec(tau, D, edges, n, factor, solver, xmin)
    path = cubic_bundle_path(spec, cache_dir=cache_dir)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['spec'] != spec:
                raise ValueError('Numerical cache signature mismatch.')
            bundle = {key: data[key].copy() for key in ('x', 'B1', 'B3', 'UCNA', 'B3_midpoint_weights')}
        bundle.update(metadata=meta, cache_file=str(path.relative_to(cubic_ROOT)))
        return bundle
    if not allow_solve:
        return None
    start = time.perf_counter()
    if solver == 'uniform':
        x, p, diagnostics = cubic_solve_b1_uniform(tau, D, spec['L'], n)
    elif solver == 'log':
        x, p, diagnostics = cubic_solve_b1_log(tau, D, spec['L'], xmin=xmin, n=n)
    else:
        raise ValueError('Unknown solver.')
    correction = cubic_generated_theory(tau, D, spec['L'])
    b3, ucna = np.asarray(correction(x)), np.asarray(correction.ucna_density(x))
    midpoint_weights = np.asarray(correction((edges[:-1]+edges[1:])/2))*np.diff(edges)
    # Numerical B3 equivalence check is itself grid-dependent; inspect in pilot.
    check = cubic_daniel_b3(x, tau, D)
    diagnostics.update(
        operator_source_check_error=cubic_check_operator_source(correction, tau, D, spec['L']),
        B3_generator_check_max=float(np.max(np.abs(check-b3))),
        B3_generator_check_L1=float(2*simpson(np.abs(check-b3), x=x)),
        elapsed_seconds=float(time.perf_counter()-start))
    meta = dict(spec=spec, diagnostics=diagnostics)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: existing numerical caches are never overwritten.
    with path.open('xb') as handle:
        np.savez_compressed(handle, x=x, B1=p, B3=b3, UCNA=ucna, B3_midpoint_weights=midpoint_weights,
                            metadata=json.dumps(meta, sort_keys=True))
    return dict(x=x, B1=p, B3=b3, UCNA=ucna, B3_midpoint_weights=midpoint_weights, metadata=meta,
                cache_file=str(path.relative_to(cubic_ROOT)))


def cubic_analyse_bundle(bundle, edges, counts, provenance):
    meta = bundle['metadata']
    spec, solver_diag = meta['spec'], meta['diagnostics']
    tau, D = spec['tau'], spec['D']
    xx, _ = cubic_reflect(bundle['x'], bundle['B1'])
    curves = {name: cubic_reflect(bundle['x'], bundle[name])[1] for name in ('UCNA', 'B3', 'B1')}
    weights = {name: cubic_integrate_bins(xx, p, edges) for name, p in curves.items()}
    common = np.logical_and.reduce([np.isfinite(w) & (w > 0) for w in weights.values()])
    total = int(counts.sum())
    retained = int(counts[:, common].sum())
    excluded = 1-retained/total if total else np.nan
    common_counts = counts[:, common]
    can_condition = common.sum() >= 2 and retained > 0 and np.all(common_counts.sum(axis=1) > 0)
    rows = []
    conditional_theory = {}
    for name, p in curves.items():
        w = weights[name]
        mass = float(np.trapezoid(p, x=xx))
        negative = float(np.trapezoid(np.maximum(-p, 0), x=xx))
        bin_negative = float(-w[w < 0].sum())
        individual_excluded = float(counts[:, ~(np.isfinite(w) & (w > 0))].sum()/total) if total else np.nan
        normalized_bin_negative = bin_negative/w.sum() if w.sum() > 0 else np.inf
        historical_valid = (normalized_bin_negative <= cubic_NEGATIVE_MASS_THRESHOLD
                            and individual_excluded <= cubic_EXCLUDED_EMPIRICAL_MASS_THRESHOLD)
        full_valid = (np.all(np.isfinite(p)) and np.min(p) >= 0
                      and np.all(np.isfinite(w)) and np.all(w > 0)
                      and provenance['N_nonfinite'] == 0 and total > 0
                      and np.all(counts.sum(axis=1) > 0)
                      and solver_diag['solver_residual'] <= cubic_RESIDUAL_TOL)
        row = dict(tau=tau, D=D, tau2D=D*tau**2, Approximation=name,
                   solver=spec['solver'], requested_n=spec['requested_n'], L=spec['L'],
                   domain_factor=spec['domain_factor'], xmin=spec['xmin'],
                   cache_file=bundle['cache_file'], analysis_version=cubic_ANALYSIS_VERSION,
                   density_mass=mass, density_min=float(np.min(p)), negative_mass=negative,
                   binned_negative_mass=bin_negative, histogram_theory_mass=float(w.sum()),
                   histogram_normalized_negative_mass=normalized_bin_negative,
                   individual_excluded_empirical_fraction=individual_excluded,
                   theory_mass_outside_histogram=float(mass-w.sum()),
                   full_histogram_valid=bool(full_valid), historical_threshold_valid=bool(historical_valid),
                   common_positive_bins=int(common.sum()), common_N_used=retained,
                   common_excluded_empirical_fraction=float(excluded),
                   common_excluded_signed_theory_mass=float(w[~common].sum()),
                   common_excluded_absolute_theory_mass=float(np.abs(w[~common]).sum()),
                   conditional_replica_N=common_counts.sum(axis=1).tolist(),
                   status='full_histogram_valid' if full_valid else 'diagnostic_only',
                   **{key: value for key, value in solver_diag.items() if key not in ('solver', 'xmin')},
                   **provenance)
        if full_valid:
            row.update(summarize_kl(counts, w/w.sum(), measure=cubic_KL_MEASURE))
        else:
            row.update(KL=np.nan, KL_uncertainty=np.nan, KL_measure=cubic_KL_MEASURE)
        if can_condition:
            q = w[common]/w[common].sum()
            conditional_theory[name] = q
            summary = summarize_kl(common_counts, q, measure=cubic_KL_MEASURE)
            row.update({'conditional_'+key: value for key, value in summary.items()})
        rows.append(row)
    pair = dict(tau=tau, D=D, tau2D=D*tau**2, solver=spec['solver'],
                requested_n=spec['requested_n'], L=spec['L'], domain_factor=spec['domain_factor'],
                xmin=spec['xmin'], common_positive_bins=int(common.sum()), common_N_used=retained,
                common_excluded_empirical_fraction=float(excluded), cache_file=bundle['cache_file'])
    pair['half_L1_B1_B3'] = float(0.5*np.trapezoid(np.abs(curves['B1']-curves['B3']), x=xx))
    density_masses = [np.trapezoid(curves[name], x=xx) for name in ('B1', 'B3')]
    pair['distance_kind'] = ('total_variation_approx' if
        all(np.min(curves[name]) >= 0 for name in ('B1', 'B3'))
        and np.allclose(density_masses, 1, atol=cubic_ABS_OBSERVABLE_TOL, rtol=0)
        else 'signed_density_half_L1')
    if can_condition:
        ratio = np.log(conditional_theory['B3']/conditional_theory['B1'])
        values = common_counts @ ratio/common_counts.sum(axis=1)
        raw_delta = float(common_counts.sum(axis=0) @ ratio/retained)
        raw_sem = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else np.nan
        scale = retained/(common.sum()-1) if cubic_KL_MEASURE == 'rescaled' else 1.0
        pair.update(paired_KL_difference_raw=raw_delta, paired_KL_difference=scale*raw_delta,
                    paired_raw_SEM=raw_sem, paired_SEM=scale*raw_sem, KL_measure=cubic_KL_MEASURE)
        # Compare midpoint and integrated B3 using exactly the same extra support.
        mid = bundle['B3_midpoint_weights']
        comparison_support = common & (mid > 0) & np.isfinite(mid)
        pair['midpoint_comparison_bins'] = int(comparison_support.sum())
        cc = counts[:, comparison_support]
        if comparison_support.sum() >= 2 and np.all(cc.sum(axis=1) > 0):
            integrated = weights['B3'][comparison_support]
            midpoint = mid[comparison_support]
            q_int, q_mid = integrated/integrated.sum(), midpoint/midpoint.sum()
            int_summary = summarize_kl(cc, q_int, measure=cubic_KL_MEASURE)
            mid_summary = summarize_kl(cc, q_mid, measure=cubic_KL_MEASURE)
            pair.update(B3_integrated_minus_midpoint_KL=float(int_summary['KL']-mid_summary['KL']),
                        B3_integrated_midpoint_probability_L1=float(np.abs(q_int-q_mid).sum()),
                        midpoint_comparison_N=int(cc.sum()))
    return rows, pair, weights


def cubic_execute_case(tau, D, n, factor, solver='uniform', xmin=None, allow_solve=False):
    edges, counts, provenance = cubic_load_saved(tau, D)
    bundle = cubic_load_or_solve(tau, D, edges, n, factor, solver=solver, xmin=xmin, allow_solve=allow_solve)
    if bundle is None:
        return None
    rows, pair, weights = cubic_analyse_bundle(bundle, edges, counts, provenance)
    return dict(bundle=bundle, rows=rows, pair=pair, weights=weights, edges=edges)


def cubic_record_failure(tau, D, n, factor, solver, xmin, exc):
    return dict(tau=tau, D=D, requested_n=n, domain_factor=factor,
                solver=solver, xmin=xmin, status='failed', error=f'{type(exc).__name__}: {exc}')


def cubic_save_tables(prefix, rows, pairs, failures):
    cubic_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # These are derived summaries; raw simulations and density caches stay untouched.
    pd.DataFrame(rows).to_csv(cubic_CACHE_DIR / f'{prefix}_{cubic_KL_MEASURE}_summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(cubic_CACHE_DIR / f'{prefix}_{cubic_KL_MEASURE}_pairs.csv', index=False)
    pd.DataFrame(failures).to_csv(cubic_CACHE_DIR / f'{prefix}_{cubic_KL_MEASURE}_failures.csv', index=False)


def cubic_rebuild_comparison(raw_dir, *, measure='bias_corrected', allow_solve=False):
    """Reanalyze saved simulations; missing stationary solves require explicit opt-in."""
    global cubic_ROOT, cubic_RAW_DIR, cubic_CACHE_DIR, cubic_KL_MEASURE
    cubic_ROOT = Path.cwd()
    cubic_RAW_DIR = Path(raw_dir).resolve()
    cubic_CACHE_DIR = cubic_RAW_DIR / 'b1_vs_b3'
    cubic_KL_MEASURE = measure
    production_results, rows = [], []
    # Six additional pairs complete the appendix heatmap.
    pairs = cubic_MAIN_PAIRS + [(1., 3.), (2., 2.), (2., 3.), (3., 1.), (3., 2.), (3., 3.)]
    for tau, D in pairs:
        result = cubic_execute_case(tau, D, cubic_PRODUCTION_N, cubic_PRODUCTION_DOMAIN_FACTOR,
                              allow_solve=allow_solve)
        if result is None:
            raise RuntimeError('Missing B1 cache: enable RUN_B1_SOLVES to compute stationary theory.')
        production_results.append(result)
        rows.extend(result['rows'])
    production_table = pd.DataFrame(rows)
    heatmap_table = production_table.copy()
    paper_figure_2b_results = production_table.copy()
    from tqdm.auto import tqdm
    paper_cbfpe_rows = []
    paper_cbfpe_inputs = production_results if True else []
    for result in tqdm(paper_cbfpe_inputs, desc='cBFPE on saved grids (no solves)',
                       disable=not True):
        spec = result['bundle']['metadata']['spec']
        tau, D = spec['tau'], spec['D']
        edges, counts, provenance = cubic_load_saved(tau, D)
        old = production_table[(production_table.tau == tau)
                               & (production_table.D == D)].iloc[0]
        if any(old[key] != provenance[key] for key in ('raw_size', 'raw_mtime_ns')):
            raise RuntimeError('Saved simulation data changed; rerun the production analysis/load cell.')
        x = result['bundle']['x']
        # Same effective diffusion as cubic_rebuttal; its x=0 limit is D.
        nonzero = x != 0
        product = np.zeros_like(x)
        product[nonzero] = (np.sqrt(2*tau)*x[nonzero]
                           * dawsn(1/(np.sqrt(2*tau)*x[nonzero])))
        diffusion = D*(1 + 3*tau*x**2*(product-1))
        row = dict(tau=tau, D=D, tau2D=D*tau**2, Approximation='cBFPE',
                   full_histogram_valid=False, KL=np.nan, KL_uncertainty=np.nan,
                   KL_measure=cubic_KL_MEASURE, status='diagnostic_only')
        if np.all(np.isfinite(diffusion)) and np.all(diffusion > 0):
            exponent = cumulative_trapezoid(-x**3/diffusion, x, initial=0)
            log_density = exponent - np.log(diffusion)
            density = np.exp(log_density - log_density.max())
            density /= 2*simpson(density, x=x)
            full_x, full_density = cubic_reflect(x, density)
            weights = cubic_integrate_bins(full_x, full_density, edges)
            valid = (np.all(np.isfinite(weights)) and np.all(weights > 0)
                     and provenance['N_nonfinite'] == 0
                     and np.all(counts.sum(axis=1) > 0))
            if valid:
                row.update(summarize_kl(counts, weights, measure=cubic_KL_MEASURE),
                           full_histogram_valid=True, status='full_histogram_valid')
        paper_cbfpe_rows.append(row)

    if paper_cbfpe_rows:
        paper_figure_2b_results = pd.concat(
            [paper_figure_2b_results, pd.DataFrame(paper_cbfpe_rows)], ignore_index=True)

    main = paper_figure_2b_results.apply(lambda r: (r.tau, r.D) in cubic_MAIN_PAIRS, axis=1)
    return paper_figure_2b_results.loc[main].copy(), heatmap_table


# Logistic B1/B3 stationary analysis.
# Model-specific globals are prefixed to keep both workflows independent.
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import pickle
import types

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import simpson
from scipy.linalg import solve_banded
from tqdm.auto import tqdm


logistic_ROOT = Path(__file__).resolve().parent
logistic_DATA_DIR = logistic_ROOT / 'results/logistic_stationary'
logistic_CACHE_DIR = logistic_DATA_DIR / 'b1_vs_b3'
logistic_TAUS = [0.2, 0.5, 1., 2., 5., 10.]
logistic_D = 2.
logistic_N_GRID = 8001
logistic_DOMAIN_PAD = 1.
logistic_RETRY_FAILED = False
logistic_NEGATIVE_MASS_THRESHOLD = 1e-3
logistic_EXCLUDED_MASS_THRESHOLD = 1e-3
logistic_KL_MEASURE = 'bias_corrected'
logistic_STYLES = {'UCNA': {}, 'B3': {}, 'B1': {}}

def logistic_correction_coefficients(x, D, tau):
    ex = np.exp(x)
    G, G1 = 1+2*tau*ex, 2*tau*ex
    G2, G3 = G1, G1
    f, f1 = 2*(1-ex), -2*ex
    alpha = 3*D**2*G1/G**6.5
    beta = (D*(2*G*f1-7*f*G1)/(2*G**5.5)
            + D**2*(5*G*G2-39*G1**2)/(2*G**7.5))
    chi = (-f*(G*f1-f*G1)/G**4.5
           + D*(39*f*G1**2-6*G*f*G2-16*G*G1*f1)/(4*G**6.5)
           + D**2*(2*G**2*G3-53*G*G1*G2+138*G1**3)/(4*G**8.5))
    root = np.sqrt(G)
    root1 = G1/(2*root)
    root2 = G2/(2*root)-G1**2/(4*G**1.5)
    return alpha*root, 2*alpha*root1+beta*root, alpha*root2+beta*root1+chi*root


def logistic_b1_coefficients(x, D, tau):
    G, G1 = 1+2*tau*np.exp(x), 2*tau*np.exp(x)
    a, b, c = logistic_correction_coefficients(x, D, tau)
    return tau*a, -D/G**2+tau*b, 2*(1-np.exp(x))/G+D*G1/G**3+tau*c


def logistic_solve_linear_bvp(x, A, B, C):
    h = x[1]-x[0]
    if len(x) < 7 or not np.allclose(np.diff(x), h) or h <= 0:
        raise ValueError('A uniform increasing grid with at least seven points is required.')
    lo = A[1:-1]/h**2-B[1:-1]/(2*h)
    diag = -2*A[1:-1]/h**2+C[1:-1]
    hi = A[1:-1]/h**2+B[1:-1]/(2*h)
    # Row scaling improves conditioning without changing the equation.
    scale = np.maximum.reduce([abs(lo), abs(diag), abs(hi)])
    if np.any(~np.isfinite(scale)) or np.any(scale == 0):
        raise ValueError('Invalid differential-equation coefficients.')
    lo, diag, hi = lo/scale, diag/scale, hi/scale
    matrix = np.zeros((3, len(x)-2))
    matrix[0, 1:], matrix[1], matrix[2, :-1] = hi[:-1], diag, lo[1:]
    rhs = np.zeros(len(x)-2)
    rhs[0] = -lo[0]
    p = np.r_[1., solve_banded((1, 1), matrix, rhs), 0.]
    if np.any(~np.isfinite(p)):
        raise ValueError('B1 has nonfinite values.')
    mass = simpson(p, x=x)
    absolute_mass = simpson(np.abs(p), x=x)
    if (not np.isfinite(mass) or not np.isfinite(absolute_mass)
            or absolute_mass <= 0 or abs(mass) <= 1e-12*absolute_mass):
        raise ValueError('B1 has nonfinite or nearly cancelling signed mass.')
    # The left amplitude is arbitrary: a negative integral is not a solve failure.
    # Normalize without clipping; analyze_case diagnoses the remaining negative tail.
    p /= mass
    terms = np.array([lo*p[:-2], diag*p[1:-1], hi*p[2:]])
    residual = np.max(abs(terms.sum(axis=0)))/max(np.max(abs(terms).sum(axis=0)), 1e-300)
    dp = np.gradient(p, x, edge_order=2)
    ddp = np.gradient(dp, x, edge_order=2)
    terms = np.array([A*ddp, B*dp, C*p])[:, 3:-3]
    gradient_residual = np.max(abs(terms.sum(axis=0)))/max(np.max(abs(terms).sum(axis=0)), 1e-300)
    return p, dict(solver_residual=float(residual), gradient_residual=float(gradient_residual),
                   normalized_left_density=float(p[0]), raw_signed_mass=float(mass),
                   raw_absolute_mass=float(absolute_mass), normalization_sign_reversed=bool(mass < 0))


@lru_cache(maxsize=12)
def logistic_generated_theory(tau, D):
    x = sp.Symbol('x', real=True)
    return generate_stationary_correction(2*(1-sp.exp(x)), x, tau=float(tau), D=float(D))


def logistic_integrate_bins(x, p, log_edges):
    if log_edges[0] < x[0] or log_edges[-1] > x[-1]:
        raise ValueError('Solver grid must cover all histogram edges.')
    # Local integrals avoid subtraction of almost equal tail CDF values.
    weights = []
    for left, right in zip(log_edges[:-1], log_edges[1:]):
        xx = np.r_[left, x[(x > left) & (x < right)], right]
        weights.append(np.trapezoid(np.interp(xx, x, p), x=xx))
    return np.asarray(weights)

def logistic_load_saved(tau):
    filename = logistic_DATA_DIR/f'stationary_tau_{tau:g}_D_{logistic_D:g}.pkl'
    stat = filename.stat()
    return logistic__load_saved(str(filename), stat.st_size, stat.st_mtime_ns, float(tau), float(logistic_D))


@lru_cache(maxsize=12)
def logistic__load_saved(filename, size, mtime, tau, D):
    with open(filename, 'rb') as handle:
        saved = pickle.load(handle)  # trusted local simulation outputs only
    config = saved['config']
    if not np.isclose(config['tau'], tau) or not np.isclose(config['D'], D):
        raise ValueError('Filename and saved parameters disagree.')
    edges = np.asarray(saved['edges'], dtype=float)
    counts = np.asarray(saved['replica_counts'])
    endpoints = np.asarray(saved['replica_endpoints_I'])
    if (endpoints.ndim != 2 or np.any(~np.isfinite(endpoints)) or endpoints.size == 0
            or len(endpoints) != len(counts)):
        raise ValueError('Nonfinite/empty saved endpoints: entire pair excluded.')
    if (np.any(~np.isfinite(edges)) or np.any(edges <= 0) or np.any(np.diff(edges) <= 0)
            or counts.ndim != 2 or counts.shape[1] != len(edges)-1
            or len(counts) < 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0)
            or np.any(counts.sum(axis=1) <= 0)):
        raise ValueError('Invalid edges or replica histograms.')
    provenance = dict(file=filename, size=size, mtime_ns=mtime,
                      N_endpoints=int(endpoints.size), N_histogram=int(counts.sum()),
                      outside_fraction=float(1-counts.sum()/endpoints.size))
    return edges, counts, provenance


def logistic_function_signature(function):
    def pack(c):
        return (c.co_code.hex(), c.co_names,
                tuple(pack(v) if isinstance(v, types.CodeType) else repr(v) for v in c.co_consts))
    return pack(function.__code__)


def logistic_load_or_solve(tau, n=logistic_N_GRID, extra_left=0., extra_right=0., allow_solve=False):
    edges, counts, provenance = logistic_load_saved(tau)
    le = np.log(edges)
    left, right = float(le[0]-logistic_DOMAIN_PAD-extra_left), float(le[-1]+logistic_DOMAIN_PAD+extra_right)
    if n < 7 or left >= le[0] or right <= le[-1]:
        raise ValueError('Invalid grid/domain settings.')
    spec = dict(tau=float(tau), D=float(logistic_D), n=int(n), left=left, right=right,
                provenance=provenance, edges=edges.tolist(),
                implementation=[logistic_function_signature(f) for f in
                    (logistic_correction_coefficients, logistic_b1_coefficients, logistic_solve_linear_bvp, logistic_generated_theory.__wrapped__, logistic_load_or_solve)],
                utils_hash=hashlib.sha256((logistic_ROOT/'ucna_utils.py').read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    filename = logistic_CACHE_DIR/f'tau_{tau:g}_D_{logistic_D:g}_{key}.npz'
    failed = filename.with_suffix('.failed.json')
    if filename.exists():
        with np.load(filename, allow_pickle=False) as data:
            return {k: data[k].copy() for k in ('x', 'UCNA', 'B1', 'B3')}, json.loads(str(data['metadata'])), edges, counts
    if failed.exists() and not logistic_RETRY_FAILED:
        raise RuntimeError('Cached failure: '+json.loads(failed.read_text())['error'])
    if not allow_solve:
        return None
    logistic_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'B1 logistic: tau={tau:g}, D={logistic_D:g}, x in [{left:.3g}, {right:.3g}], n={n}', flush=True)
    try:
        x = np.linspace(left, right, n)
        p, diagnostics = logistic_solve_linear_bvp(x, *logistic_b1_coefficients(x, logistic_D, tau))
        correction = logistic_generated_theory(float(tau), float(logistic_D))
        curves = dict(x=x, UCNA=np.asarray(correction.ucna_density(x)),
                      B1=p, B3=np.asarray(correction(x)))
        if any(np.any(~np.isfinite(v)) for v in curves.values()):
            raise ValueError('Nonfinite stationary density.')
        metadata = dict(spec=spec, diagnostics=diagnostics, cache_file=str(filename))
        temporary = filename.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **curves, metadata=json.dumps(metadata))
        temporary.replace(filename)
        return curves, metadata, edges, counts
    except Exception as exc:
        failed.write_text(json.dumps(dict(spec=spec, error=f'{type(exc).__name__}: {exc}'), indent=2))
        raise

def logistic_analyze_case(case):
    curves, meta, edges, counts = case
    x = curves['x']
    weights = {name: logistic_integrate_bins(x, curves[name], np.log(edges)) for name in logistic_STYLES}
    common = np.logical_and.reduce([w > 0 for w in weights.values()])
    retained = counts[:, common]
    can_condition = common.sum() >= 2 and np.all(retained.sum(axis=1) > 0)
    excluded = float(1-retained.sum()/counts.sum())
    rows = []
    for name, w in weights.items():
        p = curves[name]
        negative = float(np.trapezoid(np.maximum(-p, 0), x=x))
        full_valid = bool(np.min(p) >= 0 and np.all(w >= 0) and w.sum() > 0
                          and not np.any((w <= 0) & (counts.sum(axis=0) > 0)))
        row = dict(tau=meta['spec']['tau'], D=meta['spec']['D'], Approximation=name,
                   negative_mass=negative, negative_mass_threshold=logistic_NEGATIVE_MASS_THRESHOLD,
                   negative_mass_flagged=bool(negative > logistic_NEGATIVE_MASS_THRESHOLD),
                   binned_negative_mass=float(-w[w < 0].sum()),
                   density_min=float(p.min()), density_mass=float(np.trapezoid(p, x=x)),
                   histogram_theory_mass=float(w.sum()), full_histogram_valid=full_valid,
                   common_excluded_fraction=excluded,
                   excluded_mass_flagged=bool(excluded > logistic_EXCLUDED_MASS_THRESHOLD), outside_fraction=meta['spec']['provenance']['outside_fraction'],
                   conditional_reliable=negative <= logistic_NEGATIVE_MASS_THRESHOLD and excluded <= logistic_EXCLUDED_MASS_THRESHOLD,
                   n=meta['spec']['n'], left=meta['spec']['left'], right=meta['spec']['right'],
                   **meta['diagnostics'])
        row.update(summarize_kl(counts, w, logistic_KL_MEASURE) if full_valid else dict(KL=np.nan, KL_uncertainty=np.nan))
        if can_condition:
            row.update({'conditional_'+k: v for k, v in summarize_kl(retained, w[common], logistic_KL_MEASURE).items()})
        else:
            row.update(conditional_KL=np.nan, conditional_KL_uncertainty=np.nan)
        rows.append(row)
    pair = dict(tau=meta['spec']['tau'], common_excluded_fraction=excluded,
                negative_mass_flagged=any(r['negative_mass_flagged'] for r in rows
                                         if r['Approximation'] in ('B1', 'B3')),
                half_L1_B1_B3=float(.5*np.trapezoid(abs(curves['B1']-curves['B3']), x=x)))
    if can_condition:
        q1, q3 = (weights[k][common]/weights[k][common].sum() for k in ('B1', 'B3'))
        ratio = np.log(q3/q1)
        replica = retained@ratio/retained.sum(axis=1)
        scale = retained.sum()/(common.sum()-1) if logistic_KL_MEASURE == 'rescaled' else 1.
        pair.update(difference=float(scale*(retained.sum(axis=0)@ratio)/retained.sum()),
                    SEM=float(scale*np.std(replica, ddof=1)/np.sqrt(len(replica))))
    return rows, pair, weights, common


def logistic_collect_cases(requests, allow_solve):
    cases, failures = {}, []
    for tau, tag, n, dl, dr in tqdm(requests, desc='Logistic B1/B3: solve or load'):
        try:
            case = logistic_load_or_solve(tau, n, dl, dr, allow_solve=allow_solve)
            if case is not None:
                cases[tau, tag] = case
        except (ValueError, RuntimeError, OSError, KeyError, np.linalg.LinAlgError) as exc:
            failures.append(dict(tau=tau, test=tag, error=str(exc)))
            tqdm.write(f'Skipped tau={tau:g}, {tag}: {exc}')
    return cases, pd.DataFrame(failures)

from scipy.integrate import cumulative_trapezoid


def logistic_Dbfpe(x, tau, D):
    """Logistic cBFPE diffusion from the backward-flow integral."""
    values = np.asarray(x, dtype=float)
    intensity = np.exp(values)
    diffusion = np.empty_like(intensity)
    below = intensity <= 1.0
    diffusion[below] = D * (
        1.0 - intensity[below] + intensity[below] / (1.0 + 2.0*tau)
    )
    above = ~below
    if np.any(above):
        current_I = intensity[above]
        u_max = np.log(current_I / (current_I - 1.0)) / (2.0*tau)
        diffusion[above] = D * (
            (1.0-current_I) * (-np.expm1(-u_max))
            + current_I/(1.0+2.0*tau)
              * (-np.expm1(-(1.0+2.0*tau)*u_max))
        )
    if np.any(~np.isfinite(diffusion)) or np.any(diffusion <= 0):
        raise ValueError('logistic cBFPE diffusion is nonpositive or nonfinite')
    return float(diffusion) if values.ndim == 0 else diffusion


def logistic_bfpe_bin_probabilities(tau, D, I_edges, points=65):
    """Integrate the cBFPE stationary density on fixed intensity bins."""
    x_edges = np.log(np.asarray(I_edges, dtype=float))
    pieces = [np.linspace(a, b, points) for a, b in zip(x_edges[:-1], x_edges[1:])]
    x_grid = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    diffusion = logistic_Dbfpe(x_grid, tau, D)
    exponent = cumulative_trapezoid(
        (2*(1-np.exp(x_grid))) / diffusion, x_grid, initial=0.0
    )
    log_density = exponent - np.log(diffusion)
    density = np.exp(log_density - np.max(log_density))
    weights = np.empty(len(I_edges)-1)
    for index in range(len(weights)):
        start = index * (points-1)
        stop = start + points
        integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        weights[index] = integrate(
            density[start:stop], x_grid[start:stop]
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('logistic cBFPE produced invalid bin weights')
    return weights / weights.sum()


def logistic_compare_with_cbfpe(case, points=65):
    # Recompute the common support for all four theories, never mix different supports.
    curves, meta, edges, counts = case
    rows, _, weights, original_support = logistic_analyze_case(case)
    tau, D = meta['spec']['tau'], meta['spec']['D']
    weights['cBFPE'] = logistic_bfpe_bin_probabilities(tau, D, edges, points=points)
    common = np.logical_and.reduce([np.isfinite(w) & (w > 0) for w in weights.values()])
    retained = counts[:, common]
    if common.sum() < 2 or np.any(retained.sum(axis=1) <= 0):
        raise ValueError('No usable common positive-bin support for all four models.')
    excluded = float(1-retained.sum()/counts.sum())
    rows.append(dict(tau=tau, D=D, Approximation='cBFPE', negative_mass=0.,
                     negative_mass_flagged=False, negative_mass_threshold=logistic_NEGATIVE_MASS_THRESHOLD,
                     outside_fraction=meta['spec']['provenance']['outside_fraction']))
    for row in rows:
        row.update({'conditional_'+k: v for k,v in summarize_kl(
            retained, weights[row['Approximation']][common], logistic_KL_MEASURE).items()})
        row.update(KL_measure=logistic_KL_MEASURE, common_excluded_fraction=excluded,
                   excluded_mass_flagged=bool(excluded > logistic_EXCLUDED_MASS_THRESHOLD),
                   conditional_reliable=(row['negative_mass'] <= logistic_NEGATIVE_MASS_THRESHOLD
                                         and excluded <= logistic_EXCLUDED_MASS_THRESHOLD),
                   cbfpe_changed_support=bool(not np.array_equal(common, original_support)))
    return rows


def logistic_comparison_tables(output_dir, measure='bias_corrected', allow_solve=False,
                      negative_mass_threshold=1e-3, excluded_mass_threshold=1e-3):
    """Load/regenerate stationary theory on existing logistic replica histograms."""
    global logistic_DATA_DIR, logistic_CACHE_DIR, logistic_KL_MEASURE
    global logistic_NEGATIVE_MASS_THRESHOLD, logistic_EXCLUDED_MASS_THRESHOLD
    if measure not in {'bias_corrected', 'rescaled'}:
        raise ValueError('Unknown KL measure')
    logistic_DATA_DIR = Path(output_dir).resolve()
    logistic_CACHE_DIR = logistic_DATA_DIR / 'b1_vs_b3'
    logistic_KL_MEASURE = measure
    logistic_NEGATIVE_MASS_THRESHOLD = negative_mass_threshold
    logistic_EXCLUDED_MASS_THRESHOLD = excluded_mass_threshold
    requests = [(tau, 'main', logistic_N_GRID, 0., 0.) for tau in logistic_TAUS]
    cases, failures = logistic_collect_cases(requests, allow_solve)
    if len(cases) != len(logistic_TAUS) or len(failures):
        raise RuntimeError('Missing logistic B1 theory. Set RUN_B1_SOLVES=True '
                           'to generate it from existing raw histograms. '
                           + failures.to_string(index=False))
    rows, pairs, four_rows = [], [], []
    for case in cases.values():
        current, pair, _, _ = logistic_analyze_case(case)
        rows.extend(current)
        pairs.append(pair)
        four_rows.extend(logistic_compare_with_cbfpe(case))
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(four_rows)
