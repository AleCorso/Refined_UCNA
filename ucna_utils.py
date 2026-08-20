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
_trapezoid = getattr(np, "trapezoid", np.trapz)
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
