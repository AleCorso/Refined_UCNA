# Refined UCNA: code and data

Run these notebooks from this directory to reproduce the manuscript figures
from compact CSV data. The default `USE_PRECOMPUTED_PLOT_DATA=True` requires no
raw simulations, numerical solves, or external repository files. All simulation
and solve switches are disabled by default.

## Contents

| File | Contents |
| --- | --- |
| `cubic_stationary.ipynb` | Original density comparison; UCNA/B1/B3/cBFPE divergence curves with the UCNA inset; three-panel appendix heatmap |
| `logistic_stationary.ipynb` | Original intensity-density comparison; UCNA/B1/B3/cBFPE divergence curves on common positive support |
| `cubic_dynamics.ipynb` | Gaussian-preparation dynamics comparing UCNA and LLA |
| `ucna_utils.py` | All simulation, discretization, symbolic-correction and analysis routines, including both models' B1/B3 solvers |
| `Refined_UCNA_time-scale_separation.nb` | Mathematica derivation; not required by the Python notebooks |

`plot_data/` contains the density, divergence, uncertainty and provenance tables.
The cubic additions are `cubic_b1_kl.csv` and `cubic_heatmap.csv`; the logistic
additions are `logistic_b1_b3.csv`, `logistic_b1_b3_cbfpe.csv` and
`logistic_b1_b3_pairs.csv`. The original `cubic_density.csv`, `cubic_kl.csv`,
`cubic_inset_pairs.csv`, `cubic_inset_curve.csv`, `cubic_inset_exclusions.csv`,
`logistic_density.csv`, `logistic_kl.csv` and `cubic_dynamics.csv` are retained.
Rendered copies in `figures/` are optional. Raw endpoints, trajectories and
numerical caches are not distributed.

The single utility file groups the shared routines and the cubic/logistic B1
routines in separate sections. Model-specific names use `cubic_` and `logistic_`
prefixes; notebook regeneration imports `cubic_rebuild_comparison` and
`logistic_comparison_tables` from this file.

## Figure data

The cubic density figure retains the single correction curve for clarity. Each
configuration, $(\tau,D)=(0.1,2)$ and $(0.2,0.3)$, uses one million production
endpoints plus nine million cached visual-comparison endpoints, with 60 bins on
$[-2,2]$. Density normalization and error bars use the actual in-range count.
B1 and B3 are shown separately in the divergence figure, whose inset axis
annotations can be edited through `PAPER_INSET_LABELS`.

The completed cubic B1/B3 sweep and all six upper-right extension configurations
are included. On the appendix's $6\times6$ grid, UCNA and B1 each have 36 valid
cells; B3 has 26. Invalid full-histogram estimates remain blank and leave gaps
in the curves. The heatmap always uses bias-corrected values, with logarithmic
colors from $10^{-5}$ to $10^{-1}$; smaller values, including negatives, share
the lowest color. Hatching compares each correction's original signed estimate
with UCNA. Palette and annotation controls remain in the plotting cell.

The logistic divergence figure uses the intersection of positive bins shared
by UCNA, B1, B3 and cBFPE. These are conditional comparisons, not full-distribution
KL values when support is excluded. Tables retain negative-mass, excluded-sample
and validity diagnostics, as well as the B1/B3 paired comparison. B1 uses the
same finite-domain boundary prescription as the research notebook; the packaged
results do not establish positivity or infinite-domain uniqueness.

Exploratory timing pilots, full-domain B1 checks and convergence-study figures
are excluded from these concise publication notebooks.

## Statistic and uncertainty

`KL_MEASURE='bias_corrected'` selects

\[
d_{\mathrm{KL}}=\sum_i\hat p_i\log(\hat p_i/q_i)-\frac{M-1}{2N}.
\]

Natural logarithms give nats. $N$ and $M$ are the actual retained sample and bin
counts on the declared support. `KL_MEASURE='rescaled'` selects the legacy
$N D_{\mathrm{KL}}/(M-1)$ statistic from the same data. Inset pairs are converted
before averaging equal $D\tau^2$, using each pair's own counts. The three
quarantined inset pairs remain recorded in `cubic_inset_exclusions.csv`.

Central estimates use pooled counts; bands use replica-KL SEM, without the
legacy multiplier in bias-corrected mode. SEM is not an exact confidence
interval for the pooled estimator. The leading bias correction assumes a
sufficiently sampled fixed partition; finite negative estimates are retained.

The dynamics figure reports peak excess above the late-time plateau. A constant
bias subtraction cancels in this difference for fixed $N,M$. Its cached PDE
solutions use `pde_dt=0.01` and `pde_points=401`, matching the current defaults.
The PDE starts from the sampled histogram, so the usual independent-theory bias
argument does not establish an unbiased transient estimate. Existing density
clipping and renormalization are unchanged.

## Running and regenerating

Install Python 3 dependencies, start Jupyter here, and run a notebook from top
to bottom:

```bash
python -m pip install 'numpy>=2' 'scipy>=1.12' pandas matplotlib seaborn numba sympy tqdm jupyter
jupyter lab
```

Mathematica is needed only to inspect or evaluate the `.nb` derivation.

For raw regeneration, set `USE_PRECOMPUTED_PLOT_DATA=False` and enable only the
required switches. Configurations and random seeds are explicit in the code.

| Notebook | Generation switches |
| --- | --- |
| `cubic_stationary.ipynb` | `RUN_SIMULATIONS`, `RUN_FIGURE_2A_SIMULATIONS`, `RUN_INSET_SIMULATIONS`; `RUN_B1_SOLVES` for missing stationary theory |
| `logistic_stationary.ipynb` | `RUN_SIMULATIONS`, `RUN_VISUAL_SIMULATIONS`; `RUN_B1_SOLVES` for missing stationary theory |
| `cubic_dynamics.ipynb` | `RUN_SIMULATIONS`, `ALLOW_PDE_SOLVES`; retain `RUN_ANALYSIS=True` |

Raw results and theory caches are written below `results/`. Compatible saved
replicas are reused. The cubic inset is an independent, expensive sweep; B1
switches enable stationary solves rather than stochastic simulations. Dynamics
cache replay checks recorded PDE settings; `USE_CACHED_ANALYSIS_SETTINGS=True`
explicitly selects cached settings when they differ from the requested ones.
