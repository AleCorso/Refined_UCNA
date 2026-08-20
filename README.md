# UCNA figure-reproduction code

This directory contains the code and compact, plot-ready numerical data needed
to reproduce the stationary and time-dependent figures accompanying the
manuscript. The Python notebooks also retain the complete simulation and
analysis workflow, with explicit configurations and random seeds.

## Contents

- `cubic_stationary.ipynb`: stationary cubic-model density and
  Kullback--Leibler-divergence figures.
- `logistic_stationary.ipynb`: stationary logistic-model figures in the
  intensity variable.
- `cubic_dynamics.ipynb`: Gaussian-preparation dynamics figure comparing UCNA
  and LLA.
- `ucna_utils.py`: shared simulation, discretization, analysis, and symbolic
  correction utilities used by the notebooks.
- `Refined_UCNA_time-scale_separation.nb`: Mathematica derivation of the
  time-scale-separation correction. It is included for derivational
  transparency but is not required to run the Python notebooks; the correction
  callables are generated directly in Python.
- `plot_data/`: compact CSV tables containing only the numerical values and
  uncertainties displayed in the publication figures.

Raw trajectories, endpoints, quench results, and convergence-study data are not
included. The seven CSV files in `plot_data/` total less than 100 kB.

## Python requirements

Use a recent Python 3 installation with Jupyter and the following packages:

```text
numpy
scipy
pandas
matplotlib
seaborn
numba
sympy
tqdm
jupyter
```

For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pandas matplotlib seaborn numba sympy tqdm jupyter
jupyter lab
```

Wolfram Mathematica is needed only to inspect or evaluate the `.nb` derivation.

## Running the notebooks

Start Jupyter from this directory so that `ucna_utils.py` is importable and the
relative result paths resolve consistently. Then open a notebook and run it
from top to bottom.

By default, `USE_PRECOMPUTED_PLOT_DATA=True`, so each notebook reads the compact
CSV tables and reproduces its figures without loading or generating raw
simulation data.

To regenerate the results from the stochastic simulations, set
`USE_PRECOMPUTED_PLOT_DATA=False` and enable the relevant switches near the
beginning of each notebook:

| Notebook | Switches for generating data |
| --- | --- |
| `cubic_stationary.ipynb` | `RUN_SIMULATIONS`, `RUN_FIGURE_2A_SIMULATIONS`, and `RUN_INSET_SIMULATIONS` |
| `logistic_stationary.ipynb` | `RUN_SIMULATIONS` and `RUN_VISUAL_SIMULATIONS` |
| `cubic_dynamics.ipynb` | `RUN_SIMULATIONS`; leave `RUN_ANALYSIS=True` |

The large cubic inset is a separate, particularly expensive sweep. It can be
enabled independently with `RUN_INSET_SIMULATIONS=True`. Progress bars report
the expected completion time while simulations run.

New raw data are written below `results/`, which is created only when the raw
workflow is used. Existing compatible replica files are reused, and increasing
the requested sample count runs only the missing replicas. The dynamics raw
workflow similarly creates and reuses
`results/cubic_dynamics/analysis_cache.pkl`.

## Reproducibility scope

This directory includes the source code and the processed numerical values used
in the publication figures. The raw stochastic data are deliberately omitted
to keep the archive small; they can be regenerated from the documented
configurations and seeds. The directory does not yet include a locked package
environment. For long-term archival reproducibility, record the exact package
versions in a `requirements.txt` or `environment.yml` alongside this folder.
