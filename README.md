# FMDV immune dynamics

## A transmission model of foot-and-model-disease viruses in African buffalo with temporary loss of immunity

**Jan Medlock
[\<medlockj@oregonstate.edu\>](mailto:medlockj@oregonstate.edu),
Ricardo Noé Gerardo Reyes Grimaldo, Erin Gorisch,
Brianna Beechler, Peter Buss, Bryan Charleston, Brian Dugovich,
Simon Gubbins, Anna Jolles, Nick Juleff, Lin-Mari de Klerk-Lorist,
Francois Maree, Eva Perez-Martin, O.L. van Schalkwyk, Katherine Scott,
Fuquan Zhang.**

**Copyright 2014–2026, Jan Medlock et al.  All rights reserved.
Released under the [GNU AGPL 3](LICENSE).**

This repository contains Python code used to estimate parameters,
simulate, and analyze FMDV transmission in African buffalo for our
paper
> Reyes Grimaldo RNG, Buss P, Charleston B, de Klerk-Lorist L-M,
> Gorsich E, Gubbins S, Maree F, Perez-Martin E, van Schalkwyk OL,
> Scott K, Zang F, Dugovich B, Medlock J, Beechler B, Jolles A.
> A novel mechanism for the persistence of a highly contagious
> pathogen: unstable antibody protection against foot-and-mouth
> disease in African buffalo.
> *In preparation*.

This code extends a
[previous version](https://github.com/janmedlock/FMDV/tree/master)
to include temporary loss and recovery of immunity.

The scripts and model code are written in Python, using many
third-party libraries.  Most notably:
[Python](https://www.python.org/),
[NumPy & SciPy](https://www.scipy.org/),
[statsmodels](https://www.statsmodels.org/),
[pandas](https://pandas.pydata.org/),
[PyTables](https://www.pytables.org/),
[Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers/),
[matplotlib](https://matplotlib.org/),
& [Seaborn](https://seaborn.pydata.org/).

[PDM](https://pdm-project.org/) can use the included
[pyproject.toml](pyproject.toml) and [pdm.lock](pdm.lock) to generate
a virtual environment with the Python and library versions that we
used to write and run the code.

### Statistical estimation

#### Data

The directory [data](data) contains the data files, along with some
Python code.

* [data/antibodies.csv](data/antibodies.csv) contains the antibodies
  titers for FMDV serotypes SAT1, SAT2, and SAT3 for 106 buffalo over
  16 captures.

* [data/captures.csv](data/captures.csv) contains the dates of the
  captures.

* [data/animals.csv](data/animals.csv) contains the sex and age in
  years at first capture for the buffalo.

* [data/data_table.py](data/data_table.py) is a script to make a
  summary table of the data.

* [data/antibody_distribution_plot.py](data/antibody_distribution_plot.py)
  is a script to make a plot the distributions of antibody titers.

* [data/seropositives_test.py](data/seropositives_test.py) is a script
  to compare titers by SAT for seropositives.

* [data/seronegative_report.py](data/seronegative_report.py) is a
  script to compare the proportion of buffalo aged 2+ years that are
  seronegative by SAT.

* [data/data_.py](data/data_.py) contains code to load the data.

#### Inference

The directory [inference](inference) contains Python code to estimate the
rates of loss and gain of antibodies.

* [inference/estimate.py](inference/estimate.py) is a script that
  performs the estimation of the rates and plots the results.

* [inference/variation_plot.py](inference/variation_plot.py) is a
  script that plots the variation between strains in estimated rates,
  titer for seropositives, and proportion seronegative by age.

* [inference/model.py](inference/model.py) implements the
  log-likelihood for the 2-state continuous-time Markov-chain model of
  the antibody data.

The other files in this directory contain the code for estimation and
plotting.

### Simulation

The Python module [herd](herd) simulates the FMDV model.

The submodule [herd.floquet](herd/floquet) contains the solver to find
the population stable age distribution with birth seasonality. In the
folder [herd/floquet](herd/floquet) is an optional faster
implementation in [Cython](https://cython.org/) of
`herd.floquet.monodromy` that can be built using the included
[Makefile](herd/floquet/Makefile).

#### Simulation scripts

The scripts ending in `_run.py` run the model simulations.
**Most of these takes many cpu-days to run.**

* [simulation_run.py](simulation_run.py), for each of the 3 SATs, runs
  one simulation using the baseline parameter values. It produces a
  file called `simulation.h5`.

* [baseline_run.py](baseline_run.py), for each of the 3 SATs, runs
  1,000 simulations using the baseline parameter values. It produces a
  file called `baseline.h5`.

* [population_size_run.py](population_size_run.py), for each of the 3
  SATs and for 14 different population sizes, runs 1,000 simulations
  using the baseline values for the other parameters. It produces a
  file called `population_size.h5`.

* [susceptibility_run.py](susceptibility_run.py), for each of the 3
  SATs and for 11 different values for susceptibility of the
  lost-immunity state, runs 1,000 simulations using the baseline
  values for the other parameters. It produces a file called
  `susceptibility.h5`.

* [population_size_and_susceptibility_run.py](population_size_and_susceptibility_run.py),
 for each of the 3 SATs, for 14 different population sizes, and for 11
 different values for susceptibility of the lost-immunity state, runs
 1,000 simulations using the baseline values for the other
 parameters. It produces a file called
 `population_size_and_susceptibility.h5`. **This takes a very long
 time to run.**

#### Analysis and plotting scripts

These scripts analyze and plot the simulation results. Most of them
require having run the simulation scripts above.

* [mortality_and_birth_rates_plot.py](mortality_and_birth_rates_plot.py)
  plots the mortality and birth rates.

* [initial_conditions_plot.py](initial_conditions_plot.py) plots the
  simulation initial conditions.

* [simulation_plot.py](simulation_plot.py) plots the single simulation
  for each serotype with the baseline parameter values. This requires
  the file `simulation.h5`.

* [baseline_plot.py](baseline_plot.py) analyzes and plots the results
  of the simulations with the baseline parameter values. This requires
  the file `baseline.h5`.

* [population_size_plot.py](population_size_plot.py) analyzes and
  plots the results of varying the population size. This requires the
  file `population_size.h5`.

* [susceptibility_plot.py](susceptibility_plot.py) analyzes and
  plots the results of varying the susceptibility of the lost-immunity
  state. This requires the file `susceptibility.h5`.

* [sensitivity_plot.py](sensitivity_plot.py) plots a summary of the
  persistence for varying either population size or susceptibility of
  the lost-immunity state. This requires the files
  `population_size.h5` and `susceptibility.h5`.

* [population_size_and_susceptibility_plot.py](population_size_and_susceptibility_plot.py)
  analyzes and plots the results of varying both the population size
  and the susceptibility of the lost-immunity state. This requires the
  file `population_size_and_susceptibility.h5`.

#### Other Python files in the base directory

In the base directory, besides the scripts for simulation, analysis,
and plotting, there are helper Python modules for those scripts.

* [common.py](common.py) contains common functions and other
  definitions.

* [plotting.py](plotting.py) contains common functions and other
  definitions for plotting.

* [h5.py](h5.py) implements reading and writing the HDF5 files used for
  storing simulation output.

* [sensitivity.py](sensitivity.py) has common plotting code used by
  [population_size_plot.py](population_size_plot.py),
  [susceptibility_plot.py](susceptibility_plot.py), and
  [sensitivity_plot.py](sensitivity_plot.py).

* [simulation.py](simulation.py) has common code used by
  [simulation_run.py](simulation_run.py) and
  [simulation_plot.py](simulation_plot.py).

* [baseline.py](baseline.py) has common code used by
  [baseline_run.py](baseline_run.py) and
  [baseline_plot.py](baseline_plot.py).

* [population_size.py](population_size.py) has common code used by
  [population_size_run.py](population_size_run.py) and
  [population_size_plot.py](population_size_plot.py).

* [susceptibility.py](susceptibility.py) has common code used by
  [susceptibility_run.py](susceptibility_run.py) and
  [susceptibility_plot.py](susceptibility_plot.py).

* [population_size_and_susceptibility.py](population_size_and_susceptibility.py)
  has common code used by
  [population_size_and_susceptibility_run.py](population_size_and_susceptibility_run.py)
  and
  [population_size_and_susceptibility_plot.py](population_size_and_susceptibility_plot.py).

#### Test scripts

The [test](test) directory contains some scripts to test various parts
of the model code, along with [context.py](test/context.py) to allow
the test scripts to import from [herd](herd).

* [run_one.py](test/run_one.py) runs 1 model simulation and plots the
  results.

* [run_many.py](test/run_many.py) runs 100 model simulations for one
  SAT and plots the results.

* [age_structure.py](test/age_structure.py) and
  [age_structure_3d.py](test/age_structure_3d.py) plot the stable age
  structure of the model buffalo population.

* [floquet_benchmark.py](test/floquet_benchmark.py) times the Floquet
  solver in [herd.floquet](herd/floquet) for finding the stable age
  structure.

* [initial_conditions.py](test/initial_conditions.py) plots the model
  initial conditions.

* [initial_conditions_aggregate.py](test/initial_conditions_aggregate.py)
  plots the initial conditions, similar to
  [initial_conditions.py](test/initial_conditions.py), but using
  more-explicit calculations.

* [endemic_equilibrium.py](test/endemic_equilibrium.py) shows the
  hazard of infection and the proportion of newborns immune at the
  endemic equilibrium for each SAT.

* [solver_check.py](test/solver_check.py) checks the consistency of
  the solver for the initial conditions, plots block structure of the
  solver matrix, and plots a solution.

* [force_of_infection.ipynb](test/force_of_infection.ipynb) is a
  [Jupyter](https://jupyter.org/) notebook that runs one short
  simulation, examines the force of infection over the simulation, and
  examines the force of infection for random samples of initial
  conditions.

#### Metadata files in the base directory

* [pyproject.toml](pyproject.toml) describes the project, including
  the versions of libraries used directly.

	* [pdm.lock](pdm.lock) describes the versions of all libraries
      used, including those used indirectly.

* [LICENSE](LICENSE).

* [README.md](README.md), i.e. this file.

  * [Makefile](Makefile) to convert `README.md` to HTML.
