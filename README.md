# FMDV immune_dynamics

## A transmission model of foot-and-model-disease viruses in African buffalo with temporary loss of immunity

**Jan Medlock
[\<medlockj@oregonstate.edu\>](mailto:medlockj@oregonstate.edu),
Ricardo Noé Gerardo Reyes Grimaldo, Erin Gorisch,
Brianna Beechler, Peter Buss, Bryan Charleston, Brian Dugovich,
Simon Gubbins, Anna Jolles, Nick Juleff, Lin-Mari de Klerk-Lorist,
Francois Maree, Eva Perez-Martin, O.L. van Schalkwyk, Katherine Scott,
Fuquan Zhang.**

**Copyright 2014–2025, Jan Medlock et al.  All rights reserved.
Released under the [GNU AGPL 3](LICENSE).**

This repository contains Python code used to simulate and analyze FMDV
transmission in African buffalo for our paper
> Reyes Grimaldo RNG, Buss P, Charleston B, de Klerk-Lorist L-M,
> Gorsich E, Gubbins S, Maree F, Perez-Martin E, van Schalkwyk OL,
> Scott K, Zang F, Dugovich B, Medlock J, Beechler B, Jolles A.
> A novel mechanism for the persistence of a highly contagious
> pathogen: unstable antibody protection against foot-and-mouth
> disease in African buffalo.
> *In preparation*.

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

This code extends a
[previous version](https://github.com/janmedlock/FMDV/tree/master)
to include temporary loss and recovery of immunity.

### Main simulation code

The Python module [herd](herd) simulates the FMDV model.

The submodule [herd.floquet](herd/floquet) contains the solver to find
the population stable age distribution with birth seasonality. In the
folder [herd/floquet](herd/floquet) is an optional faster
implementation in [Cython](https://cython.org/) of
`herd.floquet.monodromy` that can be built using the included
[Makefile](herd/floquet/Makefile).

### Simulation scripts

The scripts ending in `_run.py` run the model simulations.
**Each of these takes many cpu-days to run.**

* [baseline_run.py](baseline_run.py), for each of the 3 SATs, runs
  1,000 simulations using the baseline parameter values. It produces a
  file called `baseline.h5`.

* [acute_run.py](acute_run.py), for each of the 3 SATs, runs 1,000
  simulations with no carrier infections and using the
  baseline values for the other parameters. It produces a file
  called `acute.h5`.

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

* [samples_run.py](samples_run.py), for each of the 3 SATs and for
  each of 20,000 parameter posterior samples, runs 1 simulation. It
  produces a file called `samples.h5`. **This takes a very long
  time to run.**

### Analysis and plotting scripts

These scripts analyze and plot the simulation results. Most of them
require having run the simulation scripts above.

* [R0.py](R0.py) computes the basic reproduction number,
  *R*<sub>0</sub>, for each of the 3 SATs and for each of 20,000
  parameter posterior samples, and reports the median and 95% credible
  interval.

* [baseline_plot.py](baseline_plot.py) analyzes and plots the results
  of the simulations with the baseline parameter values. This requires
  the file `baseline.h5`.

* [acute_plot.py](acute_plot.py) analyzes and plots the results
  of the simulations with no carrier infections. This requires
  the file `acute.h5`.

* [population_size_plot.py](population_size_plot.py) analyzes and
  plots the results of varying the population size. This requires the
  file `population_size.h5`.

* [susceptibility_plot.py](susceptibility_plot.py) analyzes and
  plots the results of varying the susceptibility of the lost-immunity
  state. This requires the file `susceptibility.h5`.

  * [susceptibility_null_plot.py](susceptibility_null_plot.py) analyzes
    and plots the results with zero susceptibility of the
    lost-immunity state to confirm the dynamics are equivalent to the
    model without the lost-immunity class. This requires the file
    `susceptibility.h5`.

* [sensitivity_plot.py](sensitivity_plot.py) plots a summary of the
  persistence for varying either population size or susceptibility of
  the lost-immunity state. This requires the files
  `population_size.h5` and `susceptibility.h5`.

* [population_size_and_susceptibility_plot.py](population_size_and_susceptibility_plot.py)
  analyzes and plots the results of varying both the population size
  and the susceptibility of the lost-immunity state. This requires the
  file `population_size_and_susceptibility.h5`.

* [samples_plot.py](samples_plot.py), from the results of the
  simulations over the parameter posterior samples, calculates the
  PRCC sensitivity of extinction time to the model parameters. This
  requires the file `samples.h5`.

### Other files in the base directory

In the base directory, besides the scripts for simulation, analysis,
and plotting, there are helper Python modules for those scripts. For
example, [baseline.py](baseline.py) has common code used by
`baseline_run.py` and `baseline_plot.py`.

There are also a few metadata files: this [README.md](README.md), a
[Makefile](Makefile) to convert `README.md` to HTML, and the
[LICENSE](LICENSE).

### Test scripts

The [test](test) directory contains some scripts to test various parts
of the model code, along with [context.py](test/context.py) to allow
the test scripts to import from [herd](herd).

* [run_one.py](test/run_one.py) runs 1 model simulation and plots the
  results.

* [run_many.py](test/run_many.py) runs 100 model simulations for one
  SAT and plots the results.

* [samples_run_test.py](test/samples_run_test.py) sequentially runs
  simulations with the parameter posterior samples for one SAT.

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
  more explicit calculations.

* [endemic_equilibrium.py](test/endemic_equilibrium.py) shows the
  hazard of infection and the proportion of newborns immune at the
  endemic equilibrium for each SAT.

* [samples_pdfs_finite.py](test/samples_pdfs_finite.py) ensures that
  for each posterior parameter sample, all of the evaluations in the
  model code of the probability density functions are finite.

* [solver_check.py](test/solver_check.py) checks the consistency of
  the solver for the initial conditions, plots block structure of the
  solver matrix, and plots a solution.

* [h5_check.py](test/h5_check.py) checks the simulation output files for
  consistency.

* [force_of_infection.ipynb](test/force_of_infection.ipynb) is a
  [Jupyter](https://jupyter.org/) notebook that runs one short
  simulation, examines the force of infection over the simulation, and
  examines the force of infection for random samples of initial
  conditions.

### Notes

The [notes](notes) directory contains notes on the model initial
conditions and files used

* [initial_conditions.tex](notes/initial_conditions.tex) is a LaTeX
  file with notes on the solver used for the model initial conditions.

The remaining files in this directory are used by LaTeX to compile the
notes on the initial conditions.
