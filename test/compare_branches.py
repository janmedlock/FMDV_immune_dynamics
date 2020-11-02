#!/usr/bin/python3

from operator import attrgetter
import pickle
import subprocess
import sys

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas

sys.path.append('..')
from herd import mortality, Parameters
from herd.initial_conditions import immune_status
sys.path.pop()


SATS = (1, 2, 3)
BRANCHES = ('unconditional', 'conditional')


class Solution:
    def __init__(self, branch, SAT):
        self.branch = branch
        self.SAT = SAT
        solver = immune_status.Solver(Parameters(SAT=self.SAT))
        P = self.solve(solver)
        if branch == 'unconditional':
            self.P_unconditional = P
            self.P_conditional = P.divide(solver.params.survival.mortality,
                                          axis='index')
        elif branch == 'conditional':
            self.P_conditional = P
            self.P_unconditional = P.multiply(solver.params.survival.mortality,
                                              axis='index')
        else:
            raise ValueError(f'Unknown {branch=}!')
        self.hazard_infection = solver.get_hazard_infection(P)
        self.newborn_proportion_immune = (
            solver.get_newborn_proportion_immune(P))

    def solve(self, solver,
              hazard_infection=10, newborn_proportion_immune=0.8):
        return solver.solve_step(solver.transform((hazard_infection,
                                                   newborn_proportion_immune)))


def get_branch():
    cp = subprocess.run(('git', 'branch', '--show-current'),
                        capture_output=True, check=True)
    branch = cp.stdout.decode().strip()
    if branch == 'immune_dynamics':
        return 'unconditional'
    elif branch == 'immune_dynamics_conditional_probability':
        return 'conditional'
    else:
        raise ValueError(f'Unknown {branch=}!')


def build():
    branch = get_branch()
    solution = {SAT: Solution(branch, SAT)
                for SAT in SATS}
    with open(f'solution_{branch}.pkl', 'wb') as fd:
        pickle.dump(solution, fd, protocol=-1)
    return solution


def load():
    solutions = {}
    for branch in BRANCHES:
        with open(f'solution_{branch}.pkl', 'rb') as fd:
            solutions[branch] = pickle.load(fd)
    return solutions


def isclose(a, b, rtol=1e-3, atol=1e-3, *args, **kwds):
    '''Relax the default tolerances from `rtol=1e-5` and `atol=1e-8`.'''
    return numpy.isclose(a, b, rtol=rtol, atol=atol, *args, **kwds)


def allclose(a, b, *args, **kwds):
    return numpy.all(isclose(a, b, *args, **kwds))


def plot_solutions(solutions, which):
    get_P = attrgetter(f'P_{which}')
    P = {branch: pandas.concat({SAT: get_P(solutions[branch][SAT])
                                for SAT in SATS},
                               axis='columns')
         for branch in BRANCHES}
    err = P['unconditional'] - P['conditional']
    immune_states = err.columns.levels[1]
    X = range(len(immune_states))
    ages = err.index
    cmap = 'PiYG'
    # Put the middle of the colormap at 0.
    norm = matplotlib.colors.TwoSlopeNorm(0, err.min().min(), err.max().max())
    (fig, axes) = matplotlib.pyplot.subplots(len(SATS), 1,
                                             sharex=True,
                                             constrained_layout=True)
    for (ax, SAT) in zip(axes, SATS):
        ax.pcolormesh(X, ages, err[SAT], cmap=cmap, norm=norm,
                      shading='nearest')
        ax.set_ylabel(f'SAT {SAT}')
    axes[-1].set_xticks(X)
    axes[-1].set_xticklabels(immune_states.str.replace(' ', '\n'))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap),
                 ax=axes, label='error')


def check_solutions(solutions):
    # Check that the solutions from the two branches agree,
    # in both unconditional and conditional form.
    for SAT in SATS:
        for which in BRANCHES:
            get_P = attrgetter(f'P_{which}')
            check = allclose(*(get_P(solutions[branch][SAT])
                               for branch in BRANCHES))
            assert check, f'{which=}, {SAT=} failed!'
    # Check that the sums over the immune states is 1 for all ages
    # for the conditional form.
    for branch in BRANCHES:
        for SAT in SATS:
            check = allclose(
                solutions[branch][SAT].P_conditional.sum(axis='columns'),
                1)
            assert check, f'{branch=}, {SAT=} failed!'
    # Check that the hazards of infection agree.
    for SAT in SATS:
        check = isclose(*(solutions[branch][SAT].hazard_infection
                          for branch in BRANCHES))
        assert check, f'{SAT=} failed!'
    # Check that the newborn proportions immune agree.
    for SAT in SATS:
        check = isclose(*(solutions[branch][SAT].newborn_proportion_immune
                          for branch in BRANCHES))
        assert check, f'{SAT=} failed!'


if __name__ == '__main__':
    # solution = build()

    solutions = load()
    plot_solutions(solutions, 'unconditional')
    check_solutions(solutions)

    matplotlib.pyplot.show()
