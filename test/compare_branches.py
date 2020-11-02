#!/usr/bin/python3

import operator
import pickle
import subprocess
import sys

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas

sys.path.append('..')
import herd
import herd.initial_conditions.immune_status
sys.path.pop()


SATS = (1, 2, 3)
BRANCHES = ('unconditional', 'conditional')


class Solution:
    def __init__(self, branch, SAT):
        self.branch = branch
        self.SAT = SAT
        solver = herd.initial_conditions.immune_status.Solver(
            herd.Parameters(SAT=self.SAT))
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
    '''Get the branch of the solver from git.'''
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


def plot_solutions(solutions, which):
    get_P = operator.attrgetter(f'P_{which}')
    P = {branch: pandas.concat({SAT: get_P(solutions[branch][SAT])
                                for SAT in SATS},
                               axis='columns')
         for branch in BRANCHES}
    # Error between the solutions from the two solvers.
    err = P['unconditional'] - P['conditional']
    immune_states = err.columns.levels[1]
    X = range(len(immune_states))
    ages = err.index
    cmap = 'PiYG'
    # Use the same norm for all the subplots and put the middle of the
    # colormap at 0.
    norm = matplotlib.colors.TwoSlopeNorm(0,
                                          err.values.min(),
                                          err.values.max())
    (fig, axes) = matplotlib.pyplot.subplots(len(SATS),
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


def check_solutions(solutions, rtol=1e-3, atol=1e-3):
    '''Compare the solutions from the two solver branches.
    The arguments `rtol` and `atol` set the tolerances for
    `numpy.isclose()` and `numpy.allclose()` from their defaults of
    `rtol=1e-5` and `atol=1e-8`.'''
    # Some of the tests below only work with two branches.
    assert len(BRANCHES) == 2
    for SAT in SATS:
        # Check that the solutions from the two branches agree, in
        # both unconditional and conditional form.
        for which in BRANCHES:
            get_P = operator.attrgetter(f'P_{which}')
            P = (get_P(solutions[branch][SAT])
                 for branch in BRANCHES)
            P_are_close = numpy.allclose(*P, rtol=rtol, atol=atol)
            assert P_are_close, f'{SAT=}, {which=} failed!'
        # Check that the sums over the immune states is 1 for all ages
        # for the conditional form.
        for branch in BRANCHES:
            P_conditional = solutions[branch][SAT].P_conditional
            sums_to_one = numpy.allclose(P_conditional.sum(axis='columns'), 1,
                                         rtol=rtol, atol=atol)
            assert sums_to_one, f'{SAT=}, {branch=} failed!'
        # Check that the hazards of infection and newborn proportions
        # immune from the two branches agree.
        for stat in ('hazard_infection', 'newborn_proportion_immune'):
            get_stat = operator.attrgetter(stat)
            stats = (get_stat(solutions[branch][SAT])
                     for branch in BRANCHES)
            stats_are_close = numpy.isclose(*stats, rtol=rtol, atol=atol)
            assert stats_are_close, f'{SAT=}, {stat=} failed!'


if __name__ == '__main__':
    # solution = build()

    solutions = load()
    plot_solutions(solutions, 'unconditional')
    check_solutions(solutions)

    matplotlib.pyplot.show()
