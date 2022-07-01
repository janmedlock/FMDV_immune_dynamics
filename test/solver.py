#!/usr/bin/python3

import matplotlib.pyplot
import numpy

from context import herd
from herd.initial_conditions.immune_status import _blocks, _solver


def check_parameters(solver):
    '''Ensure the solver parameters correctly set the matrix and vector
    elements.'''
    solver.update_blocks()
    A = solver.get_A()
    b = solver.get_b()
    n = len(solver.ages)
    assert numpy.isclose(b[0, 0],
                         solver.params.newborn_proportion_immune)
    assert numpy.isclose(b[n, 0],
                         1 - solver.params.newborn_proportion_immune)
    assert numpy.allclose(A[n + 1, n + 1],
                          (1 + ((solver.params.hazard.mortality[0]
                                 + solver.params.hazard.infection)
                                * solver.step / 2)))


def plot_blocks(solver):
    A = solver.get_A()
    n = len(solver.ages)
    m = int(A.shape[0] / n)
    assert A.shape[0] == m * n
    blocks = numpy.empty((m, m), dtype=int)
    for j in range(m):
        J = slice(j * n, (j + 1) * n)
        for k in range(m):
            K = slice(k * n, (k + 1) * n)
            B = A[J, K]
            b_nonzero = B[B.nonzero()]
            if b_nonzero.shape[1] == 0:
                blocks[j, k] = 0
            elif (b_nonzero > 0).any():
                blocks[j, k] = 1
            else:
                blocks[j, k] = -1
    (fig, axes) = matplotlib.pyplot.subplots(constrained_layout=True)
    axes.matshow(blocks, cmap='PiYG')
    vars_ = _blocks.vars_ode + _blocks.vars_pde
    axes.set_xticks(range(len(vars_)))
    axes.set_yticks(range(len(vars_)))
    axes.set_xticklabels(vars_)
    axes.set_yticklabels(vars_)


def plot_solution(prob):
    (fig, axes) = matplotlib.pyplot.subplots(constrained_layout=True)
    axes.stackplot(prob.index, prob.T, labels=prob.columns)
    axes.set_xlabel('age (y)')
    axes.set_ylabel('probability')
    axes.legend()
    return axes


if __name__ == '__main__':
    parameters = herd.Parameters(SAT=1)
    solver = _solver.Solver(parameters)
    HAZARD_INFECTION = 2
    NEWBORN_PROPORTION_IMMUNE = 0.6
    prob = solver.solve_step(solver.transform((HAZARD_INFECTION,
                                               NEWBORN_PROPORTION_IMMUNE)))
    check_parameters(solver)
    plot_blocks(solver)
    plot_solution(prob)
    matplotlib.pyplot.show()
