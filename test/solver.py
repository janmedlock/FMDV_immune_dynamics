#!/usr/bin/python3

import matplotlib.pyplot
import numpy

from context import herd
from herd.initial_conditions.immune_status import blocks, solver


def check_parameters(slvr):
    '''Ensure the solver parameters correctly set the matrix and vector
    elements.'''
    slvr.update_blocks()
    A = slvr.get_A()
    b = slvr.get_b()
    n = len(slvr.ages)
    assert numpy.isclose(b[0, 0],
                         slvr.params.newborn_proportion_immune)
    assert numpy.isclose(b[n, 0],
                         1 - slvr.params.newborn_proportion_immune)
    assert numpy.allclose(A[n + 1, n + 1],
                          (1 + ((slvr.params.hazard.mortality[0]
                                 + slvr.params.hazard.infection)
                                * slvr.step / 2)))


def plot_blocks(slvr):
    A = slvr.get_A()
    n = len(slvr.ages)
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
    vars_ = blocks.vars_ode + blocks.vars_pde
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
    slvr = solver.Solver(parameters)
    HAZARD_INFECTION = 2
    NEWBORN_PROPORTION_IMMUNE = 0.6
    prob = slvr.solve_step(slvr.transform((HAZARD_INFECTION,
                                           NEWBORN_PROPORTION_IMMUNE)))
    check_parameters(slvr)
    plot_blocks(slvr)
    plot_solution(prob)
    matplotlib.pyplot.show()
