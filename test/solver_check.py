#!/usr/bin/env python
'''Check the consistency of the solver for the initial conditions,
plots block structure of the solver matrix, and plots a solution.'''

import matplotlib
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot
import numpy

from _context import common, herd
import herd.initial_conditions.test


def check_parameters(slvr):
    '''Ensure the solver parameters correctly set the matrix and
    vector elements.'''
    slvr.update_blocks()
    M = slvr.get_M()
    b = slvr.get_b()
    n = len(slvr.ages)
    birth_maternal_immunity_proportion = (
        slvr.block_params.birth_maternal_immunity_proportion)
    assert numpy.isclose(b[0, 0], birth_maternal_immunity_proportion)
    assert numpy.isclose(b[n, 0], 1 - birth_maternal_immunity_proportion)
    assert numpy.allclose(M[n + 1, n + 1],
                          (1 + ((slvr.block_params.hazard.mortality[0]
                                 + slvr.block_params.hazard.infection)
                                * slvr.step / 2)))


def plot_blocks(slvr):
    M = slvr.get_M()
    n = len(slvr.ages)
    m = int(M.shape[0] / n)
    assert M.shape[0] == m * n
    blcks = numpy.empty((m, m), dtype=int)
    for j in range(m):
        J = slice(j * n, (j + 1) * n)
        for k in range(m):
            K = slice(k * n, (k + 1) * n)
            M_JK = M[J, K]
            if M_JK.nnz == 0:
                blcks[j, k] = 0
            else:
                m_jk_nonzero = M_JK[M_JK.nonzero()]
                if (m_jk_nonzero > 0).any():
                    blcks[j, k] = 1
                else:
                    blcks[j, k] = -1
    vars_ = (
        herd.initial_conditions.test.vars_ode
        + herd.initial_conditions.test.vars_pde
    )
    order = [
        vars_.index(var)
        for var in herd.initial_conditions.test.SHORT_NAMES.values()
    ]
    blcks = blcks[order][:, order]
    vars_ = [vars_[i] for i in order]
    (_, axes) = matplotlib.pyplot.subplots(constrained_layout=True)
    cmap = matplotlib.colormaps['PiYG']
    norm = matplotlib.colors.Normalize(-1, 1)
    axes.matshow(blcks, cmap=cmap, norm=norm)
    axes.set_xticks(range(m))
    axes.set_yticks(range(m))
    axes.set_xticklabels(vars_)
    axes.set_yticklabels(vars_)
    axes.legend(
        handles=(
            matplotlib.patches.Patch(
                color=cmap(norm(1)),
                label='Some $m_{jk} > 0$'
            ),
            matplotlib.patches.Patch(
                color=cmap(norm(-1)),
                label=r'Some $m_{jk} < 0$ and all $m_{jk} \leq 0$'
            ),
            matplotlib.patches.Patch(
                color=cmap(norm(0)),
                label='All $m_{jk} = 0$'
            ),
        ),
    )
    return axes


def plot_solution(prob):
    (_, axes) = matplotlib.pyplot.subplots(constrained_layout=True)
    axes.stackplot(prob.index, prob.T, labels=prob.columns)
    axes.set_xlabel(f'age ({common.TIME_UNIT})')
    axes.set_ylabel('constant-birth\ndensity')
    axes.legend()
    return axes


if __name__ == '__main__':
    parameters = herd.Parameters(SAT=3)
    slvr = herd.initial_conditions.test.Solver(parameters)
    HAZARD_INFECTION = 2
    NEWBORN_PROPORTION_IMMUNE = 0.6
    prob = slvr.solve_step(slvr.transform((HAZARD_INFECTION,
                                           NEWBORN_PROPORTION_IMMUNE)))
    check_parameters(slvr)
    plot_blocks(slvr)
    plot_solution(prob)
    matplotlib.pyplot.show()
