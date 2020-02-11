#!/usr/bin/python3
'''Understand Solver.get_row_I(), especially A_IE.
It seems like there's a mismatch between the PDE for p_E
using h_progression(r^{j - 1/2}) and the boundary condition
for p_I(a, 0) using h_progression(r^j).
What about using h^{j - 1/2} * (p^{i, j} + p^{i - 1, j - 1}) / 2
for the integral? Doesn't that give conservation of mass?

Choose \Delta a so that all the hazards have
h(r) ≤ 2 / \Delta a for all 0 ≤ a ≤ 5 \Delta a.
(Remember to check the midpoints, too.)
For each p_X, choose r_max_X ≤ a_max
so that h(r_max_X) ≤ 2 / \Delta a.
\Delta a = 0.002 seems to work for progression.'''


import sys
import time

import numpy
import pandas
from scipy import sparse

sys.path.append('..')
from herd import Parameters, RandomVariables
sys.path.pop()


def arange(start, stop, step, dtype=None):
    '''Like `numpy.arange()`, but `stop` is included in the result.'''
    retval = numpy.arange(start, stop, step, dtype=dtype)
    if stop not in retval:
        retval = numpy.hstack([retval, stop])
    return retval


class Solver:
    '''Crank–Nicolson solver to find the probability of being in each
    compartment as a function of age.'''

    def __init__(self, hazard_infection, RVs, age_max, age_step):
        self.hazard_infection = hazard_infection
        self.RVs = RVs
        self.age_max = age_max
        self.age_step = age_step
        self.ages = arange(0, self.age_max, self.age_step)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[:-1] + self.ages[1:]) / 2

    def get_k(self, i, j):
        '''Get the position `k` in the stacked vector represenation
        p_X^k \approx p_X(a^i, r^j) of the entry for age a^i,
        residence time r^j.'''
        # Convert iterables to arrays so the arithmetic works on them.
        (i, j) = map(numpy.asarray, (i, j))
        assert (i > 0).all()
        assert (i < self.I).all()
        assert (j >= 0).all()
        assert (j < i).all()
        return i * (i - 1) // 2 + j

    @property
    def I(self):
        '''The number of age values, i = 0, 1, ..., I - 1.'''
        return len(self.ages)

    @property
    def K(self):
        '''The number of entries in
        the stacked vector representation p_X^k \approx p_X(a^i, r^j),
        k = 0, 1, ..., K - 1.'''
        # The last entry is
        # K - 1 = self.get_k(self.I - 1, self.I - 2),
        # so
        # K = self.get_k(self.I - 1, self.I - 2) + 1.
        return self.get_k(self.I - 1, self.I - 2) + 1

    def get_T(self):
        '''Get the matrix `T` so that `T.dot(f)`
        approximates the integral
        \int_0^a f(a, r) dr
        using the composite trapezoid rule.
        The result is a I × K matrix that
        when left-multiplied with a K vector
        produces a vector over ages a.'''
        T = sparse.lil_matrix((self.I, self.K))
        for i in range(1, self.I):
            T[i, self.get_k(i, range(i))] = self.age_step
        return T.tocsr()

    def clip(self, rate):
        # This ensures that d_1 ≤ 0 in `get_A_XX()` so that P_X ≥ 0.
        return numpy.clip(rate, 0, 2 / self.age_step)

    def get_A_XX(self, n, rate_out):
        '''Get the n × n diagonal block `A_XX` that maps state X to itself.'''
        # The values on the subdiagonal.
        d_1 = - ((2 - rate_out * self.age_step)
                 / (2 + rate_out * self.age_step))
        assert (d_1 <= 0).all()
        if n == self.I:
            A_XX = sparse.diags([1, d_1], [0, -1], shape=(n, n))
        elif n == self.K:
            A_XX = sparse.lil_matrix((self.K, self.K))
            for i in range(1, self.I):
                for j in range(i):
                    k = self.get_k(i, j)
                    A_XX[k, k] = 1
                    if j > 0:
                        A_XX[k, self.get_k(i - 1, j - 1)] = d_1[j - 1]
        else:
            raise NotImplementedError(f'\'n\'={n}!')
        return A_XX

    def get_A_XY(self, n, rate_in, rate_out):
        '''Get the n × n off-diagonal block `A_XY` that maps state Y to X.'''
        v = - ((rate_in * self.age_step)
               / (2 + rate_out * self.age_step))
        # The values on the diagonal.
        d_0 = numpy.hstack([0, v])
        # The values on the subdiagonal.
        d_1 = v
        if n == self.I:
            A_XY = sparse.diags([d_0, d_1], [0, -1], shape=(n, n))
        else:
            raise NotImplementedError(f'\'n\'={n}!')
        return A_XY

    @staticmethod
    def get_b_X(n, data=[], row_ind=[]):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # There is only 1 column, so any data must be in that column.
        col_ind = [0] * len(row_ind)
        b_X = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, 1))
        return b_X

    def get_row_M(self, rate_out):
        n = self.I
        A_MM = self.get_A_XX(n, rate_out)
        A_M = [A_MM, None, None, None]
        # b_M = [1, 0, 0, ..., 0].
        b_M = self.get_b_X(n, [1], [0])
        return (A_M, b_M)

    def get_row_S(self, rate_in, rate_out):
        n = self.I
        A_SM = self.get_A_XY(n, rate_in, rate_out)
        A_SS = self.get_A_XX(n, rate_out)
        A_S = [A_SM, A_SS, None, None]
        # b_S = [0, 0, ..., 0].
        b_S = self.get_b_X(n)
        return (A_S, b_S)

    def get_row_E(self, rate_in, rate_out):
        n = self.K
        A_ES = sparse.lil_matrix((self.K, self.I))
        for i in range(1, self.I):
            A_ES[self.get_k(i, 0), [i - 1, i]] = - rate_in[i - 1] / 2
        A_EE = self.get_A_XX(n, rate_out)
        A_E = [None, A_ES, A_EE, None]
        # b_E = [0, 0, ..., 0].
        b_E = self.get_b_X(n)
        return (A_E, b_E)

    def get_row_I(self, rate_in, rate_out):
        n = self.K
        A_IE = sparse.lil_matrix((self.K, self.K))
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            for j in range(1, i):
                l = [self.get_k(i - 1, j - 1),
                     self.get_k(i, j)]
                A_IE[k, l] = - rate_in[j - 1] / 2 * self.age_step
        # For consistency of the initial condition,
        # the first row must be all 0.
        assert (A_IE[0].count_nonzero() == 0)
        A_II = self.get_A_XX(n, rate_out)
        A_I = [None, None, A_IE, A_II]
        # b_I = [0, 0, ..., 0].
        b_I = self.get_b_X(n)
        return (A_I, b_I)

    def get_rates(self):
        with numpy.errstate(divide='ignore'):
            rates = {
                'maternal_immunity_waning':
                self.RVs.maternal_immunity_waning.hazard(self.ages_mid),
                'infection': self.hazard_infection,
                'progression': self.RVs.progression.hazard(self.ages_mid),
                'recovery': 0}
        for (k, v) in rates.items():
            if numpy.isscalar(v):
                rates[k] = v * numpy.ones(self.I - 1)
        return numpy.rec.fromarrays(rates.values(),
                                    names=list(rates.keys()))

    def get_A_b(self, format='csr'):
        rates = self.get_rates()
        (A_M, b_M) = self.get_row_M(rates.maternal_immunity_waning)
        (A_S, b_S) = self.get_row_S(rates.maternal_immunity_waning,
                                    rates.infection)
        (A_E, b_E) = self.get_row_E(rates.infection,
                                    rates.progression)
        (A_I, b_I) = self.get_row_I(rates.progression,
                                    rates.recovery)
        A = sparse.bmat([A_M, A_S, A_E, A_I],
                        format=format)
        b = sparse.vstack([b_M, b_S, b_E, b_I],
                          format=format)
        return (A, b)

    def solve(self):
        t0 = time.time()
        (A, b) = self.get_A_b()
        t1 = time.time()
        print(f'Setup took {t1 - t0} seconds.')
        assert numpy.isfinite(A.data).all()
        t2 = time.time()
        Pp = sparse.linalg.spsolve(A, b)
        t3 = time.time()
        print(f'Solve took {t3 - t2} seconds.')
        i_split = 2 * self.I
        [P, p] = [Pp[:i_split], Pp[i_split:]]
        [P_M, P_S] = numpy.hsplit(P, 2)
        [p_E, p_I] = numpy.hsplit(p, 2)
        T = self.get_T()
        [P_E, P_I] = map(T.dot, [p_E, p_I])
        rows = pandas.Index(self.ages, name='age')
        P = pandas.DataFrame({'maternal immunity': P_M,
                              'susceptible': P_S,
                              'exposed': P_E,
                              'infectious': P_I},
                             index=rows)
        return P


def plot_prob_cond(ax, status_prob_cond):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    ages = status_prob_cond.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(status_prob_cond.items()):
        total += v
        z = status_prob_cond.shape[1] - i
        ax.fill_between(ages, total, label=k, zorder=z)
    ax.set_ylabel('probability\ngiven age')
    ax.set_ylim(-0.05, 1.05)


def plot_prob(ax, status_prob):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    ages = status_prob.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(status_prob.items()):
        total += v
        z = len(status_prob) - i
        ax.fill_between(ages, total, label=None, zorder=z)
    ax.set_ylabel('joint\ndensity')


def plot_sample(ax, status_ages, width=0.1):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    age_max = max(max(ages) for ages in status_ages.values()) + width
    left = numpy.arange(0, age_max, width)
    bins = numpy.hstack((left, age_max))
    bottom = numpy.zeros_like(left)
    for (status, ages) in status_ages.items():
        height, _ = numpy.histogram(ages, bins=bins)
        ax.bar(left, height, width, bottom, label=None, align='edge')
        bottom += height
    ax.set_ylabel('number in\nsample')


# Eventually, I want the `plot()` interface to look like this:
# def plot(parameters, ages):
def plot(status_prob_cond):
    from matplotlib import pyplot
    # RVs = RandomVariables(parameters)
    # ICs = RVs.initial_conditions
    # fig, axes = pyplot.subplots(3, 1, sharex=True)
    fig, axes = pyplot.subplots(1, 1, sharex=True)
    axes = [axes]  # make `axes` a 1-d iterable with `pyplot.subplots(1, 1)`.
    # status_prob_cond = ICs._proportion(ages)
    plot_prob_cond(axes[0], status_prob_cond)
    # status_prob = ICs.pdf(ages)
    # plot_prob(axes[1], status_prob)
    # numpy.random.seed(1) # Make `ICs.rvs()` cache friendly.
    # status_ages = ICs.rvs(parameters.population_size)
    # plot_sample(axes[2], status_ages)
    axes[-1].set_xlabel('age', labelpad=-9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (_, labels) = axes[0].get_legend_handles_labels()
    nrow = 2
    ncol = (len(labels) + nrow - 1) // nrow
    fig.legend(loc='lower center', ncol=ncol)
    pyplot.show()


if __name__ == '__main__':
    hazard_infection = 1
    parameters = Parameters()
    # Incremental tests.
    # No M -> S.
    # parameters.maternal_immunity_duration_mean = numpy.inf
    # No S -> E.
    # hazard_infection = 0
    # No E -> I.
    # parameters.progression_mean = numpy.inf
    # Slower progression.
    parameters.progression_mean = 1
    RVs = RandomVariables(parameters, _initial_conditions=False)
    age_max = 10
    age_step = 0.1
    solver = Solver(hazard_infection, RVs, age_max, age_step)
    P = solver.solve()
    plot(P)
