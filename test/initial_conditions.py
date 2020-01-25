#!/usr/bin/python3
import sys

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


def call_and_clip(fcn, *args, absmax=1e9, **kwds):
    # Ignore division by zero warnings.
    with numpy.errstate(divide='ignore'):
        val = fcn(*args, **kwds)
    # Clip in place.
    numpy.clip(val, -absmax, absmax, val)
    return val


class Solver:
    # TODO
    # Summarize this class.

    def __init__(self, hazard_infection, RVs, age_max, age_step):
        self.hazard_infection = hazard_infection
        self.RVs = RVs
        self.age_max = age_max
        self.age_step = age_step
        self.ages = arange(0, self.age_max, self.age_step)
        self.ages_mid = (self.ages[:-1] + self.ages[1:]) / 2

    @staticmethod
    def get_k(i, j):
        '''Get the position `k` in the stacked vector represenation
        p_X^k \approx p_X(a^i, r^j) of the entry for age a^i,
        residence time r^j.'''
        # Convert iterables to arrays so the arithmetic works on them.
        (i, j) = map(numpy.asarray, (i, j))
        return i * (i + 1) // 2 + j

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
        # K - 1 = self.get_k(self.I - 1, self.I - 1),
        # so
        # K = self.get_k(self.I - 1, self.I - 1) + 1.
        return self.get_k(self.I - 1, self.I - 1) + 1

    def get_T(self):
        '''Get the matrix `T` so that `T.dot(f)`
        approximates the integral
        \int_0^a f(a, r) dr
        using the sum
        \sum_{j = 0}^i f(a^i, r^j) \delta r.
        The result is a vector over ages a.
        '''
        T = sparse.lil_matrix((self.I, self.K))
        for i in range(self.I):
            T[i, self.get_k(i, range(i + 1))] = self.age_step
        return T.tocsr()

    @staticmethod
    def get_A_XX(rate_out_times_age_step):
        return sparse.diags(
            [numpy.hstack([1, - 2 - rate_out_times_age_step]),
                                2 - rate_out_times_age_step  ],
            [0, -1])

    def get_A_YY(self, rate_out_times_age_step):
        A = sparse.lil_matrix((self.K, self.K))
        # i = 0.
        k = self.get_k(0, 0)
        A[k, k] = 1
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            A[k, k] = - 2
            for j in range(1, i + 1):
                # k = self.get_k(i, j)
                # k_1 = self.get_k(i - 1, j - 1)
                [k, k_1] = self.get_k([i, i - 1], [j, j - 1])
                r = rate_out_times_age_step[j - 1]
                A[k, [k, k_1]] = [- 2 - r, 2 - r]
        return A

    @staticmethod
    def get_A_XZ(rate_times_age_step):
        return sparse.diags(
            [numpy.hstack([0, rate_times_age_step]),
                              rate_times_age_step  ],
            [0, -1])

    @staticmethod
    def get_b_Z(n, data=[], row_ind=[]):
        # There is only 1 column, so any data must be in that column.
        col_ind = [0] * len(row_ind)
        return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, 1))

    def get_row_M(self):
        hazard_waning = self.RVs.maternal_immunity_waning.hazard(self.ages_mid)
        A_MM = self.get_A_XX(hazard_waning * self.age_step)
        A_M = [A_MM, None, None, None]
        b_M = self.get_b_Z(self.I, [1], [0])
        return (A_M, b_M)

    def get_row_S(self):
        hazard_waning = self.RVs.maternal_immunity_waning.hazard(self.ages_mid)
        A_SM = self.get_A_XZ(hazard_waning * self.age_step)
        hazard_infection = (self.hazard_infection
                            * numpy.ones_like(self.ages_mid))
        A_SS = self.get_A_XX(hazard_infection * self.age_step)
        A_S = [A_SM, A_SS, None, None]
        b_S = self.get_b_Z(self.I)
        return (A_S, b_S)

    def get_row_E(self):
        hazard_infection = (self.hazard_infection
                            * numpy.ones_like(self.ages_mid))
        A_ES = sparse.lil_matrix((self.K, self.I))
        # i = 0.
        # A_ES[k, self.get_k(0, 0)] = 0  # No op.
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            h = hazard_infection[i - 1]
            A_ES[k, [i, i - 1]] = h
        hazard_progression = call_and_clip(self.RVs.progression.hazard,
                                           self.ages_mid)
        A_EE = self.get_A_YY(hazard_progression * self.age_step)
        A_E = [None, A_ES, A_EE, None]
        b_E = self.get_b_Z(self.K)
        return (A_E, b_E)

    def get_row_I(self):
        A_IE = sparse.lil_matrix((self.K, self.K))
        # This hazard is at `ages`, not `ages_mid`.
        hazard_progression = call_and_clip(self.RVs.progression.hazard,
                                           self.ages)
        rate = hazard_progression * self.age_step ** 2
        # i = 0.
        # A_IE[self.get_k(0, 0), 0] = 0  # No op.
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            # Trapezoid rule for boundary condition.
            j = [0, i]
            l = self.get_k(i, j)
            A_IE[k, l] = rate[j] / 2
            j = range(1, i)
            l = self.get_k(i, j)
            A_IE[k, l] = rate[j]
            j = [0, i - 1]
            l = self.get_k(i - 1, j)
            A_IE[k, l] = rate[j] / 2
            j = range(1, i - 1)
            l = self.get_k(i - 1, j)
            A_IE[k, l] = rate[j]
        hazard_recovery = numpy.zeros_like(self.ages_mid)
        A_II = self.get_A_YY(hazard_recovery * self.age_step)
        A_I = [None, None, A_IE, A_II]
        b_I = self.get_b_Z(self.K)
        return (A_I, b_I)

    def get_A_b(self, format='csr'):
        (A_M, b_M) = self.get_row_M()
        (A_S, b_S) = self.get_row_S()
        (A_E, b_E) = self.get_row_E()
        (A_I, b_I) = self.get_row_I()
        A = sparse.bmat([A_M, A_S, A_E, A_I],
                        format=format)
        b = sparse.vstack([b_M, b_S, b_E, b_I],
                          format=format)
        return (A, b)

    def solve(self, absmax=1e9):
        (A, b) = self.get_A_b()
        assert numpy.isfinite(A.data).all()
        Pp = sparse.linalg.spsolve(A, b)
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
    ax.set_ylim(-0.05, 1.05)


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
    RVs = RandomVariables(parameters, _initial_conditions=False)
    age_max = 10
    age_step = 0.1
    solver = Solver(hazard_infection, RVs, age_max, age_step)
    P = solver.solve()
    plot(P)
