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
        using the composite trapezoid rule
        \sum_{j = 0}^{i - 1} \frac{f(a^i, r^j) + f(a^i, r^{j + 1})}{2}
                             \delta r.
        The result is a vector over ages a.
        '''
        T = sparse.lil_matrix((self.I, self.K))
        # i = 0.
        # T[0, self.get_k(0, 0)] = 0  # No op.
        for i in range(1, self.I):
            T[i, self.get_k(i, [0, i])] = self.age_step / 2
            T[i, self.get_k(i, range(1, i))] = self.age_step
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
        A_M = [A_MM, None, None, None, None, None, None]
        b_M = self.get_b_Z(self.I, [1], [0])
        return (A_M, b_M)

    def get_row_S(self):
        hazard_waning = self.RVs.maternal_immunity_waning.hazard(self.ages_mid)
        A_SM = self.get_A_XZ(hazard_waning * self.age_step)
        hazard_infection = (self.hazard_infection
                            * numpy.ones_like(self.ages_mid))
        A_SS = self.get_A_XX(hazard_infection * self.age_step)
        A_S = [A_SM, A_SS, None, None, None, None, None]
        b_S = self.get_b_Z(self.I)
        return (A_S, b_S)

    def get_row_R(self):
        # All evaluated at the the initial time.
        hazard_antibody_loss = self.RVs.antibody_loss.hazard(
            self.RVs.antibody_loss.time_min * numpy.ones_like(self.ages_mid))
        A_RR = self.get_A_XX(hazard_antibody_loss * self.age_step)
        hazard_antibody_gain = self.RVs.antibody_gain.hazard(self.ages_mid)
        A_RL = self.get_A_XZ(hazard_antibody_gain * self.age_step)
        A_RI = sparse.lil_matrix((self.I, self.K))
        A_RC = sparse.lil_matrix((self.I, self.K))
        # These hazards are at `ages`, not `ages_mid`.
        with numpy.errstate(divide='ignore'):
            hazard_recovery = self.RVs.recovery.hazard(self.ages)
        hazard_chronic_recovery = self.RVs.chronic_recovery.hazard(self.ages)
        probability_chronic = self.RVs.probability_chronic.probability_chronic
        rate_I = ((1 - probability_chronic) * hazard_recovery
                  * self.age_step ** 2)
        rate_C = hazard_chronic_recovery * self.age_step ** 2
        for i in range(1, self.I):
            j = [0, i]
            k = self.get_k(i, j)
            A_RI[i, k] = rate_I[j] / 2
            A_RC[i, k] = rate_C[j] / 2
            j = range(1, i)
            k = self.get_k(i, j)
            A_RI[i, k] = rate_I[j]
            A_RC[i, k] = rate_C[j]
            j = [0, i - 1]
            k = self.get_k(i - 1, j)
            A_RI[i, k] = rate_I[j] / 2
            A_RC[i, k] = rate_C[j] / 2
            j = range(1, i - 1)
            k = self.get_k(i - 1, j)
            A_RI[i, k] = rate_I[j]
            A_RC[i, k] = rate_C[j]
        A_R = [None, None, A_RR, A_RL, None, A_RI, A_RC]
        b_R = self.get_b_Z(self.I)
        return (A_R, b_R)

    def get_row_L(self):
        # All evaluated at the the initial time.
        hazard_antibody_loss = self.RVs.antibody_loss.hazard(
            self.RVs.antibody_loss.time_min * numpy.ones_like(self.ages_mid))
        A_LR = self.get_A_XZ(hazard_antibody_loss * self.age_step)
        hazard_antibody_gain = self.RVs.antibody_gain.hazard(self.ages_mid)
        A_LL = self.get_A_XX(hazard_antibody_gain * self.age_step)
        A_L = [None, None, A_LR, A_LL, None, None, None]
        b_L = self.get_b_Z(self.I)
        return (A_L, b_L)

    def get_row_E(self):
        hazard_infection = (self.hazard_infection
                            * numpy.ones_like(self.ages_mid))
        A_ES = sparse.lil_matrix((self.K, self.I))
        A_EL = sparse.lil_matrix((self.K, self.I))
        # i = 0.
        # A_ES[k, self.get_k(0, 0)] = 0  # No op.
        # A_EL[k, self.get_k(0, 0)] = 0  # No op.
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            h = hazard_infection[i - 1]
            A_ES[k, [i, i - 1]] = h
            A_EL[k, [i, i - 1]] = h
        with numpy.errstate(divide='ignore'):
            hazard_progression = self.RVs.progression.hazard(self.ages_mid)
        A_EE = self.get_A_YY(hazard_progression * self.age_step)
        A_E = [None, A_ES, None, A_EL, A_EE, None, None]
        b_E = self.get_b_Z(self.K)
        return (A_E, b_E)

    def get_row_I(self):
        A_IE = sparse.lil_matrix((self.K, self.K))
        # This hazard is at `ages`, not `ages_mid`.
        with numpy.errstate(divide='ignore'):
            hazard_progression = self.RVs.progression.hazard(self.ages)
        rate = hazard_progression * self.age_step ** 2
        # A_IE[self.get_k(0, 0), 0] = 0  # No op.
        for i in range(1, self.I):
            k = self.get_k(i, 0)
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
        with numpy.errstate(divide='ignore'):
            hazard_recovery = self.RVs.recovery.hazard(self.ages_mid)
        A_II = self.get_A_YY(hazard_recovery * self.age_step)
        A_I = [None, None, None, None, A_IE, A_II, None]
        b_I = self.get_b_Z(self.K)
        return (A_I, b_I)

    def get_row_C(self):
        A_CI = sparse.lil_matrix((self.K, self.K))
        # This hazard is at `ages`, not `ages_mid`.
        with numpy.errstate(divide='ignore'):
            hazard_recovery = self.RVs.recovery.hazard(self.ages)
        probability_chronic = self.RVs.probability_chronic.probability_chronic
        rate = probability_chronic * hazard_recovery * self.age_step ** 2
        # A_CI[self.get_k(0, 0), 0] = 0  # No op.
        for i in range(1, self.I):
            k = self.get_k(i, 0)
            j = [0, i]
            l = self.get_k(i, j)
            A_CI[k, l] = rate[j] / 2
            j = range(1, i)
            l = self.get_k(i, j)
            A_CI[k, l] = rate[j]
            j = [0, i - 1]
            l = self.get_k(i - 1, j)
            A_CI[k, l] = rate[j] / 2
            j = range(1, i - 1)
            l = self.get_k(i - 1, j)
            A_CI[k, l] = rate[j]
        hazard_chronic_recovery = self.RVs.chronic_recovery.hazard(
            self.ages_mid)
        A_CC = self.get_A_YY(hazard_chronic_recovery * self.age_step)
        A_C = [None, None, None, None, None, A_CI, A_CC]
        b_C = self.get_b_Z(self.K)
        return (A_C, b_C)

    def get_A_b(self, format='csr'):
        (A_M, b_M) = self.get_row_M()
        (A_S, b_S) = self.get_row_S()
        (A_R, b_R) = self.get_row_R()
        (A_L, b_L) = self.get_row_L()
        (A_E, b_E) = self.get_row_E()
        (A_I, b_I) = self.get_row_I()
        (A_C, b_C) = self.get_row_C()
        A = sparse.bmat([A_M, A_S, A_R, A_L, A_E, A_I, A_C],
                        format=format)
        b = sparse.vstack([b_M, b_S, b_R, b_L, b_E, b_I, b_C],
                          format=format)
        return (A, b)

    def solve(self, absmax=1e9):
        (A, b) = self.get_A_b()
        # TODO
        # What is this `numpy.clip()`?!?
        numpy.clip(A.data, -absmax, absmax, A.data)
        Pp = sparse.linalg.spsolve(A, b)
        i_split = 4 * self.I
        [P, p] = [Pp[:i_split], Pp[i_split:]]
        [P_M, P_S, P_R, P_L] = numpy.hsplit(P, 4)
        [p_E, p_I, p_C] = numpy.hsplit(p, 3)
        T = self.get_T()
        [P_E, P_I, P_C] = map(T.dot, [p_E, p_I, p_C])
        rows = pandas.Index(self.ages, name='age')
        P = pandas.DataFrame({'maternal immunity': P_M,
                              'susceptible': P_S,
                              'exposed': P_E,
                              'infectious': P_I,
                              'chronic': P_C,
                              'recovered': P_R,
                              'lost immunity': P_L},
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
    # Turning on infection gives an incorrect blip around a = 0.5.
    # hazard_infection = 0
    # No E -> I.
    # Turning on progression gives nonsense.
    parameters.progression_mean = numpy.inf
    # No I -> C.
    # parameters.probability_chronic = 0
    # No I -> R (and C).
    # parameters.recovery_mean = numpy.inf
    # No C -> R.
    # parameters.chronic_recovery_mean = numpy.inf
    # No R -> L.
    # parameters.antibody_loss_hazard_alpha = \
    #     parameters.antibody_loss_hazard_beta = 0
    # No L -> R.
    # parameters.antibody_gain_hazard = 0
    RVs = RandomVariables(parameters, _initial_conditions=False)
    age_max = 10
    age_step = 0.1
    solver = Solver(hazard_infection, RVs, age_max, age_step)
    P = solver.solve()
    plot(P)
