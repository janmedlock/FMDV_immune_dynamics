#!/usr/bin/python3
#
# TODO: Vectorize loops.

import re
import sys
import time
import warnings

import numpy
import numpy.lib.recfunctions
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


class Block:
    '''Get a block row A_X of A and block b_X of b.'''
    def __init__(self, solver):
        self.solver = solver

    @property
    def X(self):
        '''Get the variable name from the last letter of the class name.'''
        return self.__class__.__name__[-1]

    @property
    def b(self):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # The initial value is in the 0th entry,
        # then zeros everywhere else.
        return sparse.csr_matrix(([self.initial_value], ([0], [0])),
                                 shape=(len(self), 1))

    def get_Y_match(self, block_name):
        '''`block_name` is 'A_XY' for some Y.'''
        return re.match(f'A_{self.X}([A-Z])$', block_name)

    def get_Y(self, block_name):
        '''Get the 'Y' value out of 'A_XY'.'''
        return self.get_Y_match(block_name).group(1)

    def is_A_XY(self, block_name):
        '''Check if `block_name` is 'A_XY' for some Y.'''
        return (self.get_Y_match(block_name) is not None)

    @property
    def A(self):
        '''Assemble the block row A_X from its columns A_XY.'''
        return {self.get_Y(block_name): getattr(self, block_name)
                for block_name in dir(self)
                if self.is_A_XY(block_name)}


class SizeI:
    '''An `XBlock()` of size I.'''
    def __len__(self):
        return self.solver.I

    def A_XX(self, rate_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        # The values on the diagonal.
        d_0 = numpy.hstack([1, 1 + rate_out * self.solver.age_step / 2])
        # The values on the subdiagonal.
        d_1 = - (1 - rate_out * self.solver.age_step / 2)
        assert (d_1 <= 0).all()
        return sparse.diags([d_0, d_1], [0, -1],
                            shape=(len(self), len(self)))

    def A_XY(self, rate_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X.'''
        v = - rate_in * self.solver.age_step / 2
        # The values on the diagonal.
        d_0 = numpy.hstack([0, v])
        # The values on the subdiagonal.
        d_1 = v
        return sparse.diags([d_0, d_1], [0, -1],
                            shape=(len(self), len(self)))


class SizeK:
    '''An `XBlock()` of size K.'''
    def __len__(self):
        return self.solver.K

    def A_XX(self, rate_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        A_XX = sparse.lil_matrix((len(self), len(self)))
        d_0 = numpy.hstack([1, 1 + rate_out * self.solver.age_step / 2])
        d_1 = - (1 - rate_out * self.solver.age_step / 2)
        assert (d_1 <= 0).all()
        for i in range(self.solver.I):
            for j in range(i + 1):
                k = self.solver.get_k(i, j)
                A_XX[k, k] = d_0[j]
                if j > 0:
                    A_XX[k, self.solver.get_k(i - 1, j - 1)] = d_1[j - 1]
        return A_XX


class BlockM(Block, SizeI):
    initial_value = 1

    @property
    def A_MM(self):
        return self.A_XX(self.solver.rates.maternal_immunity_waning)


class BlockS(Block, SizeI):
    initial_value = 0

    @property
    def A_SM(self):
        return self.A_XY(self.solver.rates.maternal_immunity_waning)

    @property
    def A_SS(self):
        return self.A_XX(self.solver.rates.infection)


class BlockE(Block, SizeK):
    initial_value = 0

    @property
    def A_ES(self):
        A_ES = sparse.lil_matrix((len(self), self.solver.I))
        for i in range(1, self.solver.I):
            k = self.solver.get_k(i, 0)
            A_ES[k, [i - 1, i]] = - self.solver.rates.infection[i - 1] / 2
        return A_ES

    @property
    def A_EL(self):
        A_EL = sparse.lil_matrix((len(self), self.solver.I))
        for i in range(1, self.solver.I):
            k = self.solver.get_k(i, 0)
            A_EL[k, [i - 1, i]] = - self.solver.rates.infection[i - 1] / 2
        return A_EL

    @property
    def A_EE(self):
        return self.A_XX(self.solver.rates.progression)


class BlockI(Block, SizeK):
    initial_value = 0

    @property
    def A_IE(self):
        A_IE = sparse.lil_matrix((len(self), len(self)))
        for i in range(1, self.solver.I):
            k = self.solver.get_k(i, 0)
            for j in range(1, i + 1):
                l = self.solver.get_k([i - 1, i], [j - 1, j])
                A_IE[k, l] = - (self.solver.rates.progression[j - 1]
                                * self.solver.age_step / 2)
        return A_IE

    @property
    def A_II(self):
        return self.A_XX(self.solver.rates.recovery)


class BlockR(Block, SizeI):
    initial_value = 0

    @property
    def A_RI(self):
        A_RI = sparse.lil_matrix((len(self), self.solver.K))
        for i in range(1, self.solver.I):
            for j in range(1, i + 1):
                k = self.solver.get_k([i - 1, i], [j - 1, j])
                A_RI[i, k] = - (self.solver.rates.recovery[j - 1]
                                * self.solver.age_step ** 2 / 2)
        return A_RI

    @property
    def A_RR(self):
        return self.A_XX(self.solver.rates.antibody_loss)

    @property
    def A_RL(self):
        return self.A_XY(self.solver.rates.antibody_gain)


class BlockL(Block, SizeI):
    initial_value = 0

    @property
    def A_LR(self):
        return self.A_XY(self.solver.rates.antibody_loss)

    @property
    def A_LL(self):
        return self.A_XX(self.solver.rates.antibody_gain
                         + self.solver.rates.infection)


class Blocks:
    def __init__(self, solver):
        self.blocks = {}
        for cls in Block.__subclasses__():
            block = cls(solver)
            self.blocks[block.X] = block

    @property
    def A(self):
        return {var: block.A for (var, block) in self.blocks.items()}

    @property
    def b(self):
        return {var: block.b for (var, block) in self.blocks.items()}


class Solver:
    '''Crank–Nicolson solver to find the probability of being in each
    compartment as a function of age.'''

    # The P_X vectors, the ones of length I.
    P_vars = ('M', 'S', 'R', 'L')

    # The p_X vectors, the ones of length K.
    p_vars = ('E', 'I')

    # The long names of the above.
    # The order of the output is determined by the order of these, too.
    col_names = {'M': 'maternal immunity',
                 'S': 'susceptible',
                 'E': 'exposed',
                 'I': 'infectious',
                 'R': 'recovered',
                 'L': 'lost immunity'}

    def __init__(self, hazard_infection, RVs, age_max, age_step):
        self.hazard_infection = hazard_infection
        self.RVs = RVs
        self.age_max = age_max
        self.age_step = age_step
        self.ages = arange(0, self.age_max, self.age_step)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[:-1] + self.ages[1:]) / 2
        self.rates = self.get_rates()

    def clip(self, rate):
        # This ensures that d_1 ≤ 0 in `SizeI.A_XX()` and `SizeK.A_XX()`
        # so that P_X ≥ 0.
        if numpy.any(rate > 2 / self.age_step):
            warnings.warn('Clipping!')
        return numpy.clip(rate, 0, 2 / self.age_step)

    def get_rates(self):
        with numpy.errstate(divide='ignore'):
            rates = {
                'maternal_immunity_waning':
                self.RVs.maternal_immunity_waning.hazard(self.ages_mid),
                'infection': self.hazard_infection,
                'progression': self.RVs.progression.hazard(self.ages_mid),
                'recovery': self.RVs.recovery.hazard(self.ages_mid),
                'antibody_loss': self.RVs.antibody_loss.hazard(
                    self.RVs.antibody_loss.time_min),
                'antibody_gain': self.RVs.antibody_gain.antibody_gain_hazard}
        for (k, v) in rates.items():
            if numpy.isscalar(v):
                rates[k] = v * numpy.ones(self.I - 1)
            rates[k] = self.clip(rates[k])
        return numpy.rec.fromarrays(rates.values(),
                                    names=list(rates.keys()))

    def get_k(self, i, j):
        '''Get the position `k` in the stacked vector representation
        p_X^k \approx p_X(a^i, r^j) of the entry for age a^i,
        residence time r^j.'''
        # Convert iterables to arrays so the arithmetic works on them.
        (i, j) = map(numpy.asarray, (i, j))
        assert ((0 <= i) & (i < self.I)).all()
        assert ((0 <= j) & (j <= i)).all()
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
        using the composite trapezoid rule.
        The result is a I × K matrix that
        when left-multiplied with a K vector
        produces a vector over ages a.'''
        T = sparse.lil_matrix((self.I, self.K))
        for i in range(self.I):
            T[i, self.get_k(i, range(i + 1))] = self.age_step
        return T.tocsr()

    def stack(self, rows):
        '''Stack the P_X and p_X vectors into one big vector.'''
        # Recurse for 2-d arrays.
        if not isinstance(rows, dict):
            return rows
        else:
            # `P_vars` first, then `p_vars`.
            return [self.stack(rows.get(k))
                    for k in (self.P_vars + self.p_vars)]

    def unstack(self, Pp):
        '''Unstack the P_X and p_X vectors.'''
        i_split = len(self.P_vars) * self.I
        # `P_vars` are first, then `p_vars`.
        (P, p) = (Pp[:i_split], Pp[i_split:])
        # Split into columns and add labels.
        P = numpy.rec.fromarrays(numpy.hsplit(P, len(self.P_vars)),
                                 names=self.P_vars)
        p = numpy.rec.fromarrays(numpy.hsplit(p, len(self.p_vars)),
                                 names=self.p_vars)
        return (P, p)

    def get_A_b(self, format='csr'):
        blocks = Blocks(self)
        # Stack the columns.
        A = sparse.bmat(self.stack(blocks.A),
                        format=format)
        b = sparse.vstack(self.stack(blocks.b),
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
        (P, p) = self.unstack(Pp)
        # Integrate the p_X's over r.
        T = self.get_T()
        p_names = p.dtype.names
        P_integrated = numpy.rec.fromarrays(
            [T.dot(p[n]) for n in p_names],
            names=p_names)
        P = numpy.lib.recfunctions.merge_arrays(
            (P, P_integrated),
            asrecarray=True, flatten=True)
        P = pandas.DataFrame(P,
                             index=pandas.Index(self.ages, name='age'))
        P.rename(columns=self.col_names, inplace=True)
        # Order columns.
        P.set_axis(pandas.CategoricalIndex(P.columns,
                                           self.col_names.values(),
                                           ordered=True),
                   axis='columns', inplace=True)
        P.sort_index(axis='columns', inplace=True)
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
    # Slower recovery.
    parameters.recovery_mean = 5
    RVs = RandomVariables(parameters, _initial_conditions=False)
    age_max = 10
    age_step = 0.1
    solver = Solver(hazard_infection, RVs, age_max, age_step)
    P = solver.solve()
    plot(P)
