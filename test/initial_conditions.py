#!/usr/bin/python3


import re
import sys
import time

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

    def get_b(self):
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

    def get_A(self):
        '''Assemble the block row A_X from its columns A_XY.'''
        return {self.get_Y(block_name): getattr(self, block_name)
                for block_name in dir(self)
                if self.is_A_XY(block_name)}


class BlockODE(Block):
    '''A `Block()` for a variable governed by an ODE.'''
    def __len__(self):
        return self.solver.length_ODE

    def get_A_XX(self, hazard_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        # The values on the diagonal.
        d_0 = 1 + hazard_out * self.solver.age_step / 2
        # The values on the subdiagonal.
        d_1 = - 1 + hazard_out * self.solver.age_step / 2
        assert (d_1 <= 0).all()
        return sparse.diags([numpy.hstack([1, d_0]), d_1], [0, -1],
                            shape=(len(self), len(self)))

    def get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        d = - hazard_in * self.solver.age_step / 2
        return sparse.diags([numpy.hstack([0, d]), d], [0, -1],
                            shape=(len(self), self.solver.length_ODE))

    def get_A_XY_PDE(self, survival_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.solver.length_PDE))
        dS = - numpy.diff(survival_in)
        for i in range(1, self.solver.length_ODE):
            j = numpy.arange(1, i + 1)
            A_XY[i, i - j] = - dS[j - 1] * self.solver.age_step
        return A_XY


class BlockPDE(Block):
    '''A `Block()` for a variable governed by a PDE.'''
    def __len__(self):
        return self.solver.length_PDE

    def get_A_XX(self):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        return sparse.eye(len(self))

    def get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        A_XY = sparse.lil_matrix((len(self), self.solver.length_ODE))
        i = numpy.arange(1, self.solver.length_ODE)
        A_XY[i, i - 1] = A_XY[i, i] = - hazard_in / 2
        return A_XY

    def get_A_XY_PDE(self, survival_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.solver.length_PDE))
        dS = - numpy.diff(survival_in)
        for i in range(1, self.solver.length_ODE):
            j = numpy.arange(1, i + 1)
            A_XY[i, i - j] = - dS[j - 1]
        return A_XY

    def integrate(self, p_X, survival_out):
        P_X = numpy.empty(self.solver.length_ODE)
        for i in range(self.solver.length_ODE):
            j = numpy.arange(i + 1)
            P_X[i] = (numpy.dot(survival_out[j],
                                p_X[i - j])
                      * self.solver.age_step)
        return P_X


class BlockM(BlockODE):
    initial_value = 1

    @property
    def A_MM(self):
        return self.get_A_XX(self.solver.hazard.maternal_immunity_waning)


class BlockS(BlockODE):
    initial_value = 0

    @property
    def A_SM(self):
        return self.get_A_XY_ODE(self.solver.hazard.maternal_immunity_waning)

    @property
    def A_SS(self):
        return self.get_A_XX(self.solver.hazard.infection)


class BlockE(BlockPDE):
    initial_value = 0

    @property
    def A_ES(self):
        return self.get_A_XY_ODE(self.solver.hazard.infection)

    @property
    def A_EL(self):
        return self.get_A_XY_ODE(self.solver.hazard.infection)

    @property
    def A_EE(self):
        return self.get_A_XX()

    def integrate(self, p_E):
        return super().integrate(p_E, self.solver.survival.progression)


class BlockI(BlockPDE):
    initial_value = 0

    @property
    def A_IE(self):
        return self.get_A_XY_PDE(self.solver.survival.progression)

    @property
    def A_II(self):
        return self.get_A_XX()

    def integrate(self, p_I):
        return super().integrate(p_I, self.solver.survival.recovery)


class BlockC(BlockPDE):
    initial_value = 0

    @property
    def A_CI(self):
        return self.get_A_XY_PDE(self.solver.probability_chronic
                                 * self.solver.survival.recovery)

    @property
    def A_CC(self):
        return self.get_A_XX()

    def integrate(self, p_C):
        return super().integrate(p_C, self.solver.survival.chronic_recovery)


class BlockR(BlockODE):
    initial_value = 0

    @property
    def A_RI(self):
        return self.get_A_XY_PDE((1 - self.solver.probability_chronic)
                                 * self.solver.survival.recovery)

    @property
    def A_RC(self):
        return self.get_A_XY_PDE(self.solver.survival.chronic_recovery)

    @property
    def A_RR(self):
        return self.get_A_XX(self.solver.hazard.antibody_loss)

    @property
    def A_RL(self):
        return self.get_A_XY_ODE(self.solver.hazard.antibody_gain)


class BlockL(BlockODE):
    initial_value = 0

    @property
    def A_LR(self):
        return self.get_A_XY_ODE(self.solver.hazard.antibody_loss)

    @property
    def A_LL(self):
        return self.get_A_XX(self.solver.hazard.antibody_gain
                             + self.solver.hazard.infection)


class Solver:
    '''Crank–Nicolson solver to find the probability of being in each
    compartment as a function of age.'''

    # The variables P_X that are governed by ODEs.
    vars_ODE = ('M', 'S', 'R', 'L')

    # The variables p_X that are governed by PDEs.
    vars_PDE = ('E', 'I', 'C')

    # The long names of the above.
    # The order of the output is determined by the order of these, too.
    col_names = {'M': 'maternal immunity',
                 'S': 'susceptible',
                 'E': 'exposed',
                 'I': 'infectious',
                 'C': 'chronic',
                 'R': 'recovered',
                 'L': 'lost immunity'}

    def __init__(self, hazard_infection, RVs, age_max, age_step):
        self.age_max = age_max
        self.age_step = age_step
        self.ages = arange(0, self.age_max, self.age_step)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[ : -1] + self.ages[1 : ]) / 2
        self.length_ODE = self.length_PDE = len(self.ages)
        self.set_params(hazard_infection, RVs)
        self.set_blocks()

    @staticmethod
    def rec_fromkwds(**kwds):
        return numpy.rec.fromarrays(
            numpy.broadcast_arrays(*kwds.values()),
            names=list(kwds.keys()))

    def set_params(self, hazard_infection, RVs):
        waiting_times = {'maternal_immunity_waning',
                         'progression',
                         'recovery',
                         'chronic_recovery',
                         'antibody_gain'}
        hazard = {}
        survival = {}
        for k in waiting_times:
            RV = getattr(RVs, k)
            with numpy.errstate(divide='ignore'):
                hazard[k] = RV.hazard(self.ages_mid)
            survival[k] = RV.sf(self.ages)
        hazard['infection'] = hazard_infection
        hazard['antibody_loss'] = RVs.antibody_loss.hazard(
            RVs.antibody_loss.time_min)
        self.hazard = self.rec_fromkwds(**hazard)
        self.survival = self.rec_fromkwds(**survival)
        self.probability_chronic = RVs.probability_chronic.probability_chronic

    def set_blocks(self):
        self.blocks = {}
        # Loop over all the subclasses of `BlockODE` and `BlockPDE`.
        for typ in Block.__subclasses__():
            for BlockX in typ.__subclasses__():
                blockX = BlockX(self)
                self.blocks[blockX.X] = blockX

    def stack(self, rows):
        '''Stack the P_X and p_X vectors into one big vector.'''
        # Recurse for 2-d arrays.
        if not isinstance(rows, dict):
            return rows
        else:
            # `vars_ODE` first, then `vars_PDE`.
            return [self.stack(rows.get(k))
                    for k in (self.vars_ODE + self.vars_PDE)]

    @classmethod
    def unstack_and_label(cls, x, x_vars):
        # Split into columns.
        X = numpy.hsplit(x, len(x_vars))
        # Add labels.
        return cls.rec_fromkwds(**dict(zip(x_vars, X)))

    def unstack(self, Pp):
        '''Unstack the P_X and p_X vectors.'''
        # `vars_ODE` are first, then `vars_PDE`.
        split = len(self.vars_ODE) * self.length_ODE
        P = self.unstack_and_label(Pp[ : split], self.vars_ODE)
        p = self.unstack_and_label(Pp[split : ], self.vars_PDE)
        return (P, p)

    def stack_sparse(self, rows, format='csr'):
        return sparse.bmat(self.stack(rows), format=format)

    def get_A(self):
        A = {var: block.get_A() for (var, block) in self.blocks.items()}
        return self.stack_sparse(A)

    def get_b(self):
        b = {var: [block.get_b()] for (var, block) in self.blocks.items()}
        return self.stack_sparse(b)

    def integrate_PDE_vars(self, p):
        # Integrate the p_X variables over r.
        p_integrated = {n: self.blocks[n].integrate(p[n])
                        for n in p.dtype.names}
        return self.rec_fromkwds(**p_integrated)

    def solve(self):
        t0 = time.time()
        A = self.get_A()
        b = self.get_b()
        t1 = time.time()
        print(f'Setup took {t1 - t0} seconds.')
        assert numpy.isfinite(A.data).all()
        t2 = time.time()
        Pp = sparse.linalg.spsolve(A, b)
        t3 = time.time()
        print(f'Solve took {t3 - t2} seconds.')
        (P, p) = self.unstack(Pp)
        p_integrated = self.integrate_PDE_vars(p)
        P = numpy.lib.recfunctions.merge_arrays(
            (P, p_integrated),
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
    RVs = RandomVariables(parameters, _initial_conditions=False)
    age_max = 10
    age_step = 0.01
    solver = Solver(hazard_infection, RVs, age_max, age_step)
    P = solver.solve()
    plot(P)
