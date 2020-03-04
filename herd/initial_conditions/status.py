import os.path
import re

from joblib import Memory
import numpy
import numpy.lib.recfunctions
import pandas
from scipy import sparse

from herd import (antibody_gain, antibody_loss, chronic_recovery,
                  maternal_immunity_waning, parameters, progression,
                  recovery, utility)


class Block:
    '''Get a block row A_X of A and block b_X of b.'''

    def __init__(self, params):
        self.params = params
        self.set_A_X()
        self.set_b_X()

    @property
    def X(self):
        '''Get the variable name from the last letter of the class name.'''
        return self.__class__.__name__[-1]

    def set_b_X(self):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # The initial value is in the 0th entry,
        # then zeros everywhere else.
        self.b_X = sparse.csr_matrix(([self.initial_value], ([0], [0])),
                                     shape=(len(self), 1))

    def _is_set_A_XY(self, attr):
        '''Check if `attr` is 'set_A_XY' for some Y.'''
        return (re.match(f'set_A_{self.X}([A-Z])$', attr) is not None)

    def set_A_X(self):
        '''Assemble the block row A_X from its columns A_XY.'''
        self.A_X = {}
        for attr in dir(self):
            if self._is_set_A_XY(attr):
                getattr(self, attr)()

    def update_hazard_infection(self):
        '''Do nothing unless overridden.'''


class BlockODE(Block):
    '''A `Block()` for a variable governed by an ODE.'''
    def __len__(self):
        return self.params.length_ODE

    def _get_A_XX(self, hazard_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        # The values on the diagonal.
        d_0 = 1 + hazard_out * self.params.step / 2
        # The values on the subdiagonal.
        d_1 = - 1 + hazard_out * self.params.step / 2
        assert (d_1 <= 0).all()
        return sparse.diags([numpy.hstack([1, d_0]), d_1], [0, -1],
                            shape=(len(self), len(self)))

    def _get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        d = - hazard_in * self.params.step / 2
        return sparse.diags([numpy.hstack([0, d]), d], [0, -1],
                            shape=(len(self), self.params.length_ODE))

    def _get_A_XY_PDE(self, survival_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_PDE))
        dS = - numpy.diff(survival_in)
        for i in range(1, self.params.length_ODE):
            j = numpy.arange(1, i + 1)
            A_XY[i, i - j] = - dS[j - 1] * self.params.step
        return A_XY


class BlockPDE(Block):
    '''A `Block()` for a variable governed by a PDE.'''
    def __len__(self):
        return self.params.length_PDE

    def _get_A_XX(self):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        return sparse.eye(len(self))

    def _get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_ODE))
        i = numpy.arange(1, self.params.length_ODE)
        A_XY[i, i - 1] = A_XY[i, i] = - hazard_in / 2
        return A_XY

    def _get_A_XY_PDE(self, survival_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_PDE))
        dS = - numpy.diff(survival_in)
        for i in range(1, self.params.length_ODE):
            j = numpy.arange(1, i + 1)
            A_XY[i, i - j] = - dS[j - 1]
        return A_XY

    def integrate(self, p_X, survival_out):
        P_X = numpy.empty(self.params.length_ODE)
        for i in range(self.params.length_ODE):
            j = numpy.arange(i + 1)
            P_X[i] = numpy.dot(survival_out[j], p_X[i - j]) * self.params.step
        return P_X


class BlockM(BlockODE):
    initial_value = 1

    def set_A_MM(self):
        self.A_X['M'] = self._get_A_XX(
            self.params.hazard.maternal_immunity_waning)


class BlockS(BlockODE):
    initial_value = 0

    def set_A_SM(self):
        self.A_X['M'] = self._get_A_XY_ODE(
            self.params.hazard.maternal_immunity_waning)

    def set_A_SS(self):
        self.A_X['S'] = self._get_A_XX(self.params.hazard.infection)

    def update_hazard_infection(self):
        self.set_A_SS()


class BlockE(BlockPDE):
    initial_value = 0

    def set_A_ES(self):
        self.A_X['S'] = self._get_A_XY_ODE(self.params.hazard.infection)

    def set_A_EL(self):
        self.A_X['L'] = self._get_A_XY_ODE(self.params.hazard.infection)

    def set_A_EE(self):
        self.A_X['E'] = self._get_A_XX()

    def integrate(self, p_E):
        return super().integrate(p_E, self.params.survival.progression)

    def update_hazard_infection(self):
        self.set_A_ES()
        self.set_A_EL()


class BlockI(BlockPDE):
    initial_value = 0

    def set_A_IE(self):
        self.A_X['E'] = self._get_A_XY_PDE(self.params.survival.progression)

    def set_A_II(self):
        self.A_X['I'] = self._get_A_XX()

    def integrate(self, p_I):
        return super().integrate(p_I, self.params.survival.recovery)


class BlockC(BlockPDE):
    initial_value = 0

    def set_A_CI(self):
        self.A_X['I'] = self._get_A_XY_PDE(self.params.probability_chronic
                                           * self.params.survival.recovery)

    def set_A_CC(self):
        self.A_X['C'] = self._get_A_XX()

    def integrate(self, p_C):
        return super().integrate(p_C, self.params.survival.chronic_recovery)


class BlockR(BlockODE):
    initial_value = 0

    def set_A_RI(self):
        self.A_X['I'] =  self._get_A_XY_PDE(
            (1 - self.params.probability_chronic)
            * self.params.survival.recovery)

    def set_A_RC(self):
        self.A_X['C'] = self._get_A_XY_PDE(
            self.params.survival.chronic_recovery)

    def set_A_RR(self):
        self.A_X['R'] = self._get_A_XX(self.params.hazard.antibody_loss)

    def set_A_RL(self):
        self.A_X['L'] = self._get_A_XY_ODE(self.params.hazard.antibody_gain)


class BlockL(BlockODE):
    initial_value = 0

    def set_A_LR(self):
        self.A_X['R'] = self._get_A_XY_ODE(self.params.hazard.antibody_loss)

    def set_A_LL(self):
        self.A_X['L'] = self._get_A_XX(self.params.hazard.antibody_gain
                                       + self.params.hazard.infection)

    def update_hazard_infection(self):
        self.set_A_LL()


class Probability:
    '''The solution from `Solver()`, that then gets interpolated
    to different ages as needed.'''

    def __init__(self, probability):
        self._probability = probability

    def __call__(self, age):
        # Interpolate `self._probability` to `age`.
        return (self._probability.reindex(self._probability.index.union(age))
                                 .interpolate()
                                 .loc[age])


class Params:
    '''Dummy object to pass parameters from Solver() to Block()s.'''


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

    # Step size in both age (a) and residence time (r).
    step = 0.01

    age_max = 20

    def __init__(self, hazard_infection, params):
        self.ages = utility.arange(0, self.age_max, self.step,
                                   endpoint=True)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[ : -1] + self.ages[1 : ]) / 2
        self.length_ODE = self.length_PDE = len(self.ages)
        self.set_params(hazard_infection, params)
        self.set_blocks()
        self.set_A()
        self.set_b()

    @staticmethod
    def rec_fromkwds(**kwds):
        return numpy.rec.fromarrays(
            numpy.broadcast_arrays(*kwds.values()),
            names=list(kwds.keys()))

    def set_params(self, hazard_infection, params):
        self.params = Params()
        self.params.step = self.step
        self.params.length_ODE = self.length_ODE
        self.params.length_PDE = self.length_PDE
        RVs = {
            'maternal_immunity_waning': maternal_immunity_waning.gen(params),
            'progression': progression.gen(params),
            'recovery': recovery.gen(params),
            'chronic_recovery': chronic_recovery.gen(params),
            'antibody_gain': antibody_gain.gen(params)
        }
        with numpy.errstate(divide='ignore'):
            hazard = {k: RV.hazard(self.ages_mid)
                      for (k, RV) in RVs.items()}
        survival = {k: RV.sf(self.ages)
                    for (k, RV) in RVs.items()}
        antibody_loss_RV = antibody_loss.gen(params)
        hazard['antibody_loss'] = antibody_loss_RV.hazard(
            antibody_loss_RV.time_min)
        hazard['infection'] = hazard_infection
        self.params.hazard = self.rec_fromkwds(**hazard)
        self.params.survival = self.rec_fromkwds(**survival)
        self.params.probability_chronic = params.probability_chronic

    def set_blocks(self):
        self.blocks = {}
        # Loop over all the subclasses of `BlockODE` and `BlockPDE`.
        for typ in Block.__subclasses__():
            for BlockX in typ.__subclasses__():
                blockX = BlockX(self.params)
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

    def set_A(self):
        self.A = {var: block.A_X for (var, block) in self.blocks.items()}

    def set_b(self):
        self.b = {var: [block.b_X] for (var, block) in self.blocks.items()}

    def integrate_PDE_vars(self, p):
        # Integrate the p_X variables over r.
        p_integrated = {n: self.blocks[n].integrate(p[n])
                        for n in p.dtype.names}
        return self.rec_fromkwds(**p_integrated)

    def solve(self):
        A = self.stack_sparse(self.A)
        b = self.stack_sparse(self.b)
        assert numpy.isfinite(A.data).all()
        Pp = sparse.linalg.spsolve(A, b)
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
        assert ((P >= 0) | numpy.isclose(P, 0)).all(axis=None)
        assert numpy.isclose(P.sum(axis='columns'), 1).all()
        return Probability(P)

    def update_hazard_infection_and_solve(self, hazard_infection):
        self.params.hazard.infection = hazard_infection
        for block in self.blocks.values():
            block.update_hazard_infection()
        return self.solve()


class CacheParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `_status()` and `_find_hazard()`
    so that it can be efficiently cached.'''
    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        attrs = {'antibody_gain_hazard',
                 'antibody_loss_hazard_alpha',
                 'antibody_loss_hazard_beta',
                 'antibody_loss_hazard_time_max',
                 'antibody_loss_hazard_time_min',
                 'chronic_recovery_mean',
                 'chronic_recovery_shape',
                 'maternal_immunity_duration_mean',
                 'maternal_immunity_duration_shape',
                 'probability_chronic',
                 'progression_mean',
                 'progression_shape',
                 'recovery_mean',
                 'recovery_shape'}
        for attr in attrs:
            setattr(self, attr, float(getattr(params, attr)))


# The function is very slow because of the calls to
# `Solver()`, so the values are cached to disk with
# `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir, verbose=0)
@_cache.cache
def _status_probability(hazard_infection, params):
    return Solver(hazard_infection, params).solve()


def status_probability(hazard_infection, params):
    '''Get the `Probability()` that can be interpolated
    onto ages as needed.'''
    return _status_probability(hazard_infection,
                               CacheParameters(params))


def probability(age, hazard_infection, params):
    '''The probability of being in each immune status at age `a`,
    given being alive at age `a`.'''
    return status_probability(hazard_infection, params)(age)
