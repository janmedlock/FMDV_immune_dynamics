import re

import numpy
import numpy.lib.recfunctions
import pandas
from scipy import sparse

from herd import (antibody_gain, antibody_loss, chronic_recovery,
                  maternal_immunity_waning, progression, recovery, utility)


_step_default = 0.01
_age_max_default = 20


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
        d_0 = 1 + hazard_out * self.solver.step / 2
        # The values on the subdiagonal.
        d_1 = - 1 + hazard_out * self.solver.step / 2
        assert (d_1 <= 0).all()
        return sparse.diags([numpy.hstack([1, d_0]), d_1], [0, -1],
                            shape=(len(self), len(self)))

    def get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        d = - hazard_in * self.solver.step / 2
        return sparse.diags([numpy.hstack([0, d]), d], [0, -1],
                            shape=(len(self), self.solver.length_ODE))

    def get_A_XY_PDE(self, survival_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.solver.length_PDE))
        dS = - numpy.diff(survival_in)
        for i in range(1, self.solver.length_ODE):
            j = numpy.arange(1, i + 1)
            A_XY[i, i - j] = - dS[j - 1] * self.solver.step
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
                      * self.solver.step)
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


class Status:
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

    def __init__(self, hazard_infection, parameters,
                 step=_step_default, age_max=_age_max_default):
        self.step = step
        self.ages = utility.arange(0, age_max, self.step,
                                   endpoint=True)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[ : -1] + self.ages[1 : ]) / 2
        self.length_ODE = self.length_PDE = len(self.ages)
        self.set_params(hazard_infection, parameters)
        self.set_blocks()
        self.solve()

    @staticmethod
    def rec_fromkwds(**kwds):
        return numpy.rec.fromarrays(
            numpy.broadcast_arrays(*kwds.values()),
            names=list(kwds.keys()))

    def set_params(self, hazard_infection, parameters):
        RVs = {'maternal_immunity_waning':
               maternal_immunity_waning.gen(parameters),
               'progression':
               progression.gen(parameters),
               'recovery':
               recovery.gen(parameters),
               'chronic_recovery':
               chronic_recovery.gen(parameters),
               'antibody_gain':
               antibody_gain.gen(parameters)}
        with numpy.errstate(divide='ignore'):
            hazard = {k: RV.hazard(self.ages_mid)
                      for (k, RV) in RVs.items()}
        survival = {k: RV.sf(self.ages)
                    for (k, RV) in RVs.items()}
        antibody_loss_RV = antibody_loss.gen(parameters)
        hazard['antibody_loss'] = antibody_loss_RV.hazard(
            antibody_loss_RV.time_min)
        hazard['infection'] = hazard_infection
        self.hazard = self.rec_fromkwds(**hazard)
        self.survival = self.rec_fromkwds(**survival)
        self.probability_chronic = parameters.probability_chronic

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
        A = self.get_A()
        b = self.get_b()
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
        self._probability = P

    @staticmethod
    def _interpolate(df, index):
        return (df.reindex(df.index.union(index))
                  .interpolate()
                  .loc[index])

    def probability(self, age):
        return self._interpolate(self._probability, age)


def probability(age, hazard_infection, parameters):
    '''The probability of being in each immune status at age `a`,
    given being alive at age `a`.'''
    # TODO: Reuse Status() to speed up multiple calls to this function from
    # `herd.initial_conditions.infection.find_hazard()`.
    # Should I cache something here too?
    status = Status(hazard_infection, parameters)
    return status.probability(age)
