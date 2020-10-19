import os.path
import re

from joblib import Memory
import numpy
import numpy.lib.recfunctions
import pandas
from scipy import integrate, optimize, sparse

from herd import (antibody_gain, antibody_loss, birth, buffalo,
                  chronic_recovery, maternal_immunity_waning,
                  mortality, parameters, progression, recovery,
                  utility)


def hazard_birth_constant_time(ages, RV_mortality, age_step):
    RV_birth = birth.from_param_values(
        birth_seasonal_coefficient_of_variation=0,
        # This value doesn't matter when the variation is 0.
        birth_peak_time_of_year=0,
        # We will handle scaling ourselves.
        _scaling=1)
    hazard = RV_birth.hazard(
        age=ages,
        # This value doesn't matter when the variation is 0.
        time=0)
    # Scale so that the population growth rate is 0.
    hazard /= numpy.dot(hazard, RV_mortality.sf(ages)) * age_step
    return hazard


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

    def _set_b_X(self, initial_value):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # The initial value is in the 0th entry,
        # then zeros everywhere else.
        self.b_X = sparse.csr_matrix(([initial_value], ([0], [0])),
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

    def update(self):
        '''Do nothing unless overridden.'''


class BlockODE(Block):
    '''A `Block()` for a variable governed by an ODE.'''
    def __len__(self):
        return self.params.length_ODE

    def _get_A_XX(self, hazard_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        d_X = ((hazard_out + self.params.hazard.mortality)
               * self.params.step / 2)
        diags = ((numpy.hstack([1, 1 + d_X]), 0),  # The diagonal
                 (- 1 + d_X, -1))  # The subdiagonal
        # Ensure that the off-diagonal entries are non-positive.
        for (v, k) in diags:
            if k != 0:
                assert (v <= 0).all()
        return sparse.diags(*zip(*diags),
                            shape=(len(self), len(self)))

    def _get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        f_XY = - hazard_in * self.params.step / 2
        diags = ((numpy.hstack([0, f_XY]), 0),  # The diagonal
                 (f_XY, -1))  # The subdiagonal
        return sparse.diags(*zip(*diags),
                            shape=(len(self), self.params.length_ODE))

    def _get_A_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_PDE))
        for i in range(1, len(self)):
            j = numpy.arange(i + 1)
            A_XY[i, j] = - (pdf_in[i - j]
                            * self.params.survival.mortality[i]
                            / self.params.survival.mortality[j]
                            * self.params.step ** 2)
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
        # The values on the diagonal and subdiagonal.
        f_XY = - hazard_in / 2
        diags = ((numpy.hstack([0, f_XY]), 0),  # The diagonal
                 (f_XY, -1))  # The subdiagonal
        return sparse.diags(*zip(*diags),
                            shape=(len(self), self.params.length_ODE))

    def _get_A_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_PDE))
        for i in range(1, len(self)):
            j = numpy.arange(i + 1)
            A_XY[i, j] = - (pdf_in[i - j]
                            * self.params.survival.mortality[i]
                            / self.params.survival.mortality[j]
                            * self.params.step)
        return A_XY

    def set_b_X(self):
        self._set_b_X(0)

    def integrate(self, p_X, survival_out):
        P_X = numpy.empty(self.params.length_ODE)
        for i in range(self.params.length_ODE):
            j = numpy.arange(i + 1)
            P_X[i] = (numpy.dot(survival_out[j],
                                (self.params.survival.mortality[i]
                                 / self.params.survival.mortality[i - j]
                                 * p_X[i - j]))
                      * self.params.step)
        return P_X


class BlockM(BlockODE):
    def set_A_MM(self):
        self.A_X['M'] = self._get_A_XX(
            self.params.hazard.maternal_immunity_waning)

    def set_b_X(self):
        self._set_b_X(self.params.newborn_proportion_immune)

    def update(self):
        self.set_b_X()


class BlockS(BlockODE):
    def set_A_SM(self):
        self.A_X['M'] = self._get_A_XY_ODE(
            self.params.hazard.maternal_immunity_waning)

    def set_A_SS(self):
        self.A_X['S'] = self._get_A_XX(self.params.hazard.infection)

    def set_b_X(self):
        self._set_b_X(1 - self.params.newborn_proportion_immune)

    def update(self):
        self.set_A_SS()
        self.set_b_X()


class BlockE(BlockPDE):
    def set_A_ES(self):
        self.A_X['S'] = self._get_A_XY_ODE(self.params.hazard.infection)

    def set_A_EL(self):
        self.A_X['L'] = self._get_A_XY_ODE(self.params.hazard.infection)

    def set_A_EE(self):
        self.A_X['E'] = self._get_A_XX()

    def update(self):
        self.set_A_ES()
        self.set_A_EL()

    def integrate(self, p_E):
        return super().integrate(p_E, self.params.survival.progression)


class BlockI(BlockPDE):
    def set_A_IE(self):
        self.A_X['E'] = self._get_A_XY_PDE(self.params.pdf.progression)

    def set_A_II(self):
        self.A_X['I'] = self._get_A_XX()

    def integrate(self, p_I):
        return super().integrate(p_I, self.params.survival.recovery)


class BlockC(BlockPDE):
    def set_A_CI(self):
        self.A_X['I'] = self._get_A_XY_PDE(self.params.probability_chronic
                                           * self.params.pdf.recovery)

    def set_A_CC(self):
        self.A_X['C'] = self._get_A_XX()

    def integrate(self, p_C):
        return super().integrate(p_C, self.params.survival.chronic_recovery)


class BlockR(BlockODE):
    def set_A_RI(self):
        self.A_X['I'] = self._get_A_XY_PDE(
            (1 - self.params.probability_chronic)
            * self.params.pdf.recovery)

    def set_A_RC(self):
        self.A_X['C'] = self._get_A_XY_PDE(self.params.pdf.chronic_recovery)

    def set_A_RR(self):
        self.A_X['R'] = self._get_A_XX(self.params.hazard.antibody_loss)

    def set_A_RL(self):
        self.A_X['L'] = self._get_A_XY_ODE(self.params.hazard.antibody_gain)

    def set_b_X(self):
        self._set_b_X(0)


class BlockL(BlockODE):
    def set_A_LR(self):
        self.A_X['R'] = self._get_A_XY_ODE(self.params.hazard.antibody_loss)

    def set_A_LL(self):
        self.A_X['L'] = self._get_A_XX(self.params.hazard.antibody_gain
                                       + self.params.hazard.infection)

    def set_b_X(self):
        self._set_b_X(0)

    def update(self):
        self.set_A_LL()


class ProbabilityInterpolant:
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
    step = 0.005

    age_max = 20

    def __init__(self, params):
        self.ages = utility.arange(0, self.age_max, self.step,
                                   endpoint=True)
        assert len(self.ages) > 1
        self.ages_mid = (self.ages[:-1] + self.ages[1:]) / 2
        self.length_ODE = self.length_PDE = len(self.ages)
        self.set_params(params)
        self.set_blocks()
        self.set_A()
        self.set_b()

    @staticmethod
    def rec_fromkwds(**kwds):
        '''Build a `numpy.recarray()` from `kwds`.'''
        return numpy.rec.fromarrays(
            numpy.broadcast_arrays(*kwds.values()),
            dtype=[(k, float) for k in kwds.keys()])

    def set_params(self, params):
        self.params = Params()
        self.params.step = self.step
        self.params.length_ODE = self.length_ODE
        self.params.length_PDE = self.length_PDE
        # We need the survival, pdf, and hazard for these RVs.
        RVs = {
            'mortality': mortality.gen(params),
            'progression': progression.gen(params),
            'recovery': recovery.gen(params),
            'chronic_recovery': chronic_recovery.gen(params),
        }
        survival = {k: RV.sf(self.ages)
                    for (k, RV) in RVs.items()}
        self.params.survival = self.rec_fromkwds(**survival)
        pdf = {k: RV.pdf(self.ages)
               for (k, RV) in RVs.items()}
        self.params.pdf = self.rec_fromkwds(**pdf)
        # We also need the hazard for these RVs.
        RVs.update({
            'maternal_immunity_waning': maternal_immunity_waning.gen(params),
            'antibody_loss': antibody_loss.gen(params),
        })
        with numpy.errstate(divide='ignore'):
            hazard = {k: RV.hazard(self.ages_mid)
                      for (k, RV) in RVs.items()}
        antibody_gain_RV = antibody_gain.gen(params)
        hazard['antibody_gain'] = antibody_gain_RV.hazard(
            antibody_gain_RV.time_min)
        hazard['infection'] = 1  # Dummy value. Set on calls to `solve_step()`.
        self.params.hazard = self.rec_fromkwds(**hazard)
        self.params.probability_chronic = params.probability_chronic
        self.params.transmission_rate = params.transmission_rate
        self.params.chronic_transmission_rate = (
            params.chronic_transmission_rate)
        self.params.hazard_birth = hazard_birth_constant_time(
            self.ages, RVs['mortality'], self.step)
        # Dummy value. Set on calls to `solve_step()`.
        self.params.newborn_proportion_immune = 1

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
        P = self.unstack_and_label(Pp[:split], self.vars_ODE)
        p = self.unstack_and_label(Pp[split:], self.vars_PDE)
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

    def solve_step(self, x):
        (self.params.hazard.infection,
         self.params.newborn_proportion_immune) = x
        for block in self.blocks.values():
            block.update()
        A = self.stack_sparse(self.A)
        b = self.stack_sparse(self.b)
        assert numpy.isfinite(A.data).all()
        Pp = sparse.linalg.spsolve(A, b)
        (P, p) = self.unstack(Pp)
        p_integrated = self.integrate_PDE_vars(p)
        P = numpy.lib.recfunctions.merge_arrays(
            (P, p_integrated),
            asrecarray=True, flatten=True)
        P = pandas.DataFrame(P, index=pandas.Index(self.ages, name='age'))
        P.rename(columns=self.col_names, inplace=True)
        # Order columns.
        P.set_axis(pandas.CategoricalIndex(P.columns,
                                           self.col_names.values(),
                                           ordered=True),
                   axis='columns', inplace=True)
        P.sort_index(axis='columns', inplace=True)
        assert ((P >= 0) | numpy.isclose(P, 0)).all(axis=None)
        assert numpy.isclose(P.sum(axis='columns'), 1).all()
        P.clip(lower=0, inplace=True)
        return P

    def get_hazard_infection(self, P):
        '''Compute the hazard of infection from the solution `P`.'''
        # The probability of being in each immune status integrated
        # over age.
        P_total = P.apply(integrate.trapz, args=(self.ages, ))
        # Compute the hazard of infection from the solution.
        return (self.params.transmission_rate * P_total['infectious']
                + self.params.chronic_transmission_rate * P_total['chronic'])

    def get_newborn_proportion_immune(self, P):
        '''Compute the proportions of newborns who have maternal immunity from
        the solution `P`.'''
        # The birth rate from moms in each immune status and age.
        births = P.mul(self.params.hazard_birth, axis='index')
        # The birth rate from moms in each immune status.
        births_total = births.apply(integrate.trapz, args=(self.ages, ))
        return (births_total[list(buffalo.Buffalo.has_antibodies)].sum()
                / births_total.sum())

    def solve_objective(self, x):
        '''This is called by `optimize.root_scalar()` to find
        the equilibrium `hazard_infection`.'''
        P = self.solve_step(x)
        return (self.get_hazard_infection(P),
                self.get_newborn_proportion_immune(P))

    def solve(self):
        # TODO: Should I use log for x[0] & logit for x[1]
        #       to implement the constraints?
        x_guess = (15, 0.95)
        x_sol = optimize.fixed_point(self.solve_objective,
                                     x_guess)
        P = self.solve_step(x_sol)
        return ProbabilityInterpolant(P)


class CacheParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `_probability_interpolant()`
    so that it can be efficiently cached.'''
    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        attrs = {'antibody_gain_hazard_alpha',
                 'antibody_gain_hazard_beta',
                 'antibody_gain_hazard_time_max',
                 'antibody_gain_hazard_time_min',
                 'antibody_loss_hazard',
                 'birth_peak_time_of_year',
                 'birth_seasonal_coefficient_of_variation',
                 'chronic_recovery_mean',
                 'chronic_recovery_shape',
                 'chronic_transmission_rate',
                 'female_probability_at_birth',
                 'maternal_immunity_duration_mean',
                 'maternal_immunity_duration_shape',
                 'probability_chronic',
                 'progression_mean',
                 'progression_shape',
                 'recovery_mean',
                 'recovery_shape',
                 'start_time',
                 'transmission_rate'}
        for attr in attrs:
            setattr(self, attr, float(getattr(params, attr)))


# The function is very slow because of the calls to
# `Solver()`, so the values are cached to disk with
# `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir)


@_cache.cache
def _probability_interpolant(params):
    return Solver(params).solve()


def probability_interpolant(params):
    '''Get the `ProbabilityInterpolant()` that can be interpolated
    onto ages as needed.'''
    return _probability_interpolant(CacheParameters(params))


def probability(age, params):
    '''The probability of being in each immune status at age `a`.'''
    return probability_interpolant(params)(age)
