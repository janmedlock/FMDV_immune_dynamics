'''Solver for the probability of being in each immune state.'''

import numpy
import numpy.lib.recfunctions
import pandas
import scipy

from .. import _memory, immune_status, parameters, utility
from . import _blocks, _utility


class Solver:
    '''Crank–Nicolson solver to find the probability of being in each
    compartment as a function of age.'''

    # Step size in both age (a) and residence time (r).
    step = 0.005

    age_max = 20

    def __init__(self, params, debug=False, _skip_blocks=False):
        self.debug = debug
        self.ages = utility.arange(0, self.age_max, self.step,
                                   endpoint=True)
        assert len(self.ages) > 1
        self.length_ode = self.length_pde = len(self.ages)
        self.block_params = _blocks.Parameters(params, self.ages, self.step,
                                               self.length_ode,
                                               self.length_pde)
        self.index = pandas.Index(self.ages, name='age')
        # `pandas.DataFrame()` requires a `list()` indexer
        self.have_offspring_to_maternal_immunity = list(
            immune_status.have_offspring_to_maternal_immunity)
        if not _skip_blocks:
            n_ages = len(self.ages)
            I = scipy.sparse.eye_array(n_ages, format='csr')
            A_coo = scipy.sparse.coo_array((n_ages, n_ages))
            A_coo.setdiag(1, -1)
            A_coo[-1, -1] = 1
            A = A_coo.tocsr()
            self.blocks = {
                Block.X: Block(self.block_params, I, A)
                for Block in _blocks.Blocks
            }

    def update_blocks(self):
        for block in self.blocks.values():
            block.update()

    @classmethod
    def stack(cls, rows):
        '''Stack the P_X and p_X vectors into one big vector.'''
        # Recurse for 2-d arrays.
        if not isinstance(rows, dict):
            return rows
        else:
            # `vars_ode` first, then `vars_pde`.
            vars_ = _blocks.vars_ode + _blocks.vars_pde
            return [cls.stack(rows.get(k))
                    for k in vars_]

    @staticmethod
    def _unstack_and_label(x, x_vars):
        # Split into columns.
        X = numpy.hsplit(x, len(x_vars))
        # Add labels.
        return _utility.recarray_fromkwds(**dict(zip(x_vars, X)))

    def unstack(self, Pp):
        '''Unstack the P_X and p_X vectors.'''
        # `vars_ode` are first, then `vars_pde`.
        split = len(_blocks.vars_ode) * self.length_ode
        P = self._unstack_and_label(Pp[:split], _blocks.vars_ode)
        p = self._unstack_and_label(Pp[split:], _blocks.vars_pde)
        return (P, p)

    def stack_sparse(self, rows, format='csr'):
        return scipy.sparse.bmat(self.stack(rows), format=format)

    def get_M(self):
        M = self.stack_sparse({var: block.M_X
                               for (var, block) in self.blocks.items()})
        assert numpy.isfinite(M.data).all()
        return M

    def get_b(self):
        b = self.stack_sparse({var: [block.b_X]
                               for (var, block) in self.blocks.items()})
        return b

    def integrate_pde_vars(self, p):
        # Integrate the p_X variables over r.
        p_integrated = {n: self.blocks[n].integrate(p[n])
                        for n in p.dtype.names}
        return _utility.recarray_fromkwds(**p_integrated)

    @staticmethod
    def transform(x):
        '''Transform the solver variables to enforce the constraints x[0] ≥ 0
        and 0 ≤ x[1] ≤ 1.'''
        return numpy.array((numpy.log(x[0]), scipy.special.logit(x[1])))

    @staticmethod
    def transform_inverse(y):
        '''Untransform the solver variables.'''
        return numpy.array((numpy.exp(y[0]), scipy.special.expit(y[1])))

    @staticmethod
    def _columns_as_long_name_dtype(dfr):
        return dfr.set_axis(dfr.columns.astype(_blocks.LONG_NAMES_DTYPE),
                            axis='columns')

    def solve_step(self, y_curr):
        x_curr = self.transform_inverse(y_curr)
        (hazard_infection, birth_maternal_immunity_proportion) = x_curr
        if self.debug:
            print(f'{hazard_infection=:g}, '
                  + f'{birth_maternal_immunity_proportion=:g}')
        self.block_params.hazard.infection = hazard_infection
        self.block_params.birth_maternal_immunity_proportion = (
            birth_maternal_immunity_proportion)
        self.update_blocks()
        M = self.get_M()
        b = self.get_b()
        Pp = scipy.sparse.linalg.spsolve(M, b)
        (P, p) = self.unstack(Pp)
        p_integrated = self.integrate_pde_vars(p)
        P = numpy.lib.recfunctions.merge_arrays(
            (P, p_integrated),
            asrecarray=True, flatten=True)
        return (pandas.DataFrame(P, index=self.index)
                .rename(columns=_blocks.SHORT_NAMES_TO_LONG_NAMES)
                .pipe(self._columns_as_long_name_dtype)
                .sort_index(axis='columns'))

    def get_hazard_infection(self, P):
        '''Compute the hazard of infection from the solution `P`.'''
        # The probability of being in each immune status integrated
        # over age.
        P_total = P.sum() * self.step
        haz = (
            ((self.block_params.transmission_rate
              * P_total['infectious'])
             + (self.block_params.carrier_transmission_rate
                * P_total['carrier']))
            / P_total.sum()
        )
        assert (haz >= 0) | numpy.isclose(haz, 0)
        haz = numpy.clip(haz, 0, None)
        return haz

    def get_birth_maternal_immunity_proportion(self, P):
        '''Compute the proportions of newborns who have maternal
        immunity from the solution `P`.'''
        # The birth rate from moms in each immune status and age.
        births = P.mul(self.block_params.hazard.birth, axis='index')
        # The birth rate from mothers in each immune status.
        births_total = births.sum() * self.step
        imm = (
            births_total[self.have_offspring_to_maternal_immunity].sum()
            / births_total.sum()
        )
        assert (imm >= 0) | numpy.isclose(imm, 0)
        imm = numpy.clip(imm, 0, None)
        return imm

    def solve_objective(self, y_curr):
        '''This is called by `scipy.optimize.fixed_point()` to find
        the equilibrium `hazard_infection` and
        `birth_maternal_immunity_proportion`.'''
        P = self.solve_step(y_curr)
        P.clip(lower=0, inplace=True)  # TODO: Examine
        x_next = (self.get_hazard_infection(P),
                  self.get_birth_maternal_immunity_proportion(P))
        y_next = self.transform(x_next)
        return y_next

    def solve(self):
        '''Find the solution.'''
        x_guess = (100, 0.9)
        y_sol = scipy.optimize.fixed_point(self.solve_objective,
                                           self.transform(x_guess))
        P = self.solve_step(y_sol)
        # assert ((P >= 0) | numpy.isclose(P, 0)).all(axis=None)  # TODO
        P.clip(lower=0, inplace=True)  # TODO
        return P


# `Solver.solve()` is very slow, so the values are cached to disk with
# `joblib.Memory()` so that they are only computed once
@_memory.memory.cache(ignore=['debug'])
def solve_cached(params, debug):
    return Solver(params, debug=debug).solve()


class Parameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''

    def __init__(self, params):
        attrs = {
            # The values of these parameters should be floats, so
            # explicitly convert them so the cache doesn't get
            # duplicated keys for the float and int representation of
            # the same number, e.g. `float(0)` and `int(0)`.
            float: ('antibody_gain_hazard',
                    'antibody_loss_hazard',
                    'birth_menarche',
                    'birth_peak_time',
                    'birth_seasonal_coefficient_of_variation',
                    'carrier_recovery_mean',
                    'carrier_recovery_shape',
                    'carrier_transmission_rate',
                    'female_probability',
                    'lost_immunity_susceptibility',
                    'lost_immunity_true_probability',
                    'maternal_immunity_duration_mean',
                    'maternal_immunity_duration_shape',
                    'progression_mean',
                    'progression_shape',
                    'recovered_true_probability',
                    'recovery_mean',
                    'recovery_shape',
                    'recovery_to_carrier_probability',
                    'transmission_rate'),
            # This one should be a boolean.
            bool: ('mortality_age_dependent',),
        }
        for (type_, attrs_type) in attrs.items():
            for attr in attrs_type:
                setattr(self, attr, type_(getattr(params, attr)))


def solve(params, debug=False):
    '''Find the solution.'''
    return solve_cached(Parameters(params), debug)
