'''Solver for the probability of being in each immune state.'''

import pathlib

import joblib
import numpy
import numpy.lib.recfunctions
import pandas
from scipy import integrate, optimize, sparse, special

from herd import (antibody_gain, antibody_loss, birth, buffalo,
                  chronic_recovery, maternal_immunity_waning,
                  mortality, parameters, progression, recovery,
                  utility)
from . import blocks


def _rec_fromkwds(**kwds):
    '''Build a `numpy.recarray()` from `kwds`.'''
    return numpy.rec.fromarrays(
        numpy.broadcast_arrays(*kwds.values()),
        dtype=[(k, float) for k in kwds.keys()])


class _Params:
    '''Dummy object to pass parameters from `Solver()` to
    `blocks._Block()`s.'''
    def __init__(self, params, ages, ages_mid, step, length_ode, length_pde):
        self.step = step
        self.length_ode = length_ode
        self.length_pde = length_pde
        # Get the pdf for these random variables.
        rndvrs_pdf = {
            'progression': progression.gen(params),
            'recovery': recovery.gen(params),
            'chronic_recovery': chronic_recovery.gen(params),
        }
        pdf = {k: self._get_pdf(RV, ages)
               for (k, RV) in rndvrs_pdf.items()}
        self.pdf = _rec_fromkwds(**pdf)
        # Get the survival for these random variables.
        rndvrs_survival = {
            **rndvrs_pdf,  # The entries in `rndvrs_pdf`.
            'mortality': mortality.gen(params),
        }
        survival = {k: RV.sf(ages)
                    for (k, RV) in rndvrs_survival.items()}
        self.survival = _rec_fromkwds(**survival)
        # Get the hazard for these RVs.
        rndvrs_hazard = {
            'mortality': rndvrs_survival['mortality'],
            'maternal_immunity_waning': maternal_immunity_waning.gen(params),
            'antibody_gain': antibody_gain.gen(params),
            'antibody_loss': antibody_loss.gen(params),
        }
        hazard = {k: rndvr.hazard(ages_mid)
                  for (k, rndvr) in rndvrs_hazard.items()}
        hazard['infection'] = 1  # Dummy value. Set on calls to `solve_step()`.
        # Other parameters.
        self.hazard = _rec_fromkwds(**hazard)
        self.female_probability_at_birth = params.female_probability_at_birth
        self.probability_chronic = params.probability_chronic
        self.transmission_rate = params.transmission_rate
        self.chronic_transmission_rate = params.chronic_transmission_rate
        self.lost_immunity_susceptibility = params.lost_immunity_susceptibility
        # The birth hazard is needed at `ages`, not `ages_mid`.
        self.hazard_birth = self._hazard_birth(ages)
        # Dummy value. Set on calls to `Solver.solve_step()`.
        self.newborn_proportion_immune = 1

    @staticmethod
    def _get_pdf(rndvr, ages):
        '''Use the survival function `rndvr.sf()` to get the PDF over each
        interval in `ages`. The PDF `rndvr.pdf()` at each age might
        change too quickly to capture the difference over the age
        interval.'''
        # Use 0 at the first age.
        pdf = numpy.hstack(
            [0, - numpy.diff(rndvr.sf(ages)) / numpy.diff(ages)]
        )
        return pdf

    def _hazard_birth(self, ages):
        hazard = birth.hazard_no_seasonality(
            self.female_probability_at_birth, ages)
        # Scale so that the population growth rate is 0.
        hazard /= integrate.trapz(hazard * self.survival.mortality,
                                  ages)
        hazard *= self.female_probability_at_birth
        return hazard


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
        self.ages_mid = (self.ages[:-1] + self.ages[1:]) / 2
        self.length_ode = self.length_pde = len(self.ages)
        self.params = _Params(params, self.ages, self.ages_mid, self.step,
                              self.length_ode, self.length_pde)
        if not _skip_blocks:
            self.blocks = {
                Block.X: Block(self.params)
                for Block in blocks.Blocks
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
            vars_ = blocks.vars_ode + blocks.vars_pde
            return [cls.stack(rows.get(k))
                    for k in vars_]

    @staticmethod
    def _unstack_and_label(x, x_vars):
        # Split into columns.
        X = numpy.hsplit(x, len(x_vars))
        # Add labels.
        return _rec_fromkwds(**dict(zip(x_vars, X)))

    def unstack(self, Pp):
        '''Unstack the P_X and p_X vectors.'''
        # `vars_ode` are first, then `vars_pde`.
        split = len(blocks.vars_ode) * self.length_ode
        P = self._unstack_and_label(Pp[:split], blocks.vars_ode)
        p = self._unstack_and_label(Pp[split:], blocks.vars_pde)
        return (P, p)

    def stack_sparse(self, rows, format='csr'):
        return sparse.bmat(self.stack(rows), format=format)

    def get_A(self):
        A = self.stack_sparse({var: block.A_X
                               for (var, block) in self.blocks.items()})
        assert numpy.isfinite(A.data).all()
        return A

    def get_b(self):
        b = self.stack_sparse({var: [block.b_X]
                               for (var, block) in self.blocks.items()})
        return b

    def integrate_pde_vars(self, p):
        # Integrate the p_X variables over r.
        p_integrated = {n: self.blocks[n].integrate(p[n])
                        for n in p.dtype.names}
        return _rec_fromkwds(**p_integrated)

    @staticmethod
    def transform(x):
        '''Transform the solver variables to enforce the constraints x[0] ≥ 0
        and 0 ≤ x[1] ≤ 1.'''
        return numpy.array((numpy.log(x[0]), special.logit(x[1])))

    @staticmethod
    def transform_inverse(y):
        '''Untransform the solver variables.'''
        return numpy.array((numpy.exp(y[0]), special.expit(y[1])))

    def solve_step(self, y_curr):
        x_curr = self.transform_inverse(y_curr)
        if self.debug:
            msg = 'hazard_infection={:g}, newborn_proportion_immune={:g}'
            print(msg.format(*x_curr))
        (self.params.hazard.infection,
         self.params.newborn_proportion_immune) = x_curr
        self.update_blocks()
        A = self.get_A()
        b = self.get_b()
        Pp = sparse.linalg.spsolve(A, b)
        (P, p) = self.unstack(Pp)
        p_integrated = self.integrate_pde_vars(p)
        P = numpy.lib.recfunctions.merge_arrays(
            (P, p_integrated),
            asrecarray=True, flatten=True)
        P = pandas.DataFrame(P, index=pandas.Index(self.ages, name='age'))
        P.rename(columns=blocks.NAMES, inplace=True)
        # Order columns.
        P.set_axis(pandas.CategoricalIndex(P.columns,
                                           blocks.NAMES.values(),
                                           ordered=True),
                   axis='columns', inplace=True)
        P.sort_index(axis='columns', inplace=True)
        return P

    def get_hazard_infection(self, P):
        '''Compute the hazard of infection from the solution `P`.'''
        # The probability of being in each immune status integrated
        # over age.
        P_total = P.apply(integrate.trapz,
                          args=(self.ages, ))
        haz = (self.params.transmission_rate * P_total['infectious']
               + self.params.chronic_transmission_rate * P_total['chronic'])
        assert (haz >= 0) | numpy.isclose(haz, 0)
        haz = numpy.clip(haz, 0, None)
        return haz

    def get_newborn_proportion_immune(self, P):
        '''Compute the proportions of newborns who have maternal immunity from
        the solution `P`.'''
        # The birth rate from moms in each immune status and age.
        births = P.mul(self.params.hazard_birth,
                       axis='index')
        # The birth rate from moms in each immune status.
        births_total = births.apply(integrate.trapz,
                                    args=(self.ages, ))
        imm = (births_total[list(buffalo.Buffalo.has_antibodies)].sum()
               / births_total.sum())
        assert (imm >= 0) | numpy.isclose(imm, 0)
        imm = numpy.clip(imm, 0, None)
        return imm

    def solve_objective(self, y_curr):
        '''This is called by `optimize.fixed_point()` to find the equilibrium
        `hazard_infection` and `newborn_proportion_immune`.'''
        P = self.solve_step(y_curr)
        P.clip(lower=0, inplace=True)
        x_next = (self.get_hazard_infection(P),
                  self.get_newborn_proportion_immune(P))
        y_next = self.transform(x_next)
        return y_next

    def solve(self):
        '''Find the solution.'''
        x_guess = (100, 0.9)
        y_sol = optimize.fixed_point(self.solve_objective,
                                     self.transform(x_guess))
        P = self.solve_step(y_sol)
        assert ((P >= 0) | numpy.isclose(P, 0)).all(axis=None)
        P.clip(lower=0, inplace=True)
        return P


# `Solver.solve()` is very slow, so the values are cached to disk with
# `joblib.Memory()` so that they are only computed once. Set up the
# cache in a subdirectory of the directory that this source file is
# in.
_cache_path = pathlib.Path(__file__).with_name('_cache')
_cache = joblib.Memory(_cache_path, verbose=1)


@_cache.cache(ignore=['debug'])
def _solve(params, debug):
    return Solver(params, debug=debug).solve()


class _CacheParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''

    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        attrs = {'antibody_gain_hazard',
                 'antibody_loss_hazard',
                 'birth_peak_time_of_year',
                 'birth_seasonal_coefficient_of_variation',
                 'chronic_recovery_mean',
                 'chronic_recovery_shape',
                 'chronic_transmission_rate',
                 'female_probability_at_birth',
                 'lost_immunity_susceptibility',
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


def solve(params, debug=False):
    '''Find the solution.'''
    return _solve(_CacheParameters(params), debug)


def _solve_check_call_in_cache(params):
    '''Check if the solution for `params` is in the cache.'''
    # The value of `debug` is ignored by the cache.
    return _solve.check_call_in_cache(_CacheParameters(params),
                                      debug=False)


def get_optimizer(params, cached_only=False, debug=False):
    '''Get the optimizer from Solver().'''
    if cached_only and not _solve_check_call_in_cache(params):
        return dict(
            hazard_infection=None,
            newborn_proportion_immune=None,
        )
    prob = solve(params, debug=debug)
    solver = Solver(params, debug=debug, _skip_blocks=True)
    return dict(
        hazard_infection=solver.get_hazard_infection(prob),
        newborn_proportion_immune=solver.get_newborn_proportion_immune(prob),
    )
