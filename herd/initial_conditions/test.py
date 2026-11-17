__all__ = ('SHORT_NAMES', 'Solver', 'vars_ode', 'vars_pde')

from . import _solver
from ._blocks import SHORT_NAMES, vars_ode, vars_pde
from ._solver import Solver


def _in_cache(params):
    '''Check if the solution for `params` is in the cache.'''
    # The value of `debug` is ignored by the cache.
    return _solver.solve_cached.check_call_in_cache(
        _solver.Parameters(params), debug=False
    )


def get_equilibrium(params, cached_only=False, debug=False):
    '''Get the equilibrium values of the variables used in Solver().'''
    if cached_only and not _in_cache(params):
        return {
            'hazard_infection': None,
            'birth_maternal_immunity_proportion': None,
        }
    # Get the solution from the cached solver.
    P = _solver.solve(params, debug=debug)
    solver = _solver.Solver(params, debug=debug, _skip_blocks=True)
    return {
        'hazard_infection':
        solver.get_hazard_infection(P),
        'birth_maternal_immunity_proportion':
        solver.get_birth_maternal_immunity_proportion(P),
    }
