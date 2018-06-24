'''The McKendrick–von Foerster age-structured PDE model
for the density u(t, a) of buffalo of age a at time t is
(d/dt + d/da) u(t, a) = - d(a) u(t, a),
u(t, 0) = \int_0^{inf} b(t, a) u(t, a) da
u(0, a) = u_0(a).
The birth rate `b(t, a)`, which is the product of
`female_probability_at_birth` and `birth.gen.hazard(t, a)` in our
model, is periodic in `t` with period T = 1 year.

After discretizing in age, the monodromy matrix Phi(T) is the
fundamental solution of the PDEs after one period, starting from the
initial condition
Phi(0) = I,
the identity matrix.  The fundamental solution requires solving the
matrix-valued version of the orignal PDE.

Here, the matrix-valued PDE is solved using the Crank–Nicolson method
on characteristics, following the work of Fabio Milner.

So that the cache keys only depend on the relevant parts of
`herd.parameters.Parameters()` and so that the setup can be reused in
`_find_birth_scaling()`, the solver is called in 3 steps:
>>> solver_parameters = monodromy.Parameters(parameters)
>>> solver = monodromy.Solver(solver_parameters, agemax, agestep)
>>> PhiT = solver.solve(birth_scaling)
where `parameters` is a `herd.parameters.Parameters()` instance.'''

import numpy
from scipy.sparse import lil_matrix

from herd import birth, mortality, parameters
from herd.utility import arange
from herd.floquet import floquet, _monodromy


class Parameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''
    def __init__(self, parameters):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        # Normalize `parameters.birth_peak_time_of_year` by making
        # it relative to `parameters.start_time` and then modulo
        # `_period` so that it is in [0, period).
        self.birth_normalized_peak_time_of_year = float(
            (parameters.birth_peak_time_of_year - parameters.start_time)
            % floquet._period)
        self.birth_seasonal_coefficient_of_variation = float(
            parameters.birth_seasonal_coefficient_of_variation)
        self.female_probability_at_birth = float(
            parameters.female_probability_at_birth)


class Solver:
    '''Solve the monodromy problem.'''
    def __init__(self, solver_parameters, agemax, agestep):
        self.parameters = solver_parameters
        self.ages = arange(0, agemax, agestep, endpoint=True)
        tstep = agestep
        self.t = arange(0, floquet._period, tstep, endpoint=True)
        mortalityRV = mortality.from_param_values()
        self._init_crank_nicolson(tstep, mortalityRV)
        self._init_births(agestep)

    def _init_implicit_euler(self, tstep, mortalityRV):
        # The implicit Euler method is
        # (u_i^n - u_{i - 1}^{n - 1}) / dt = - d_i * u_i^n.
        # This can be written as
        # u^1 = M @ u^0,
        M_implicit_euler = lil_matrix((self.ages.size, self.ages.size))
        # with
        # M[i, i - 1] = 1 / (1 + dt * d_i),
        diag = 1 / (1 + tstep * mortalityRV.hazard(self.ages))
        M_implicit_euler.setdiag(diag[1 : ], -1)
        # and, to prevent the last age group from aging out of the population,
        # M[-1, -1] = 1 / (1 + dt * d_{-1}).
        M_implicit_euler[-1, -1] = diag[-1]
        # Convert to CSR for fast left multiplication.
        self._M_implicit_euler = M_implicit_euler.tocsr()

    def _init_crank_nicolson(self, tstep, mortalityRV):
        # The Crank–Nicolson method is
        # (u_i^n - u_{i - 2}^{n - 2}) / 2 / dt
        # = - d_{i - 1} * (u_i^n + u_{i - 2}^{n - 2}) / 2,
        # for i = 2, 3, ...,
        # with implicit Euler for i = 1,
        # (u_1^n - u_0^{n - 1}) / dt = - d_1 * u_1^n.
        # This can be written as
        # u^n = M_2 @ u^{n - 2} + M_1 @ u^{n - 1},
        M_crank_nicolson_2 = lil_matrix((self.ages.size, self.ages.size))
        M_crank_nicolson_1 = lil_matrix((self.ages.size, self.ages.size))
        # with
        # M_2[i, i - 2] = (1 - dt * d_{i - 1}) / (1 + dt * d_{i - 1});
        diag = ((1 - tstep * mortalityRV.hazard(self.ages))
                / (1 + tstep * mortalityRV.hazard(self.ages)))
        M_crank_nicolson_2.setdiag(diag[1 : -1], -2)
        # to prevent the last age group from aging out of the population,
        # M_2[-1, -1] = (1 - dt * d_{-1}) / (1 + dt * d_{-1});
        M_crank_nicolson_2[-1, -1] = diag[-1]
        # M_1[1, 0] = 1 / (1 + dt * d_i);
        diag[:] = 1 / (1 + tstep * mortalityRV.hazard(self.ages))
        M_crank_nicolson_1[1, 0] = diag[1]
        # and, to prevent the next to last age group from aging out,
        # M_1[-1, -2] = 1 / (1 + dt * d_{-1}).
        M_crank_nicolson_1[-1, -2] = diag[-1]
        # Convert to CSR for fast left multiplication.
        self._M_crank_nicolson_2 = M_crank_nicolson_2.tocsr()
        self._M_crank_nicolson_1 = M_crank_nicolson_1.tocsr()
        # The first time step, n = 1, uses implicit Euler:
        self._init_implicit_euler(tstep, mortalityRV)

    def _init_births(self, agestep):
        # The trapezoid rule for the birth integral for i = 0,
        # u_0^n = \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) * da / 2.
        # This can be written as
        # u_0^n = (v * b^n) @ u^n,
        # with
        # v = da * [0.5, 1, 1, ..., 1, 1, 0.5].
        # Put `female_probability_at_birth` in there, too, for
        # simplicity & efficiency.
        self._v_trapezoid = (
            self.parameters.female_probability_at_birth
            * agestep
            * numpy.hstack((0.5, numpy.ones(self.ages.size - 2), 0.5)))

    def solve(self, birth_scaling):
        '''Find the monodromy matrix Phi(T), where T is the period.'''
        # Set up birth rate.
        birthRV = birth.from_param_values(
            self.parameters.birth_normalized_peak_time_of_year,
            self.parameters.birth_seasonal_coefficient_of_variation,
            _scaling=birth_scaling)
        # Call the core solver.
        return _monodromy.solve(self.ages, self.t,
                                self._M_crank_nicolson_2,
                                self._M_crank_nicolson_1,
                                self._M_implicit_euler,
                                self._v_trapezoid,
                                birthRV.hazard)
