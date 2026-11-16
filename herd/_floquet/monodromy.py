r'''The McKendrick–von Foerster age-structured PDE model
for the density u(t, a) of buffalo of age a at time t is
(d/dt + d/da) u(t, a) = - d(a) u(t, a),
u(t, 0) = \int_0^{inf} b(t, a) u(t, a) da
u(0, a) = u_0(a).
The birth rate `b(t, a)`, which is the product of
`female_probability` and `birth.WaitingTime().hazard(t, a)` in our model,
is periodic in `t` with period T = 1 year.

After discretizing in age, the monodromy matrix Phi(T) is the
fundamental solution of the PDEs after one period, starting from the
initial condition
Phi(0) = I,
the identity matrix.  The fundamental solution requires solving the
matrix-valued version of the orignal PDE.

Here, the matrix-valued PDE is solved using the Crank–Nicolson method
on characteristics, following the work of Fabio Milner.'''

import numpy
import scipy

from .. import _period, events, utility
from . import default


class _Solution:
    '''A `_Solution()` is a sequence of length `1 + order`, with
    `solution[0]` storing the solution at the current time step,
    `solution[1]` storing the solution 1 time steps ago,
    ...
    `solution[order]` storing the solution `order` time steps ago.
    Calling `_Solution.cycle()` in each time iteration cycles the
    elements of `solution` so that its elements stay in the above
    order at the next time step:
    the old `solution[0]` becomes the new `solution[1]`;
    ...
    the old `solution[order - 1]` becomes the new `solution[order]`;
    the old `solution[order]` becomes the new `solution[0]`,
    recycled and ready to be set to the value of the solution at the
    new time step.'''
    def __init__(self, order, shape):
        self._array = numpy.empty((1 + order, ) + shape)
        # '_front' points to the `0` element in the current iteration.
        self._front = 0

    def __getitem__(self, i):
        '''Get the solution `i` time steps ago.'''
        j = (i + self._front) % self._array.shape[0]
        return self._array[j]

    def cycle(self):
        '''Move the entries forward one position,
        wrapping the last entry to the front.'''
        # Decrement `_front` by 1 and wrap around if it's negative.
        self._front = (self._front - 1) % self._array.shape[0]


class Solver:
    '''Solve the monodromy problem.'''
    # Crank–Nicolson is 1st order because the solution at t_n
    # depends on the solution at t_{n - 1}.
    _order = 1

    def __init__(self, params, step=default.STEP, age_max=default.AGE_MAX):
        self.params = params
        self.step = step
        self.ages = utility.arange(0, age_max, step, endpoint=True)
        period = _period.get_period()
        self._t = utility.arange(0, period, step, endpoint=True)
        mortality_waiting_time = events.mortality.WaitingTime(self.params)
        self._matrices = self._build_matrices(mortality_waiting_time)

    def _build_matrices(self, mortality_waiting_time):
        r'''The Crank–Nicolson method is
        (1 + dt / 2 * d_0) * u_0^n
        = f \sum_j (b_j^n u_j^n + b_{j + 1}^n u_{j + 1}^n) * dt / 2
          + f \sum_j (b_j^{n - 1} u_j^{n - 1}
                      + b_{j + 1}^{n - 1} u_{j + 1}^{n - 1}) * dt / 2
        (u_i^n - u_{i - 1}^{n - 1}) / dt
        = - (d_i * u_i^n + d_{i - 1} * u_{i - 1}^{n - 1}) / 2,
        for i = 1, 2, ....
        This can be written as
        A_{next} @ u^n = A_{curr} @ u^{n - 1},
        with
        A_{next} = M_{next} - b^n * B,
        A_{curr} = M_{curr} + b^{n - 1} * B,
        M_{next} = I - dt / 2 * D,
        M_{curr} = L @ (I + dt / 2 * D),
        L[i, i - 1] = L[-1, -1] = 1,
        D = Diag(-d),
        B[0] = f * dt * [1, 1, 1, ..., 1, 1, 1].
        The first row of B implements the composite endpoint rule for
        the birth integral, along with scaling by
        `female_probability`.'''
        I = scipy.sparse.eye_array(len(self.ages))
        L = scipy.sparse.coo_array((len(self.ages),) * 2)
        L.setdiag(1, -1)
        L[-1, -1] = 1
        d = mortality_waiting_time.hazard(self.ages)
        D = scipy.sparse.diags_array(-d)
        B = scipy.sparse.coo_array((len(self.ages),) * 2)
        B[0, :] = self.params.female_probability * self.step / 2
        return {
            'M_next': scipy.sparse.csr_array(I - self.step / 2 * D),
            'M_curr': scipy.sparse.csr_array(L @ (I + self.step / 2 * D)),
            'B': scipy.sparse.csr_array(B),
        }

    def _B_next(self, t_n, state):
        b_next = state['birth_rate'](t_n, self.ages, out=state['b_next'])
        return b_next * self._matrices['B']

    def _step(self, t_n, solution, state):
        '''Do a Crank–Nicolson step.'''
        B_curr = state['B_next']
        A_curr = self._matrices['M_curr'] + B_curr
        B_next = state['B_next'] = self._B_next(t_n, state)
        A_next = self._matrices['M_next'] - B_next
        solution[0][:] = scipy.sparse.linalg.spsolve(A_next,
                                                     A_curr @ solution[1])

    def _set_initial_condition(self, solution_0):
        '''The initial condition for the fundamental solution is the
        identity matrix.'''
        # Avoid creating a new matrix, like `numpy.eye()` does.
        solution_0[:] = 0
        numpy.fill_diagonal(solution_0, 1)

    def solve(self, birth_scaling):
        '''Find the monodromy matrix Phi(T), where T is the period.'''
        if len(self._t) == 0:
            return None
        # The fundamental solution is an `n_ages` x `n_ages` matrix.
        # One matrix for the current time step, plus one for each
        # order of the solver.
        solution = _Solution(self._order, (len(self.ages),) * 2)
        self._set_initial_condition(solution[0])
        if len(self._t) == 1:
            return solution[0]
        birth_waiting_time = events.birth.WaitingTime(self.params,
                                                      _scaling=birth_scaling)
        state = {
            'birth_rate': birth_waiting_time.hazard,
            'b_next': numpy.empty(len(self.ages)),  # Reuse array
        }
        # Use as `B_curr` in the next time step
        state['B_next'] = self._B_next(self._t[0], state)
        for t_n in self._t[1:]:
            solution.cycle()
            self._step(t_n, solution, state)
        return solution[0]  # The solution at the final time
