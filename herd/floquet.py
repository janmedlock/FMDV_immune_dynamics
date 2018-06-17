import numpy
from scipy.integrate import trapz
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

from . import birth
from . import mortality
from . import dominant_eigen
from .shelved import Shelved


def _arange(start, stop, step, endpoint=False):
    '''`numpy.arange()` that can optionally include the right endpoint `stop`.'''
    val = numpy.arange(start, stop, step)
    if endpoint:
        if not numpy.isclose(val[-1], stop):
            val = numpy.hstack((val, stop))
    return val


class _Solver:
    '''Find the population growth rate and stable age structure by solving
    for the fundamental solution Phi(period) with
    dPhi/dt = (B(t) + T) Phi,
    Phi(0) = I.
    Its dominant eigenvalue and eigenvector give
    the population growth rate and the stable age structure.
    The ODEs are solved using a Crank–Nicolson method.
    '''
    _period = 1

    def __init__(self, parameters, agemax=25, agestep=0.01):
        self._parameters = parameters
        self.ages = _arange(0, agemax, agestep, endpoint=True)
        self._birthRV = birth.gen(self._parameters, _scaling=1)
        mortalityRV = mortality.gen(self._parameters)
        mortality_rate = mortalityRV.hazard(self.ages)
        # Initial guess for eigenvector.
        self._v0 = mortalityRV.sf(self.ages)
        # The Crank–Nicolson method
        # (u_j^i - u_{j - 2}^{i - 2}) / 2 / dt
        # = - d_{j - 1} * (u_j^i + u_{j - 2}^{i - 2}) / 2,
        # gives
        # u_j^i = (1 - dt * d_{j - 1}) / (1 + dt * d_{j - 1})
        #         * u_{j - 2}^{i - 2}.
        tstart = self._parameters.start_time
        tstep = agestep
        self._t = _arange(tstart, tstart + self._period, tstep, endpoint=True)
        # In the ODE solver, T2 gets multiplied by Phi[i - 2].
        T2 = lil_matrix((len(self.ages), ) * 2)
        diag2 = (1 - tstep * mortality_rate) / (1 + tstep * mortality_rate)
        T2.setdiag(diag2[1 : -1], -2)
        T2[-1, -1] = diag2[-1]  # Last age group doesn't age out.
        # For ages j = 1 and N - 1, use implicit Euler.
        # T1 gets multiplied by Phi[i - 1].
        T1 = lil_matrix((len(self.ages), ) * 2)
        diag1 = 1 / (1 + tstep * mortality_rate)
        T1[1, 0] = diag1[1]
        T1[-1, -2] = diag1[-1]  # Next to last age group doesn't age out.
        # Implicit Euler for the first time step.
        # T_euler gets multiplied by Phi[i - 1].
        T_euler = lil_matrix((len(self.ages), ) * 2)
        T_euler.setdiag(diag1[1 : ], -1)
        T_euler[-1, -1] = diag1[-1]  # Last age group doesn't age out.
        # Convert to CSR for fast multiply.
        self._T2 = T2.asformat('csr')
        self._T1 = T1.asformat('csr')
        self._T_euler = T_euler.asformat('csr')

    def _find_monodromy(self, birth_scaling):
        '''Find the fundamental solution over one period.'''
        Phi = numpy.zeros((len(self._t), ) + (len(self.ages), ) * 2)
        Phi[0] = numpy.eye(len(self.ages))
        for (i, t_i) in enumerate(self._t[1 : ], 1):
            # Aging & mortality.
            if i == 1:
                # Use implicit Euler for the first time step.
                Phi[i] = self._T_euler @ Phi[i - 1]
            else:
                # Crank–Nicolson with implicit Euler for j = 1, -1.
                Phi[i] = (self._T2 @ Phi[i - 2] + self._T1 @ Phi[i - 1])
            # Birth.
            # Composite trapezoid rule at t = t_i.
            b = (birth_scaling
                 * self._parameters.female_probability_at_birth
                 * self._birthRV.hazard(t_i, self.ages))
            Phi[i, 0] = trapz(b[:, numpy.newaxis] * Phi[i], self.ages, axis=1)
        return Phi[-1]

    def _find_dominant_eigen(self, birth_scaling, return_eigenvector=True):
        '''Find the dominant Floquet exponent
        (the one with the largest real part)
        and its corresponding eigenvector.'''
        PhiT = self._find_monodromy(birth_scaling)
        # Finding the matrix B = log(Phi(T)) / T is very expensive,
        # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
        # and convert.
        rho0, v0 = dominant_eigen.find(PhiT, which='LM', v0=self._v0)
        # Try to speed up the next call of the eigenvalue solver.
        self._v0 = v0
        # rho0 is the dominant (largest magnitude) Floquet multiplier.
        # mu0 is the dominant (largest real part) Floquet exponent.
        # They are related by rho0 = exp(mu0 * T).
        mu0 = numpy.log(rho0) / self._period
        if return_eigenvector:
            # v0 is the eigenvector for both rho0 and mu0.
            # Normalize it to integrate to 1.
            v0 /= trapz(v0, self.ages)
            return (mu0, v0)
        else:
            return mu0

    def _find_growth_rate(self, birth_scaling):
        '''Find the population growth rate.'''
        return self._find_dominant_eigen(birth_scaling,
                                         return_eigenvector=False)

    def find_birth_scaling(self):
        '''Find the birth scaling that gives population growth rate r = 0.'''
        a = 0
        # We know that at the lower limit a = 0,
        # `_find_growth_rate(0, ...) < 0`,
        # so we need to find an upper limit `b`
        # with `_find_growth_rate(b, ...) >= 0`.
        b = 1
        while self._find_growth_rate(b) < 0:
            a = b
            b *= 2
        return brentq(self._find_growth_rate, a, b)

    def find_stable_age_structure(self, birth_scaling=None):
        '''Find the stable age structure.'''
        if birth_scaling is None:
            birth_scaling = self.find_birth_scaling()
        r, v = self._find_dominant_eigen(birth_scaling)
        assert numpy.isclose(r, 0, atol=1e-6), 'Nonzero growth rate.'
        return (self.ages, v)


# The functions below are slow to compute, so the values are cached to
# disk with `shelved.Shelved()` so they are only computed once.


@Shelved('birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_birth_scaling(parameters, _solver=None, *args, **kwargs):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    if _solver is None:
        _solver = _Solver(parameters, *args, **kwargs)
    return _solver.find_birth_scaling()


@Shelved('start_time',
         'birth_peak_time_of_year',
         'birth_seasonal_coefficient_of_variation',
         'female_probability_at_birth')
def find_stable_age_structure(parameters, *args, **kwargs):
    '''Find the stable age structure.'''
    solver = _Solver(parameters, *args, **kwargs)
    # The birth scaling is probably already in the cache...
    birth_scaling = find_birth_scaling(parameters, _solver=solver,
                                       *args, **kwargs)
    return solver.find_stable_age_structure(birth_scaling)
