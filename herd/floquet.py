import numpy
from scipy import integrate, optimize, sparse

from . import birth
from . import mortality
from . import dominant_eigen
from .shelved import Shelved


class _Solver:
    '''Find the population growth rate and stable age structure by solving
    for the fundamental solution Phi(period) with
    dPhi/dt = (B(t) + T) Phi,
    Phi(0) = I.
    Its dominant eigenvalue and eigenvector give
    the population growth rate and the stable age structure.
    '''
    def __init__(self, parameters, agemax=25, agestep=0.01):
        self._parameters = parameters
        self.ages = numpy.arange(0, agemax, agestep)
        if not numpy.isclose(self.ages[-1], agemax):
            self.ages = numpy.hstack((self.ages, agemax))
        self._N = len(self.ages)
        self._period = 1
        # Birth
        # The first row, B[0], is the birth hazard
        # times the probability of female birth.
        self._birthRV = birth.gen(self._parameters, _scaling=1)
        B = sparse.lil_matrix((self._N, self._N))
        # Establish sparsity pattern.  This row will get updated at each time.
        B[0] = 1
        # Mortality and aging
        mortalityRV = mortality.gen(self._parameters)
        mortality_rate = mortalityRV.hazard(self.ages)
        # No aging out of the last age group.
        aging_rate = numpy.hstack((1 / numpy.diff(self.ages), 0))
        T = sparse.dia_matrix((self._N, self._N))
        T.setdiag(- mortality_rate - aging_rate, 0)
        T.setdiag(aging_rate[: -1], -1)
        # Convert to CSR for fast multiply.
        self._B = B.asformat('csr')
        self._T = T.asformat('csr')
        # Initial guess for eigenvector.
        self._v0 = mortalityRV.sf(self.ages)

    def _ODEs(self, phi, t, birth_scaling):
        '''The right-hand side of the matrix ODE for vector `phi`,
        a flattened version of the fundamental solution matrix `Phi`.'''
        # Update the birth rate for time `t`.
        self._B[0] = (birth_scaling
                      * self._parameters.female_probability_at_birth
                      * self._birthRV.hazard(t, self.ages))
        # Convert from vector to matrix.
        Phi = phi.reshape((self._N, ) * 2)
        # Compute the deriviative.
        dPhi_dt = (self._B + self._T) @ Phi
        # Convert from matrix to vector.
        return dPhi_dt.reshape(-1)

    def _find_monodromy(self, birth_scaling):
        '''Find the fundamental solution over one period.'''
        t = (self._parameters.start_time,
             self._parameters.start_time + self._period)
        Phi0 = numpy.eye(self._N)
        # Convert from matrix to vector.
        phi0 = Phi0.reshape(-1)
        phi, info = integrate.odeint(self._ODEs, phi0, t,
                                     args=(birth_scaling, ),
                                     full_output=True,
                                     mxstep=100000)
        assert info['message'] == 'Integration successful.', info['message']
        # Convert from vector to matrix.
        PhiT = phi[-1].reshape((self._N, ) * 2)
        return PhiT

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
            v0 /= integrate.trapz(v0, self.ages)
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
        return optimize.brentq(self._find_growth_rate, a, b)

    def find_stable_age_structure(self, birth_scaling=None):
        '''Find the stable age structure.'''
        if birth_scaling is None:
            birth_scaling = self.find_birth_scaling()
        r, v = self._find_dominant_eigen(birth_scaling)
        assert numpy.isclose(r, 0, atol=1e-6), 'Nonzero growth rate.'
        return (self.ages, v)


# The functions below are slow to compute, so we cache the values to
# disk with `shelved.Shelved()` so we only need to compute them once.


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
