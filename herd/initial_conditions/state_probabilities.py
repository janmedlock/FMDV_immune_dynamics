from functools import partial

import numpy
from scipy.integrate import quadrature

from herd import chronic_recovery, maternal_immunity_waning


_quadrature_options = dict(tol=1e-6, rtol=1e-6, maxiter=2000)


def _vectorize(**kwds):
    '''Decorator to vectorize a scalar function.'''
    # Just make `numpy.vectorize()` easier to use as a decorator.
    return partial(numpy.vectorize, **kwds)


def M_logprob(age, hazard_infection, params):
    maternal_immunity_waningRV = maternal_immunity_waning.gen(params)
    return maternal_immunity_waningRV.logsf(age)


def M_prob(age, hazard_infection, params):
    '''The probability of having maternal antibodies at age `a`.'''
    return numpy.exp(M_logprob(age, hazard_infection, params))


def _S_logprob_integrand(b, hazard_infection, maternal_immunity_waningRV):
    '''The integrand is
    Prob{Transitioning from M to S at age b}
      * exp(hazard_infection * b)
    = exp(\log Prob{Transitioning from M to S at age b}
          + hazard_infection * b).'''
    return numpy.exp(maternal_immunity_waningRV.logpdf(b)
                     + hazard_infection * b)


# Make the function able to handle vector-valued `age`.
@_vectorize(otypes=[float])
def _S_logprob_integral(age, hazard_infection, params):
    maternal_immunity_waningRV = maternal_immunity_waning.gen(params)
    val, _ = quadrature(_S_logprob_integrand, 0, age,
                        args=(hazard_infection, maternal_immunity_waningRV),
                        **_quadrature_options)
    return val


def S_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of being susceptible at age `a`.
    This is
    \log \int_0^a Prob{Transitioning from M to S at age b}
                  * exp(- hazard_infection * (a - b)) db
    = \log \int_0^a Prob{Transitioning from M to S at age b}
                    * exp(hazard_infection * b) db
      - hazard_infection * a.'''
    assert hazard_infection >= 0, hazard_infection
    # Handle log(0).
    return numpy.ma.filled(
        numpy.ma.log(_S_logprob_integral(age, hazard_infection, params))
        - hazard_infection * age,
        - numpy.inf)


def S_prob(age, hazard_infection, params):
    '''The probability of being susceptible at age `a`.'''
    return numpy.exp(S_logprob(age, hazard_infection, params))


def _C_logprob_integrand(b, a, hazard_infection, params, chronic_recoveryRV):
    return numpy.exp(S_logprob(b, hazard_infection, params)
                     + chronic_recoveryRV.logsf(a - b))


# Make the function able to handle vector-valued `age`.
@_vectorize(otypes=[float])
def _C_logprob_integral(age, hazard_infection, params):
    chronic_recoveryRV = chronic_recovery.gen(params)
    val, _ = quadrature(
        _C_logprob_integrand, 0, age,
        args=(age, hazard_infection, params, chronic_recoveryRV),
        **_quadrature_options)
    return val


def C_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of being chronically infected
    at age `a`.  This is
    \log \int_0^a Prob{Infection at age b}
                  * probabilty_chronic
                  * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * hazard_infection
                    * probabilty_chronic
                    * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * Prob{survival in chronically infected for (a - b)} db
      + \log hazard_infection
      + \log probabilty_chronic.'''
    assert hazard_infection >= 0, hazard_infection
    if params.model != 'chronic':
        # Shortcut to probability = 0.
        return - numpy.inf * numpy.ones_like(age)
    # Handle log(0).
    return numpy.ma.filled(
        numpy.ma.log(_C_logprob_integral(age, hazard_infection, params))
        + numpy.ma.log(hazard_infection)
        + numpy.ma.log(params.probability_chronic),
        - numpy.inf)


def C_prob(age, hazard_infection, params):
    '''The probability of being chronically infected at age `a`.'''
    return numpy.exp(C_logprob(age, hazard_infection, params))


def L_logprob(age, hazard_infection, params):
    '''The logarithm of the probability of having reduced antibodies
    at age `a`.  This is
    \log \int_0^a Prob{Infection at age b}
                  * probabilty_chronic
                  * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * hazard_infection
                    * probabilty_chronic
                    * Prob{survival in chronically infected for (a - b)} db
    = \log \int_0^a Prob{In S at age b}
                    * Prob{survival in chronically infected for (a - b)} db
      + \log hazard_infection
      + \log probabilty_chronic.'''
    assert hazard_infection >= 0, hazard_infection
    if params.model != 'chronic':
        # Shortcut to probability = 0.
        return - numpy.inf * numpy.ones_like(age)
    # Handle log(0).
    return numpy.ma.filled(
        numpy.ma.log(_C_logprob_integral(age, hazard_infection, params))
        + numpy.ma.log(hazard_infection)
        + numpy.ma.log(params.probability_chronic),
        - numpy.inf)


def L_prob(age, hazard_infection, params):
    '''The probability of having reduced antibodies at age `a`.'''
    raise NotImplementedError
