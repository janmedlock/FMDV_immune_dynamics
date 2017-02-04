#!/usr/bin/python3

import numpy
from scipy import sparse
from scipy import integrate

import utility
import birth
import transmission_rate
import chronic_transmission_rate
import parameters


def rhs(Y, t, AM, B, force_of_infection,
        force_of_infection_carriers,
        progression_rate, recovery_rate,
        proportion_carriers, immunity_waning_rate,
        chronic_recovery_rate):
    (S, E, I, C, R) = numpy.hsplit(Y, 5)
    
    N = S + E + I + C + R

    lambda_ = force_of_infection(I)
    lambda_carriers_ = force_of_infection_carriers(C)

    dS = (B(t).dot(N)
          + AM.dot(S)
          - lambda_ * S
          - lambda_carriers_ * S
          + immunity_waning_rate * R)
    dE = (AM.dot(E)
          + lambda_ * S
          + lambda_carriers_ * S
          - progression_rate * E)
    dI = (AM.dot(I)
          + progression_rate * E
          - recovery_rate * I)
    dC = (AM.dot(C)
          + proportion_carriers * recovery_rate * I
          - chronic_recovery_rate * C)  
    dR = (AM.dot(R)
          + (1 - proportion_carriers) * recovery_rate * I
          + chronic_recovery_rate * C
          - immunity_waning_rate * R)

    dY = numpy.hstack((dS, dE, dI, dC, dR))

    return dY


def solve(tmax, agemax, agestep, parameters, Y0 = None):
    print("########## inside solve ########## ")
    matrices = utility.build_matrices(parameters,
                                      agemax = agemax,
                                      agestep = agestep)
    if not hasattr(parameters, 'population_size'):
        parameters.population_size = 1.

    (ages, (B_bar, A, M)) = matrices
    print("Birth, Aging, & Mortality matricies built of length, ",
        len(ages), B_bar.get_shape(), A.get_shape(), M.get_shape())
        
    AM = A - M

    birthRV = birth.gen(parameters)
    print("birth.gen finished")
    def B(t):
        Bval = sparse.lil_matrix((len(ages), ) * 2)
        Bval[0] = ((1 - parameters.male_probability_at_birth)
                   * birthRV.hazard(t, 0, ages - t))
        return Bval
    
    transmissibility = transmission_rate.gen(parameters)
    transmissibility_chronic = chronic_transmission_rate.gen(parameters)
    susceptibility = numpy.where(ages >= parameters.maternal_immunity_duration,
                                 1, 0)
    print("Transmissibility = ", transmissibility)
    print("Transmissibility for chronic = ", transmissibility_chronic)
    print("Susceptibility = ", susceptibility)
    print("Length susceptibility should equal ages, ", len(susceptibility))    
    
    def force_of_infection(I):
        return integrate.trapz(transmissibility * I, ages) * susceptibility

    def force_of_infection_carriers(C):
        return integrate.trapz(transmissibility * C, ages) * susceptibility

    progression_rate = 1 / parameters.progression_mean

    recovery_rate = 1 / parameters.recovery_mean

    chronic_recovery_rate = 1 / parameters.chronic_recovery

    immunity_waning_rate = 1 / parameters.immunity_waning

    proportion_carriers = parameters.probability_chronic
    print("Parameters = ", progression_rate, recovery_rate, 
        chronic_recovery_rate, immunity_waning_rate)

    if Y0 is None:
        eigenpair = utility.find_dominant_eigenpair(parameters,
                                                    _matrices = matrices)
        eigenvector = eigenpair[1][1]
        print("eigenpair found; eigenvector = ", len(eigenvector), eigenvector)
        N0 = (eigenvector / integrate.trapz(eigenvector, ages)
              * parameters.population_size)
        print("N0", len(N0), N0)
        # Everyone under 2 susceptible (except for maternal immunity).
        S0 = numpy.where(ages < 2, N0, 0)
        E0 = numpy.zeros_like(N0)
        # initial infections.
        I0 = 0.01 * S0
        C0 = numpy.zeros_like(N0)
        S0 -= I0
        R0 = N0 - S0 - I0 - C0

        Y0 = numpy.hstack((S0, E0, I0, C0, R0))

    t = numpy.linspace(0, tmax, 1001)

    (Y, info) = integrate.odeint(rhs, Y0, t,
                                 args = (AM, B,
                                         force_of_infection,
                                         force_of_infection_carriers,
                                         progression_rate,
                                         recovery_rate,
                                         proportion_carriers,
                                         immunity_waning_rate,
                                         chronic_recovery_rate),
                                 mxstep = 10000,
                                 full_output = True)
    print("Integrate ode worked!!")   
    if numpy.any(numpy.diff(info['tcur']) < -1e-16):
        raise RuntimeError('ODE solver failed!')

    (S, E, I, C, R) = numpy.hsplit(Y, 5)
    print("Split, integrated ode worked!!")
    # Split susceptibles into maternal immunity and not.
    mask = (ages < parameters.maternal_immunity_duration)
    M = numpy.where(mask[numpy.newaxis, :], S, 0)
    S -= M

    return (t, ages, (M, S, E, I, C, R))

def _get_endemic_equilibrium(parameters, tmax, agemax, agestep):
    print("######## inside _get_endemic_equilibrium,  populoation size set to 1 in solve ########")
    if parameters.start_time == 0:
        print("parameters.start_time == 0")
        (t, ages, Y) = solve(tmax, agemax, agestep, parameters)
        print("solve step ok!!")
        period = get_period(t, ages, Y)
        print("period ", period, " limit cycle")

        ICs = []
        for j in range(period):
            k = numpy.argwhere(t <= t[-1] - j)[-1, 0]
            ICs.append([y[k] for y in Y])

    else:
        print("parameters.start_time != 0")
        start_time = parameters.start_time
        parameters.start_time = 0
        (ages, ICs0) = _get_endemic_equilibrium(parameters, tmax,
                                                agemax, agestep)
        parameters.start_time = start_time

        ICs = []
        for IC0 in ICs0:
            (M0, S0, E0, I0, C0, R0) = IC0
            Y0 = numpy.hstack((M0 + S0, E0, I0, C0, R0))
            (t, ages, Y) = solve(parameters.start_time, agemax, agestep,
                                 parameters, Y0 = Y0)
            # Make sure it's non-negative.
            ICs.append([y[-1].clip(0, ) for y in Y])

    return (ages, ICs)

 
    
def get_endemic_equilibrium(parameters, tmax = 200,
                            agemax = 20, agestep = 0.01):
    # The agestep should be 0.01...
    # The PDE solutions simply scale multiplicatively with
    # population_size, so factor that out for more efficient caching.
    population_size = parameters.population_size
    del parameters.population_size
    (ages, ICs) = _get_endemic_equilibrium(parameters,
                                           tmax, agemax, agestep)
    parameters.population_size = population_size

    ICs_ = [[x * population_size for x in IC] for IC in ICs]
    
    return (ages, ICs_)



def get_period(t, ages, X, abserr = 1e-3, relerr = 1e-3,
               periodmax = 3):
    (M, S, E, I, C, R) = X
    Y = numpy.hstack((M + S, E, I, C, R))
    for period in range(1, periodmax + 1):
        i = numpy.argwhere(t <= t[-1] - period)[-1]
        if (numpy.linalg.norm(Y[i] - Y[-1])
            <= (abserr + relerr * numpy.linalg.norm(Y[-1]))):
            return period
    else:
        raise ValueError('period not found!')




 # solve is called in get_limit_cycle, and get_endemic equilibrium, below
 # both of thsoe are called in endemic_equilibrium.py (with class gen),
 # which is called in herd, self.rvs.endemic_equilibrium.rvs(self.params.population_size)
parameters = parameters.Parameters()
(ages, vals) = get_endemic_equilibrium(parameters)
len(vals)
print(vals)
(M, S, E, I, C, R) = numpy.hsplit(vals, 6)
print(C)


