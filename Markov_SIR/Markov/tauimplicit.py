import numpy
from scipy import optimize

from . import miscellany


def findtau(X_t, v, rates, g, r_critical, epsilon = 0.03):
    a = rates(X_t)

    y = (epsilon * X_t / g).clip(1., numpy.inf)

    # *** Need to also limit to non-partial-equilibrium reactions ***
    # Non-critical reactions
    nc = [j for j in range(len(a)) if j not in r_critical]
    a_nc = a[nc]
    v_nc = v[nc]
    mu = numpy.dot(a_nc, v_nc)
    sigma2 = numpy.dot(a_nc, v_nc ** 2)

    b = numpy.ma.divide(y, numpy.abs(mu))
    c = numpy.ma.divide(y ** 2, sigma2)

    tau = numpy.min((b, c))

    return tau

def step(X_t, v, rates, tau):
    '''
    X(t + \tau) = X(t)
                  + \sum_j v_j * [Poisson(a_j(X(t)) \tau)
                                  - \tau a_j(X(t)) / 2
                                  + \tau a_j(X(t + \tau)) / 2]
    '''
    a_t = rates(X_t)

    # The number of transitions are Possion.
    P = numpy.random.poisson(tau * a_t)

    def getN(X_tau):
        a_tau = rates(X_tau)
        N = P - tau * a_t / 2. + tau * a_tau / 2.
        return N

    def f(X_tau):
        N = getN(X_tau)
        return (X_tau - X_t - numpy.dot(N, v))
        
    # X_tau_opt = optimize.fsolve(f, X_t)
    sol = optimize.root(f, X_t, method = 'lm')
    X_tau_opt = sol.x

    N = getN(X_tau_opt)
    dX = numpy.dot(numpy.around(N), v)

    return (tau, dX)

def run(X_t, v, rates, g, t_initial, t_end,
        n_c = 10, epsilon = 0.03):
    '''
    Run the simulation from t_initial to t_end using the
    implicit tau-leaping algorithm:
    X(t + \tau) = X(t)
                  + \sum_j v_j * [Poisson(a_j(X(t)) \tau)
                                  - \tau a_j(X(t)) / 2
                                  + \tau a_j(X(t + \tau)) / 2]
    '''
    t = t_initial

    T = [t]
    X = [X_t.copy()]

    # loop through the times
    while t < t_end:
        r_critical = miscellany.findcriticalreax(X_t, v, rates, n_c)
        tau = findtau(X_t, v, rates, g, r_critical, epsilon)

        if (t + tau > t_end):
            tau = t_end - t

        while True:
            (dt, dX) = step(X_t, v, rates, tau)

            if numpy.all(X_t + dX >= 0):
                t += dt
                X_t += dX
                break
            else:
                tau /= 2.

        T.append(t)
        X.append(X_t.copy())

    return (numpy.array(T), numpy.array(X))
