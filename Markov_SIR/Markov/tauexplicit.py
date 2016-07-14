import numpy

from . import miscellany


def findtau(X_t, v, rates, g, r_critical, epsilon = 0.03):
    a = rates(X_t)

    y = (epsilon * X_t / g).clip(1., numpy.inf)

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
    a = rates(X_t)

    # The number of transitions are Possion.
    N = numpy.random.poisson(tau * a)

    # Do N of each transition.
    dX = numpy.dot(N, v)

    return (tau, dX)

def run(X_t, v, rates, g, t_initial, t_end,
        n_c = 10, epsilon = 0.03):
    '''
    Run the simulation from t_initial to t_end using the
    explicit tau-leaping algorithm.
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
