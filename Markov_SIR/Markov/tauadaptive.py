import numpy
from scipy import optimize

from . import miscellany
from . import SSA
from . import tauexplicit
from . import tauimplicit


def run(X_t, v, rates, g, t_initial, t_end,
        n_c = 10, epsilon = 0.03, N_stiff = 100):
    '''
    Follows

    Y Cao, DT Gillespie, LR Petzold, 2007, Adaptive explicit-implicit
    tau-leaping method with automatic tau selection, J Chem Phys 126:
    224101.
    '''
    t = t_initial

    T = [t]
    X = [X_t.copy()]

    method_last = None
    # loop through the times
    while t < t_end:
        r_critical = miscellany.findcriticalreax(X_t, v, rates, n_c)

        tau_exp = tauexplicit.findtau(X_t, v, rates, g,
                                      r_critical, epsilon)
        tau_imp = tauimplicit.findtau(X_t, v, rates, g,
                                      r_critical, epsilon)

        if tau_imp > N_stiff * tau_exp:
            method = tauimplicit.step
            tau_1 = tau_imp
        else:
            method = tauexplicit.step
            tau_1 = tau_exp

        while True:
            a = rates(X_t)

            if tau_1 < 10 * numpy.sum(a):
                method = SSA.step
                if method_last == tauimplicit.step:
                    nsteps = 10
                else:
                    nsteps = 100

                for i in xrange(nsteps):
                    (dt, dX) = SSA.step(X_t, v, rates)

                    if (t + dt < t_end):
                        # update time
                        t += dt

                        # execute transition rule
                        X_t += dX
                    else:
                        # set time to end time
                        t = t_end

                    T.append(t)
                    X.append(X_t.copy())

                    if t == t_end:
                        # Stop SSA loop.
                        break
                # Stop while loop.
                break

            else:
                a0_c = numpy.sum(a[r_critical])
                if a0_c == 0:
                    tau_2 = numpy.inf
                else:
                    tau_2 = numpy.random.exponential(1. / a0_c)

                # Non-critical reactions
                nc = [j for j in range(len(a)) if j not in r_critical]
                v_nc = v[nc]
                def rates_nc(X_t):
                    return rates(X_t)[nc]

                if tau_2 > tau_1:
                    tau = tau_1

                    (dt, dX) = method(X_t, v_nc, rates_nc, tau)
                else:
                    tau = tau2

                    a_c = a[r_critical]
                    a0_c = numpy.sum(a_c)

                    # Find which critical reaction to execute
                    j_c = numpy.searchsorted(numpy.cumsum(a_c) / a0_c,
                                             numpy.random.uniform())

                    dX_c = v[j_c]

                    if (method == tauimplicit.step and tau_2 <= tau_exp):
                        method = tauexplicit.step

                    (dt, dX_nc) = method(X_t, v_nc, rates_nc, tau)

                    dX = dX_c + dX_nc

                if numpy.all(X_t + dX >= 0):
                    t += dt
                    X_t += dX
                    break
                else:
                    tau_1 /= 2.

        if method != SSA.step:
            T.append(t)
            X.append(X_t.copy())

        method_last = method

    return (numpy.array(T), numpy.array(X))
