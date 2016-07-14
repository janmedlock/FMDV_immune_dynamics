import numpy

# X_t = state at time t; v= vectors describing state changes; rates=a
def step(X_t, v, rates):
    a = rates(X_t)   # a rates for each state
    a0 = numpy.sum(a)   # sum of all rates

    # sample exponential waiting time for sum of rates
    dt = numpy.random.exponential(1. / a0)               # calculate tau; 1/a0=  mean

    # find which transition to execute
    i = numpy.searchsorted(numpy.cumsum(a) / a0, numpy.random.uniform())

    # execute rule i
    dX = v[i]

    return (dt, dX)
    
def run(X_t, v, rates, g, t_initial, t_end):
    '''
    Run the simulation from t_initial to t_end using the
    Gillespie algorithm.
    '''
    t = t_initial

    T = [t]
    X = [X_t.copy()]

    # loop through the times
    while t < t_end:
        (dt, dX) = step(X_t, v, rates)
        
        # see if time increment would take us past the end time
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

    return (numpy.array(T), numpy.array(X))
