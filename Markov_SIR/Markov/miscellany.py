import numpy


def findcriticalreax(X_t, v, rates, n_c = 10):
    a = rates(X_t)

    r_critical = []
    for j in range(len(a)):
        if a[j] > 0:
            Lj = numpy.floor(numpy.ma.divide(- X_t, v[j]).compress(v[j] < 0))
            if len(Lj) > 0:
                Lj_ = numpy.min(Lj)
                if Lj_ < n_c:
                    r_critical.append(j)

    return r_critical


def getmean(Y):
    (T, X) = zip(*Y)
    t_mean = numpy.unique(numpy.hstack(T))
    X_mean = numpy.zeros((len(t_mean), len(X[0][0])))
    n = numpy.zeros_like(t_mean)
    for (Tk, Xk) in zip(T, X):
        Tk = numpy.array(Tk)
        Xk = numpy.array(Xk)

        # Only go to the end of this simulation.
        T_ = t_mean.compress(t_mean <= Tk[-1])

        # Find the indicies i[j] of the largest Tk with Tk[i[j]] <= T_[j]
        indices = [(Tk <= t).nonzero()[0][-1] for t in T_]

        X_mean[ : len(T_)] += Xk[indices]
        n[ : len(T_)] += 1.
    X_mean /= n[:, numpy.newaxis]

    return (t_mean, X_mean)
