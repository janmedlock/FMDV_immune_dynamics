import numpy
from scipy import stats
from scipy import integrate

from . import rv


class gen(rv.RV):
    def __init__(self, parameters, *args, **kwargs):
        from . import pde

        (ages, ICs) = pde.get_endemic_equilibrium(parameters,
                                                  *args,
                                                  **kwargs)

        self.statuses = ('maternal immunity',
                         'susceptible',
                         'exposed',
                         'infectious',
                         'chronic',
                         'recovered')

        self.weights = numpy.array([[integrate.trapz(x, ages) for x in Y]
                                    for Y in ICs])
        self.weights /= self.weights.sum(axis = 1)[:, numpy.newaxis]

        self._rvs = [[rv.age_structured(ages, x / numpy.sum(x)) for x in Y]
                     for Y in ICs]

    def rvs(self, size = 1):
        if size is None:
            size = 1

        # Which of the ICs to use.
        j = numpy.random.randint(len(self._rvs))

        # N is an array of the number of each type of event that
        # occured.
        N = numpy.random.multinomial(size, self.weights[j])

        result = {status: RV.rvs(size = n)
                  for (status, RV, n) in zip(self.statuses, self._rvs[j], N)}

        return result
