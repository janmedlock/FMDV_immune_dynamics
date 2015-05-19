import numpy
from scipy import stats
from scipy import integrate

from . import rv


class endemicEquilibrium_gen(rv.RV):
    def __init__(self, parameters, *args, **kwargs):
        from . import pde

        (ages, ICs) = pde.getEndemicEquilibrium(parameters,
                                                *args,
                                                **kwargs)

        self.statuses = ('maternal immunity',
                         'susceptible',
                         'infectious',
                         'recovered')

        self.weights = numpy.array([[integrate.trapz(x, ages) for x in Y]
                                    for Y in ICs])
        self.weights /= self.weights.sum(axis = 1)[:, numpy.newaxis]

        self.RVs = [[rv.ageStructure_gen(ages, x / numpy.sum(x)) for x in Y]
                    for Y in ICs]

    def rvs(self, size = 1):
        if size is None:
            size = 1

        # Which of the ICs to use.
        j = numpy.random.randint(len(self.RVs))

        # N is an array of the number of each type of event that
        # occured.
        N = numpy.random.multinomial(size, self.weights[j])

        result = {status: RV.rvs(size = n)
                  for (status, RV, n) in zip(self.statuses, self.RVs[j], N)}

        return result
