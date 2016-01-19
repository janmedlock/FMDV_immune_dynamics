import numpy

from . import birth_super


class gen(birth_super.gen):
    def hazard(self, time, time0, age0):
        return numpy.where(
            age0 + time < 4,
            0,
            self.scaling * (1
                            + 2 * self.seasonal_coefficient_of_variation
                            * numpy.cos(2 * numpy.pi * (time + time0))))

    def _cdf(self, time, time0, age0):
        lb = numpy.max(numpy.hstack((0, 4 - age0)))
        I = numpy.where(
            time < 4 - age0,
            0,
            self.scaling \
            * ((time - lb)
               + self.seasonal_coefficient_of_variation / numpy.pi
               * (numpy.sin(2 * numpy.pi * (time + time0))
                  - numpy.sin(2 * numpy.pi * (lb + time0)))))
        
        return 1 - numpy.exp(- I)
