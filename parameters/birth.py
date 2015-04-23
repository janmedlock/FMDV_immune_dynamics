import numpy
from scipy import stats

from . import rv
from . import utility


class birth_gen(rv.RV, stats.rv_continuous):
    def __init__(self,
                 mortality, male,
                 seasonalAmplitude = 1.,
                 *args, **kwargs):
        self.seasonalAmplitude = seasonalAmplitude

        self.mortality = mortality
        self.male = male
        self.findBirthScaling(mortality, male)

        stats.rv_continuous.__init__(self,
                                     name = 'birth',
                                     a = 0.,
                                     shapes = 'time0, age0',
                                     *args, **kwargs)

    def _argcheck(self, time0, age0):
        return (age0 >= 0.)
        
    def hazard(self, time, time0, age0):
        return numpy.where(
            age0 + time < 4.,
            0.,
            self.scaling * (1.
                            + self.seasonalAmplitude
                            * numpy.cos(2 * numpy.pi * (time + time0))))

    # def _cdf_single(self, time, time0, age0):
    #     result = scipy.integrate.quad(self.hazard, 0, time,
    #                                   args = (time0, age0),
    #                                   limit = 100, full_output = 1)
    #     I = result[0]
    #     return 1. - numpy.exp(- I)

    # def _cdf(self, time, time0, age0):
    #     return numpy.vectorize(self._cdf_single)(time, time0, age0)

    def _cdf(self, time, time0, age0):
        lb = numpy.max(numpy.hstack((0., 4 - age0)))
        I = numpy.where(
            time < 4 - age0,
            0.,
            self.scaling \
            * ((time - lb)
               + self.seasonalAmplitude / 2. / numpy.pi
               * (numpy.sin(2. * numpy.pi * (time + time0))
                  - numpy.sin(2. * numpy.pi * (lb + time0)))))
        
        return 1. - numpy.exp(- I)

    def _ppf(self, q, *args, **kwds):
        'Trap errors for _ppf'
        try:
            result = super(birth_gen, self)._ppf(q, *args, **kwds)
        except ValueError:
            # Assume the error is near q = 1,
            # so return the right-hand endpoint
            # of the support of the distribution
            # (which is +inf by default).
            result = self.b
        return result

    def findBirthScaling(self, mortality, male,
                         *args, **kwargs):
        self.scaling = utility.findBirthScaling(mortality, self, male,
                                                *args, **kwargs)

    def __repr__(self):
        return rv.RV.__repr__(self, ('seasonalAmplitude', ))
