import copy

from pandas import Timestamp


class Parameters:
    def __init__(self, SAT=1, _set_defaults=True, **kwds):
        'Initialize with default values.'
        self.SAT = SAT
        if _set_defaults:
            self.population_size = 1000
            self.female_probability_at_birth = 0.5
            self.birth_peak_time_of_year = 0  # January 01.
            # self.birth_seasonal_coefficient_of_variation = 0.505  # 1st year.
            # self.birth_seasonal_coefficient_of_variation = 0.984  # 2nd year.
            self.birth_seasonal_coefficient_of_variation = 0.613  # Both years.
            # During in M before moving to S.
            self.maternal_immunity_duration_mean = 0.37
            self.maternal_immunity_duration_shape = 1.19
            # For the rate, not duration, leaving R to P.
            # The first day is 2014 March 05.
            date_min = Timestamp(year=2014, month=3, day=5)
            # Convert to years, with January 1 at 0.
            self.start_time = (date_min.dayofyear - 1) / 365
            self.lost_immunity_susceptibility = 1
            if self.SAT == 1:
                self.progression_shape = 1.2
                self.progression_mean = 0.5 / 365
                self.recovery_shape = 11.8
                self.recovery_mean = 5.7 / 365
                self.transmission_rate = 2.8 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.90
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 243 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.028 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.0013358 * 365
                # Rate, not duration, leaving L to R.
                # TODO: Update value.
                self.antibody_gain_hazard = 0.011411 * 365
            elif self.SAT == 2:
                self.progression_shape = 1.6
                self.progression_mean = 1.3 / 365
                self.recovery_shape = 8.7
                self.recovery_mean = 4.6 / 365
                self.transmission_rate = 1.6 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.44
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 180 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.003 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.0031631 * 365
                # Rate, not duration, leaving L to R.
                # TODO: Update value.
                self.antibody_gain_hazard = 0.0044599 * 365
            elif self.SAT == 3:
                self.progression_shape = 1.6
                self.progression_mean = 2.8 / 365
                self.recovery_shape = 11.8
                self.recovery_mean = 4.2 / 365
                self.transmission_rate = 1.2 * 365
                # Proportion leaving I that become C.
                self.probability_chronic = 0.67
                # Duration in C before leaving to R.
                self.chronic_recovery_mean = 174 / 365
                self.chronic_recovery_shape = 3.2
                self.chronic_transmission_rate = 0.012 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.0053853 * 365
                # Rate, not duration, leaving L to R.
                # TODO: Update value.
                self.antibody_gain_hazard = 0.0058625 * 365
            else:
                raise ValueError(f'Unknown {SAT=}!')
        self.set(**kwds)

    def set(self, **kwds):
        for (key, val) in kwds.items():
            setattr(self, key, val)

    def __repr__(self):
        'Make instances print nicely.'
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramreprs = ['{!r}: {!r}'.format(k, self.__dict__[k])
                      for k in sorted(self.__dict__.keys())]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @classmethod
    def from_repr(cls, r_, set_defaults=True):
        assert r_.startswith('<')
        assert r_.endswith('>')
        r = r_[1 : -1]
        l = r.find(':')
        name = r[ : l]
        # Can be different depending on how the module is imported.
        clsname = '{}.{}'.format(cls.__module__, cls.__name__)
        assert (name == clsname)
        paramstr_ = r[l + 2 : ]
        assert paramstr_.startswith('{')
        assert paramstr_.endswith('}')
        paramstr = paramstr_[1 : -1]
        p = cls(_set_defaults=set_defaults)
        for s in paramstr.split(', '):
            (k_, vstr) = s.split(': ')
            assert k_.startswith("'")
            assert k_.endswith("'")
            k = k_[1 : -1]
            v = float(vstr)
            if int(v) == v:
                v = int(v)
            setattr(p, k, v)
        return p

    def copy(self):
        return copy.copy(self)

    def merge(self, **kwds):
        new = self.copy()
        new.set(**kwds)
        return new
