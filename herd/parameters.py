import copy
import warnings

import pandas

from . import antibodies

START_DATE = pandas.Timestamp('2014-03-05')


class Parameters:
    def __init__(self, SAT=1, sample_antibody_precisions=False, seed=None,
                 _set_defaults=True, **kwds):
        'Initialize with default values.'
        self.SAT = SAT
        # Whether to set `.lost_immunity_true_probability` and
        # `.recovered_true_probability` using bootstrap sampling
        self.sample_antibody_precisions = sample_antibody_precisions
        if _set_defaults:
            self.population_size = 1000
            self.female_probability = 0.5
            # January 1 on or after `START_DATE`
            birth_peak_date = (
                pandas.offsets.YearBegin().rollforward(START_DATE)
                + pandas.DateOffset(month=1, day=1)
            )
            # The duration of the year starting on `START_DATE`
            year = START_DATE + pandas.DateOffset(years=1) - START_DATE
            self.birth_peak_time = (
                (birth_peak_date - START_DATE) / year
            )  # Units: year
            # self.birth_seasonal_coefficient_of_variation = 0.505  # 1st year.
            # self.birth_seasonal_coefficient_of_variation = 0.984  # 2nd year.
            self.birth_seasonal_coefficient_of_variation = 0.613  # Both years.
            self.birth_menarche = 4  # years
            self.mortality_age_dependent = True
            # Duration in M before moving to S.
            self.maternal_immunity_duration_mean = 0.37
            self.maternal_immunity_duration_shape = 1.19
            self.lost_immunity_susceptibility = 1
            if self.SAT == 1:
                self.progression_mean = 0.5 / 365
                self.progression_shape = 1.2
                self.recovery_mean = 5.7 / 365
                self.recovery_shape = 11.8
                self.transmission_rate = 2.8 * 365
                # Proportion leaving I that become C.
                self.recovery_to_carrier_probability = 0.90
                # Duration in C before leaving to R.
                self.carrier_recovery_mean = 243 / 365
                self.carrier_recovery_shape = 3.2
                self.carrier_transmission_rate = 0.028 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.001440770075415 * 365
                # Rate, not duration, leaving L to R.
                self.antibody_gain_hazard = 0.007350989181831 * 365
            elif self.SAT == 2:
                self.progression_mean = 1.3 / 365
                self.progression_shape = 1.6
                self.recovery_mean = 4.6 / 365
                self.recovery_shape = 8.7
                self.transmission_rate = 1.6 * 365
                # Proportion leaving I that become C.
                self.recovery_to_carrier_probability = 0.44
                # Duration in C before leaving to R.
                self.carrier_recovery_mean = 180 / 365
                self.carrier_recovery_shape = 3.2
                self.carrier_transmission_rate = 0.003 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.003170648923222 * 365
                # Rate, not duration, leaving L to R.
                self.antibody_gain_hazard = 0.005030729273316 * 365
            elif self.SAT == 3:
                self.progression_mean = 2.8 / 365
                self.progression_shape = 1.6
                self.recovery_mean = 4.2 / 365
                self.recovery_shape = 11.8
                self.transmission_rate = 1.2 * 365
                # Proportion leaving I that become C.
                self.recovery_to_carrier_probability = 0.67
                # Duration in C before leaving to R.
                self.carrier_recovery_mean = 174 / 365
                self.carrier_recovery_shape = 3.2
                self.carrier_transmission_rate = 0.012 * 365
                # Rate, not duration, leaving R to L.
                self.antibody_loss_hazard = 0.005460735526547 * 365
                # Rate, not duration, leaving L to R.
                self.antibody_gain_hazard = 0.004176835549525 * 365
            else:
                raise ValueError(f'Unknown {SAT=}!')
            antibody_precisions = self._get_antibody_precisions(seed)
            self.recovered_true_probability = float(
                antibody_precisions['recovered'])
            self.lost_immunity_true_probability = float(
                antibody_precisions['lost_immunity'])
        self.set(**kwds)

    def _get_antibody_precisions(self, seed):
        if self.sample_antibody_precisions:
            return antibodies.precision.sample(self.SAT, seed)
        return antibodies.precision.estimate(self.SAT)

    def set(self, **kwds):
        for (key, val) in kwds.items():
            setattr(self, key, val)

    def __repr__(self):
        'Make instances print nicely.'
        clsname = f'{self.__module__}.{self.__class__.__name__}'
        params = vars(self)
        paramreprs = ', '.join(
            f'{k!r}: {params[k]!r}'
            for k in sorted(params)
        )
        return f'<{clsname}: {{{paramreprs}}}>'

    def _warn_if_sample_antibody_precisions(self):
        if self.sample_antibody_precisions:
            warnings.warn('Antibody precisions not resampled!')

    @classmethod
    def from_repr(cls, r_, set_defaults=True):
        assert r_.startswith('<')
        assert r_.endswith('>')
        r = r_[1 : -1]
        l = r.find(':')
        name = r[ : l]
        # Can be different depending on how the module is imported.
        clsname = f'{cls.__module__}.{cls.__name__}'
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
        p._warn_if_sample_antibody_precisions()
        return p

    def copy(self):
        new = copy.copy(self)
        new._warn_if_sample_antibody_precisions()
        return new

    def merge(self, **kwds):
        new = self.copy()
        new.set(**kwds)
        return new
