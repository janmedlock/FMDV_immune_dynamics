class Parameters(object):
    def __init__(self, SAT = 1, _set_defaults = True):
        'Initialize with default values.'
        if _set_defaults:
            self.start_time = 0
            self.male_probability_at_birth = 0.5

            self.population_size = 1000

            self.birth_seasonal_coefficient_of_variation = 0.61

            self.maternal_immunity_duration = 0.5

            if SAT == 1:
                self.progression_shape = 1.2
                self.progression_mean = 1.0 / 365
                self.recovery_shape = 3.9
                self.recovery_mean = 6.0 / 365
                self.transmission_rate = 7.1 * 365
            elif SAT == 2:
                self.progression_shape = 1.7
                self.progression_mean = 1.9 / 365
                self.recovery_shape = 3.4
                self.recovery_mean = 4.8 / 365
                self.transmission_rate = 5.6 * 365
            elif SAT == 3:
                self.progression_shape = 1.6
                self.progression_mean = 3.3 / 365
                self.recovery_shape = 3.8
                self.recovery_mean = 4.6 / 365
                self.transmission_rate = 3.6 * 365
            else:
                raise ValueError("Unknown SAT '{}'!".format(SAT))

    def __repr__(self):
        'Make instances print nicely.'

        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        paramreprs = ['{!r}: {!r}'.format(k, self.__dict__[k])
                      for k in sorted(self.__dict__.keys())]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @classmethod
    def from_repr(cls, r_, set_defaults = True):
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

        p = cls(_set_defaults = set_defaults)
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
