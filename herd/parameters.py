class Parameters(object):
    def __init__(self, _set_defaults = True):
        'Initialize with default values.'
        if _set_defaults:
            self.R0 = 4   # initailly set to 10, Ro = beta * N/ (gamma + mortality)
            self.birth_seasonal_coefficient_of_variation = 0.61  # initially set at 1
            self.male_probability_at_birth = 0.5
            self.maternal_immunity_duration = 0.5
            self.population_size = 500
            self.recovery_infection_duration = 4.35/ 365 # 4.9 / 365   # initially set at 21/365
            self.start_time = 0

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
