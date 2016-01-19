class Parameters(object):
    def __init__(self):
        'Initialize with default values.'
        self.R0 = 10
        self.birth_seasonal_coefficient_of_variation = 1
        self.male_probability_at_birth = 0.5
        self.maternal_immunity_duration = 0.5
        self.population_size = 1000
        self.recovery_infection_duration = 21 / 365

    def __repr__(self):
        'Make instances print nicely.'

        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        paramreprs = ['{!r}: {!r}'.format(k, self.__dict__[k])
                      for k in sorted(self.__dict__.keys())]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))
