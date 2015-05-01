from . import rv


class maternalImmunityWaning_gen(rv.deterministic):
    'Transition at maternalImmunityDuration with probability 1.'

    def __init__(self, parameters, *args, **kwargs):
        super(maternalImmunityWaning_gen, self).__init__(
            'maternalImmunityDuration',
            parameters.maternalImmunityDuration,
            *args, **kwargs)
