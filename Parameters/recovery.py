from . import rv


class recovery_gen(rv.deterministic):
    'Transition at infectionDuration with probability 1.'

    def __init__(self, parameters, *args, **kwargs):
        super(recovery_gen, self).__init__('infectionDuration',
                                           parameters.infectionDuration,
                                           *args, **kwargs)
