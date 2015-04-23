from . import rv


class recovery_gen(rv.deterministic):
    'Transition at infectionDuration with probability 1.'

    def __init__(self, infectionDuration, *args, **kwargs):
        super(recovery_gen, self).__init__('infectionDuration',
                                           infectionDuration,
                                           *args, **kwargs)
