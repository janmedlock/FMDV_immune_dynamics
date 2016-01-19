from . import rv


class gen(rv.deterministic):
    'Transition at maternal_immunity_duration with probability 1.'

    def __init__(self, parameters, *args, **kwargs):
        super().__init__('maternal_immunity_duration',
                         parameters.maternal_immunity_duration,
                         *args, **kwargs)
