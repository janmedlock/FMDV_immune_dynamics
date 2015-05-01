import numpy

from . import rv


class ageStructure_gen(rv.ageStructure_gen):
    def __init__(self, parameters, *args, **kwargs):
        from . import utility

        (ages, proportion) = utility.findStableAgeStructure(
            parameters, *args, **kwargs)

        super(ageStructure_gen, self).__init__(ages, proportion,
                                               *args, **kwargs)
