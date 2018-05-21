import numpy

from . import rv


class gen(rv.age_structured):
    def __init__(self, parameters, *args, **kwargs):
        from . import utility
        (ages, proportion) = utility.find_stable_age_structure(
            parameters, *args, **kwargs)
        super().__init__(ages, proportion, *args, **kwargs)
