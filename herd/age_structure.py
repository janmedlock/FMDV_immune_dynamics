import numpy

from . import rv
from . import eigen


class gen(rv.age_structured):
    def __init__(self, parameters, *args, **kwargs):
        ages, proportion = eigen.find_stable_age_structure(parameters,
                                                           *args, **kwargs)
        super().__init__(ages, proportion, *args, **kwargs)
