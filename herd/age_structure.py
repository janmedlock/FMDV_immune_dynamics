import numpy

from . import rv
from .eigen import find_stable_age_structure


class gen(rv.age_structured):
    def __init__(self, parameters, *args, **kwargs):
        ages, proportion = find_stable_age_structure(parameters,
                                                     *args, **kwargs)
        super().__init__(ages, proportion, *args, **kwargs)
