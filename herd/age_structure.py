from . import birth
from . import rv
from . import utility


@utility.shelved('birth_seasonal_coefficient_of_variation',
                 'male_probability_at_birth',
                 'start_time')
def _find_stable(parameters, *args, **kwargs):
    '''Find the stable age structure.'''
    ages, matrices = utility.build_ages_and_matrices(parameters, *args, **kwargs)
    scaling = birth._find_scaling(parameters, _matrices=matrices)
    _, v = utility.find_dominant_eigenpair(scaling, *matrices)
    return (ages, v)


class gen(rv.age_structured):
    def __init__(self, parameters, *args, **kwargs):
        ages, proportion = _find_stable(parameters, *args, **kwargs)
        super().__init__(ages, proportion, *args, **kwargs)
