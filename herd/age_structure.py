from . import birth
from . import rv
from . import utility


@utility.shelved('birth_seasonal_coefficient_of_variation',
                 'male_probability_at_birth',
                 'start_time')
def find_stable_age_structure(parameters, *args, **kwargs):
    ages, matrices = utility.build_ages_and_matrices(parameters, *args, **kwargs)
    birth_scaling = birth.find_birth_scaling(parameters, _matrices=matrices)
    _, v = utility.find_dominant_eigenpair(birth_scaling, *matrices)
    return (ages, v)


class gen(rv.age_structured):
    def __init__(self, parameters, *args, **kwargs):
        ages, proportion = find_stable_age_structure(parameters, *args, **kwargs)
        super().__init__(ages, proportion, *args, **kwargs)
