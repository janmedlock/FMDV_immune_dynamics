def gen(parameters):
    '''Scales with population size so that R_0 is constant as population
    size changes.'''
    transmission_rate = (parameters.transmission_rate
                         / parameters.population_size)
    return transmission_rate
