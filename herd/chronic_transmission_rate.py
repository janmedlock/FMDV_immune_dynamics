def gen(parameters):
    '''Scales with population size so that R_0 is constant as population
    size changes.'''
    chronic_transmission_rate = (parameters.chronic_transmission_rate
                                 / parameters.population_size)
    return chronic_transmission_rate
