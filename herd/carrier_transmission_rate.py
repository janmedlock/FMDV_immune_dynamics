def gen(parameters, seed=None):
    '''Scales with population size so that R_0 is constant as population
    size changes.'''
    carrier_transmission_rate = (parameters.carrier_transmission_rate
                                 / parameters.population_size)
    return carrier_transmission_rate
