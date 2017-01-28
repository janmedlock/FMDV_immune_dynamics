def gen(parameters):
    chronic_transmission_rate = (parameters.chronic_transmission_rate
                         / parameters.population_size)
    return chronic_transmission_rate
