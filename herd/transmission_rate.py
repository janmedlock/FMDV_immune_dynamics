def gen(parameters):
    transmission_rate = (parameters.R0
                         / parameters.recovery_infection_duration
                         / parameters.population_size)
    return transmission_rate
