def gen(parameters):
    return (parameters.R0
            / parameters.recovery_infection_duration
            / parameters.population_size)
