def transmissionRate_gen(parameters):
    return (parameters.R0
            / parameters.infectionDuration
            / parameters.populationSize)
