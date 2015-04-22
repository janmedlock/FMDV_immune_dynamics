def transmissionRate_gen(R0, recovery, populationSize):
    return R0 / recovery.mean() / populationSize
