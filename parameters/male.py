from scipy import stats


def male_gen(probabilityOfMaleBirth):
    return stats.bernoulli(probabilityOfMaleBirth)
