#!/usr/bin/python3
#
# Assumptions:
# * Constant population size, particulraly no seasonal births.
# * All susceptible, so drop M.
# * Mortality is negligible, so S->E->I is equivalent to S->I,
#   among other simplifications.
# * Transmission rate for both acute and chronic scales so that R_0 is
#   independent of population size.
#
# To do:
# * Update to use samples for chronic model.

import herd


def calculate_samples(SAT, chronic, alpha=0.05):
    assert chronic == False
    from herd.samples import samples
    s = samples[SAT]
    R0 = s.recovery_mean * s.transmission_rate
    R0 = R0.quantile([0.5, alpha / 2, 1 - alpha / 2])
    return '{:.1f}\t({:.1f}, {:.1f})'.format(*R0)


def calculate_mode(SAT, chronic):
    p = herd.Parameters(SAT=SAT, chronic=chronic)
    R0 = (p.recovery_mean * p.transmission_rate
          + p.probability_chronic * p.chronic_recovery_mean)
    return '{:.1f}'.format(R0)


def calculate(SAT, chronic):
    if chronic:
        return calculate_mode(SAT, chronic)
    else:
        return calculate_samples(SAT, chronic)


if __name__ == '__main__':
    print('Chronic\tSAT\tR_0')
    for chronic in (False, True):
        for SAT in (1, 2, 3):
            print('{}\t{}\t{}'.format(chronic, SAT, calculate(SAT, chronic)))
