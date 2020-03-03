#!/usr/bin/python3
#
# Assumptions:
# * Constant population size, particulraly no seasonal births.
# * All susceptible, so drop M.
# * Mortality is negligible, so S->E->I is equivalent to S->I,
#   among other simplifications.
# * Transmission rates for both acute and chronic scales so that R_0 is
#   independent of population size.

import herd.samples


def calculate(SAT, alpha=0.05):
    s = herd.samples.load(SAT=SAT)
    R0 = (s.recovery_mean * s.transmission_rate
          + s.probability_chronic * s.chronic_recovery_mean)
    R0 = R0.quantile([0.5, alpha / 2, 1 - alpha / 2])
    return R0


if __name__ == '__main__':
    print('SAT\tR_0')
    for SAT in (1, 2, 3):
        print('{}\t{:.1f}\t({:.1f}, {:.1f})'.format(
            SAT, *calculate(SAT)))
