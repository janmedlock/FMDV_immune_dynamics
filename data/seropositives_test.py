#!/usr/bin/python3
'''Compare seropositive titers by SAT.'''

import scipy
import statsmodels.api
import statsmodels.stats.multitest

import data_


MW_ALTERNATIVE = {
    (1, 2): 'greater',
    (1, 3): 'greater',
    (2, 3): 'greater',
}


def describe(titers):
    print(titers.describe()
          .drop(columns=['mean', 'std', 'min', 'max'])
          .astype({'count': int})
          .round(1))


def correct_p(results, method='bonferroni'):
    p = [res.pvalue for res in results.values()]
    (_, p_corrected, _, _) = statsmodels.stats.multitest.multipletests(
        p, method=method,
    )
    return dict(zip(results.keys(), p_corrected))


def mannwhitneyu(titers):
    results = {}
    for (sat_i, titer_i) in titers:
        for (sat_j, titer_j) in titers:
            if sat_i < sat_j:
                alternative = MW_ALTERNATIVE[(sat_i, sat_j)]
                results[(sat_i, sat_j)] = scipy.stats.mannwhitneyu(
                    titer_i, titer_j,
                    alternative=alternative,
                )
    p_corrected = correct_p(results)
    for ((sat_i, sat_j), res) in results.items():
        print(
            f'Mann-Whitney ({alternative}) SAT{sat_i} vs. SAT{sat_j}: '
            f'U={res.statistic}, '
            f'p={p_corrected[(sat_i, sat_j)]}',
        )


def test_titers(seropositives):
    '''Test the titers.'''
    titers = seropositives.groupby('SAT')['titer']
    describe(titers)
    mannwhitneyu(titers)


if __name__ == '__main__':
    seropositives_ = data_.load_seropositives()
    test_titers(seropositives_)
