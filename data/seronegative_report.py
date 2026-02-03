#!/usr/bin/python3
'''Report the proportion seronegative by SAT.'''

import data_


def report_seronegative(data_with_age):
    '''Report the proportion seronegative among those aged ≥ 2 years.'''
    keep = data_with_age['age (y)'] >= 2
    negative = (
        data_with_age.loc[keep]
        .groupby('SAT')
        ['negative']
    )
    proportion = (
        (negative.sum() / negative.size())
        .rename('proportion seronegative')
    )
    print(proportion.round(3))


if __name__ == '__main__':
    data_with_age_ = data_.load_with_age()
    report_seronegative(data_with_age_)
