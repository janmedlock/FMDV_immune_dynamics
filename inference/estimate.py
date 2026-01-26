#!/usr/bin/python3
'''Estimate the model parameters by SAT.'''

import numpy
import pandas

import model
import plot


def estimate_by_sat(**kwds):
    '''Estimate the ML parameters and CI by SAT.'''
    data_ = model.load_data()
    grouper = data_.groupby('SAT', observed=False)
    theta = {}
    for (sat, data_sat) in grouper:
        model_ = model.Model(data_sat)
        theta[sat] = model_.estimate_ml_and_ci(**kwds)
    return pandas.concat(theta, names=['SAT'])


def to_annual_rate(log_rate_):
    '''Convert log daily rate to annual rate.'''
    return (
        (
            log_rate_.apply(numpy.exp)
            * 365
        )
        .rename(lambda x: x.replace('log rate', 'annual rate'),
                level='parameter')
    )


if __name__ == '__main__':
    log_rate = estimate_by_sat(global_=False)
    print(
        to_annual_rate(log_rate)
        .round(2)
    )
    plot.waiting_times_by_sat(log_rate)
