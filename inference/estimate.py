#!/usr/bin/python3
'''Estimate the model parameters by SAT.'''

import numpy
import pandas

import _multiprocessing_
import model
import plot


def estimate_by_sat(pool=None, **kwds):
    '''Estimate the ML parameters and CI by SAT.'''
    data_ = model.load_data()
    grouper = data_.groupby('SAT', observed=False)
    log_lambda = {}
    with _multiprocessing_.Pool(pool) as pool_:
        for (sat, data_sat) in grouper:
            model_ = model.Model(data_sat)
            log_lambda[sat] = model_.estimate_ml_and_ci(pool=pool_, **kwds)
    return pandas.concat(log_lambda, names=['SAT'])


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


def to_parameters(log_rate_):
    '''Format the parameters like what is used in the manuscript.'''
    return (
        to_annual_rate(log_rate_)
        .round(2)
    )


if __name__ == '__main__':
    log_rate = estimate_by_sat(global_=False)
    print(to_parameters(log_rate))
    plot.rates_by_sat(log_rate)
