#!/usr/bin/python3
'''Estimate the model parameters by SAT.'''

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


if __name__ == '__main__':
    parameters = estimate_by_sat(global_=False)
    plot.waiting_times_by_sat(parameters)
