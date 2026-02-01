#!/usr/bin/python3
'''Estimate the model parameters by SAT.'''

import functools

import numpy
import pandas

from context import data as data_
import model
import plotting_
import _multiprocessing_


@functools.lru_cache(maxsize=1)
def load_data():
    '''Load the `data` and rearrange it to the form used by `Model`.
    The resuling `pandas.DataFrame()` has columns:
    1. SAT,
    2. animal,
    3. number of current capture,
    4. t_0: time of previous capture,
    4. t: time of current capture,
    5. x_0: antibody state at previous capture,
    6. x: antibody state at current capture.'''

    def get_state(data):
        '''Build 'state' column.'''
        return (
            data.loc[:, 'positive']
            .map({True: model.STATE['positive'],
                  False: model.STATE['negative']})
            .astype(model.STATE.dtype)
            .rename('state')
        )

    def days_since(ser, date_start):
        '''Days since `date_start`.'''
        return (
            (ser - date_start)
            / pandas.offsets.Day()
        )

    def convert_group(group, date_start):
        time = days_since(group.date, date_start)
        state = get_state(group)
        return (
            pandas.DataFrame({
                't_0': time.shift(1),
                'x_0': state.shift(1),
                't':   time,
                'x':   state,
            })
            # Drop the first row, which has no previous observation.
            .iloc[1:]
        )

    data = data_.load()
    grouper = (
        data.set_index('capture')
        .groupby(['SAT', 'ID'], observed=False)
    )
    date_start = data.date.min()
    return grouper.apply(convert_group, date_start,
                         include_groups=False)


@functools.lru_cache(maxsize=1)
def estimate_by_sat(pool=None, **kwds):
    '''Estimate the ML parameters and CI by SAT.'''
    model_data = load_data()
    grouper = model_data.groupby('SAT', observed=False)
    log_lambda = {}
    with _multiprocessing_.Pool(pool) as pool_:
        for (sat, model_data_sat) in grouper:
            model_ = model.Model(model_data_sat)
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
        .unstack('SAT')
        .iloc[::-1]
        .stack('SAT', future_stack=True)
    )


if __name__ == '__main__':
    log_rate = estimate_by_sat(global_=False)
    print(to_parameters(log_rate))
    # plotting_.rates_by_sat(log_rate)
