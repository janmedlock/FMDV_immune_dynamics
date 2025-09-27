'''Common code, much of it for plotting.'''

import os

import numpy
import pandas
import psutil

import herd.utility


SATs = (1, 2, 3)

NRUNS = 1000

TIME_UNIT = 'year'
TIME_MAX = 10

cols_infected = ['exposed', 'infectious', 'chronic']


def nice_self():
    '''Set to minimum CPU and IO prioirities.'''
    pid = os.getpid()
    proc = psutil.Process(pid)
    proc.nice(19)
    proc.ionice(psutil.IOPRIO_CLASS_BE, 7)


def insert_index_levels(dfr, i, **levels):
    dfr.index = pandas.MultiIndex.from_arrays(
        [dfr.index.get_level_values(n) for n in dfr.index.names[:i]]
        + [pandas.Index([v], name=k).repeat(len(dfr))
           for (k, v) in levels.items()]
        + [dfr.index.get_level_values(n) for n in dfr.index.names[i:]])


def append_index_levels(dfr, **levels):
    insert_index_levels(dfr, dfr.index.nlevels, **levels)


def prepend_index_levels(dfr, **levels):
    insert_index_levels(dfr, 0, **levels)


def get_logging_prefix(**kwds):
    return ', '.join(f'{key}={val}'
                     for (key, val) in kwds.items())


def get_infected(obj):
    '''Get the number of infected at each time.'''
    # Sum over columns for a `pandas.DataFrame()`,
    # over index for a `pandas.Series()`.
    axis = obj.ndim - 1
    infected = (
        obj[cols_infected]
        .sum(axis=axis)
    )
    try:
        infected.rename('infected', inplace=True)
    except AttributeError:
        pass  # `obj` was a `pandas.Series()`.
    return infected


def get_daily(dfr):
    '''Resample `dfr` daily.'''
    assert TIME_UNIT == 'year'
    t_step = 1 / 365
    t_daily = herd.utility.arange(0, TIME_MAX, t_step, endpoint=True)
    # `by` is all of the index levels except 'time'.
    by = dfr.index.names.difference({'time'})
    grouper = dfr.groupby(by)

    def get_one(group):
        t = group.index.get_level_values('time')
        # Shift start to 0.
        t -= t.min()
        # Only interpolate between start and extinction.
        # Round up to the next multiple of `t_step`.
        t_daily_max = numpy.ceil(t.max() / t_step) * t_step
        mask = t_daily <= t_daily_max
        # Interpolate from the closest point <= t_daily.
        return (
            group.set_index(t)
            .reindex(t_daily[mask], method='ffill')
        )

    daily = grouper.apply(get_one)
    return daily


def get_infected_daily(dfr):
    '''Get the infected sampled each day.'''
    return get_infected(
        get_daily(dfr)
    )


def get_extinction_time(dfr):
    '''Get the extinction time for each run.'''
    infected = get_infected(dfr)
    # `by` is all of the index levels except 'time'.
    by = dfr.index.names.difference({'time'})
    grouper = infected.groupby(by)

    def get_one(group):
        t = group.index.get_level_values('time')
        (t_start, t_end) = (t.min(), t.max())
        t_extinction = t_end - t_start
        (infected_end,) = group[t == t_end]
        observed = infected_end == 0
        assert observed or (t_extinction == TIME_MAX), \
            (observed, t_extinction, TIME_MAX)
        return {
            'time': t_extinction,
            'observed': observed,
        }

    extinction_time = (
        grouper.apply(get_one)
        # 'time' and 'observed' are the last index level:
        # make them columns.
        .unstack(-1)
        .astype({
            'time': dfr.index.dtypes['time'],
            'observed': bool,
        })
    )
    return extinction_time


def save_result(store, result,
                simulation=False, infected_daily=False, extinction_time=False):
    '''Save the result in `store`.'''
    if simulation:
        store.put('simulation', result)
    if infected_daily:
        inf = get_infected_daily(result)
        store.put('infected_daily', inf)
    if extinction_time:
        extime = get_extinction_time(result)
        store.put('extinction_time', extime)


def get_persistence(extinction_time):
    '''Get persistence from `extinction_time`.'''
    def get_one(group):
        persisted = ~group.observed
        return sum(persisted) / len(group)

    # `by` is all of the index levels except 'time' and 'run'.
    by = extinction_time.index.names.difference({'time', 'run'})
    if len(by) == 0:
        return get_one(extinction_time)
    grouper = extinction_time.groupby(by)
    return grouper.apply(get_one)
