'''Common code, much of it for plotting.'''

import itertools
import os
import stat

import astropy.units
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas
import psutil
import statsmodels.nonparametric.api

import _dask_dataframe
import h5
from herd.utility import arange


rc = {
    'font.family': 'serif',
    'font.serif': 'Liberation Serif',  # Substitute for 'Times New Roman'
    'figure.constrained_layout.use': True,
    'figure.dpi': 300,
}

_TICK_LABELSIZE = 8
rc_text_small = {
    'font.size': 7.5,  # Minimium for Proc B.
    'xtick.labelsize': _TICK_LABELSIZE,
    'ytick.labelsize': _TICK_LABELSIZE,
    'axes.labelsize': 9,
    'axes.titlesize': 11,
}

_mm_to_inch = astropy.units.mm.to(astropy.units.imperial.inch)

WIDTH_MAXIMUM = {
    'single_column': 84 * _mm_to_inch,
    'double_column': 175 * _mm_to_inch,
}

HEIGHT_MAXIMUM = 250 * _mm_to_inch

SATs = (1, 2, 3)

NRUNS = 1000

TIME_UNIT = 'year'
TIME_MAX = 10
TIME_LABEL = f'Time ({TIME_UNIT})'

cols_infected = ['exposed', 'infectious', 'chronic']


def is_increasing(arr, strict=False, axis=-1):
    '''Check whether `arr` is increasing along `axis`.'''
    if strict:
        test = numpy.greater
    else:
        test = numpy.greater_equal
    return test(numpy.diff(arr, axis=axis), 0).all(axis=axis)


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


def set_read_only(path):
    '''Set `path` as read only.'''
    return path.chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    )


def _path_stem_append(path, postfix):
    return path.with_stem(path.stem + f'_{postfix}')


def _get_by(dfr, by=None):
    if by is None:
        # `by` is all of the index levels except 'time'.
        levels = dfr.index \
                    .to_frame() \
                    .columns
        by = list(levels.difference({'time'}))
    return by


def _is_dask(dfr):
    return isinstance(
        dfr,
        (_dask_dataframe.DataFrame, _dask_dataframe.Series)
    )


def get_path_downsampled(path):
    return _path_stem_append(path, 'downsampled')


def get_downsampled(dfr, t_min=0, t_max=TIME_MAX, t_step=1/365):
    t = arange(t_min, t_max, t_step, endpoint=True)
    by = _get_by(dfr)
    # `.groupby()` does not seem to work on index levels.
    grouper = dfr.reset_index() \
                 .groupby(by)

    def get_one(group):
        time = group['time']
        # Shift start to 0.
        time -= time.min()
        # Only interpolate between start and extinction.
        # Round up to the next multiple of `t_step`.
        t_max = numpy.ceil(time.max() / t_step) * t_step
        mask = t <= t_max
        # Interpolate from the closest point <= t.
        return (
            group.set_index('time')
            .sort_index()
            .reindex(t[mask], method='ffill')
            .reset_index()
            .loc[:, group.columns]
        )

    apply_kwds = {}
    if _is_dask(dfr):
        apply_kwds['meta'] = _dask_dataframe.utils.make_meta(
            dfr.reset_index()
        )
    downsampled = grouper.apply(get_one, **apply_kwds)
    if _is_dask(dfr):
        downsampled = downsampled.compute()
    return (
        downsampled.set_index(list(dfr.index.to_frame().columns))
        .sort_index()
    )


def build_downsampled(path):
    '''Build the downsampled store.'''
    dfr = h5.load(path, use_dask=True)
    downsampled = get_downsampled(dfr)
    path_downsampled = get_path_downsampled(path)
    h5.dump(downsampled, path_downsampled, mode='w')
    return downsampled


def load_downsampled(path, **kwds):
    path_downsampled = get_path_downsampled(path)
    try:
        return h5.load(path_downsampled, **kwds)
    except OSError:
        pass
    downsampled = build_downsampled(path)
    if len(kwds) == 0:
        return downsampled
    return load_downsampled(path, **kwds)


def sum_infected(obj):
    '''Get the number of infected at each time.'''
    infected = obj[cols_infected]
    if isinstance(infected, (pandas.Series, _dask_dataframe.Series)):
        return infected.sum()
    if isinstance(infected, (pandas.DataFrame, _dask_dataframe.DataFrame)):
        return infected.sum(axis='columns') \
                       .rename('infected')
    raise ValueError(f'Unknown {type(obj)=}!')


def get_path_infected(path):
    return _path_stem_append(path, 'infected')


def get_infected(dfr):
    '''Get the infected.'''
    infected = sum_infected(dfr)
    if _is_dask(dfr):
        infected = infected.compute()
    return infected.sort_index()


def build_infected(path):
    '''Build the infected store.'''
    downsampled = load_downsampled(path, columns=cols_infected, use_dask=True)
    infected = get_infected(downsampled)
    path_infected = get_path_infected(path)
    h5.dump(infected, path_infected, mode='w')
    return infected


def load_infected(path, **kwds):
    path_infected = get_path_infected(path)
    try:
        return h5.load(path_infected, **kwds)
    except OSError:
        pass
    infected = build_infected(path)
    if len(kwds) == 0:
        return infected
    return load_infected(path, **kwds)


def get_path_extinction_time(path):
    return _path_stem_append(path, 'extinction_time')


def get_extinction_time(dfr):
    '''Get the extinction time for each run.'''
    infected = sum_infected(dfr)
    by = _get_by(dfr)
    # `.groupby()` does not seem to work on index levels.
    grouper = infected.reset_index() \
                      .groupby(by)

    def get_one(group):
        t = group['time']
        (t_start, t_end) = (t.min(), t.max())
        time = t_end - t_start
        (infected_end,) = group['infected'][t == t_end]
        observed = infected_end == 0
        assert observed or (time == TIME_MAX), (observed, time, TIME_MAX)
        return pandas.Series({
            'time': time,
            'observed': observed,
        })

    apply_kwds = {}
    if _is_dask(dfr):
        apply_kwds['meta'] = {
            'time': dfr.index.to_frame().dtypes['time'],
            'observed': bool,
        }
    extinction_time = grouper.apply(get_one, **apply_kwds)
    if _is_dask(dfr):
        extinction_time = extinction_time.compute()
    return extinction_time.sort_index()


def build_extinction_time(path):
    '''Build the extinction-time store.'''
    dfr = h5.load(path, columns=cols_infected, use_dask=True)
    extinction_time = get_extinction_time(dfr)
    path_extinction_time = get_path_extinction_time(path)
    h5.dump(extinction_time, path_extinction_time, mode='w')
    return extinction_time


def load_extinction_time(path, **kwds):
    path_extinction_time = get_path_extinction_time(path)
    try:
        return h5.load(path_extinction_time, **kwds)
    except OSError:
        pass
    extinction_time = build_extinction_time(path)
    if len(kwds) == 0:
        return extinction_time
    return load_extinction_time(path, **kwds)


def get_persistence(extinction_time):
    persisted = ~extinction_time.observed
    return sum(persisted) / len(extinction_time)


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)


def get_density(endog, times):
    # Avoid errors if endog is empty.
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=0)
        return kde.evaluate(times)
    else:
        return numpy.zeros_like(times)


def kdeplot(endog, ax=None, shade=False, cut=0, **kwds):
    if ax is None:
        ax = matplotlib.pyplot.gca()
    endog = endog.dropna()
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=cut)
        x = numpy.linspace(kde.support.min(), kde.support.max(), 301)
        y = kde.evaluate(x)
    else:
        x, y = [], []
    line, = ax.plot(x, y, **kwds)
    if shade:
        shade_kws = dict(
            facecolor=kwds.get('facecolor', line.get_color()),
            alpha=kwds.get('alpha', 0.25),
            clip_on=kwds.get('clip_on', True),
            zorder=kwds.get('zorder', 1))
        ax.fill_between(x, 0, y, **shade_kws)
    return ax


# Erin's colors.
SAT_colors = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}


def get_cmap_SAT(SAT):
    '''White to `SAT_colors[SAT]`.'''
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'name', ['white', SAT_colors[SAT]])


def legend_multicolumn(obj, handles, labels, ncol, **kwds):
    '''Make a multicolumn legend.'''
    def reorder(items, ncol):
        return list(
            itertools.chain.from_iterable(
                items[i::ncol] for i in range(ncol)
            )
        )
    return obj.legend(reorder(handles, ncol),
                      reorder(labels, ncol),
                      ncol=ncol, **kwds)


def get_state_label(state):
    '''Make a plot label for `state`.'''
    return state.replace('_', ' ').capitalize()
