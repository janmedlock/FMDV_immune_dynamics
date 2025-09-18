'''Common code, much of it for plotting.'''

import itertools
import os
import stat

import astropy.units
import dask.dataframe
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas
import psutil
import statsmodels.nonparametric.api

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

NRUNS = 1000

SATs = (1, 2, 3)

TMAX = 10

t_name = 'time (y)'
t_label = 'Time (year)'

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


def get_by(obj, by=None):
    if by is None:
        # `by` is all of the index levels except `t_name`.
        if isinstance(obj, h5.HDFStore):
            levels = obj.get_index_names()
        elif isinstance(obj, dask.dataframe.DataFrame):
            levels = obj.index \
                        .to_frame() \
                        .columns
        else:
            raise ValueError(f'Unknown {type(obj)=}!')
        by = levels.difference({t_name})
    return by


def _build_downsampled_group(group, t, t_step, by):
    # Only keep time index.
    group = group.reset_index(by, drop=True)
    # Shift start to 0.
    group.index -= group.index[0]
    # Only interpolate between start and extinction.
    # Round up to the next multiple of `t_step`.
    t_max = numpy.ceil(group.index[-1] / t_step) * t_step
    mask = (t <= t_max)
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def _build_downsampled(path_in, path_out,
                       t_min=0, t_max=TMAX, t_step=1/365,
                       by=None):
    t = arange(t_min, t_max, t_step, endpoint=True)
    with h5.HDFStore(path_out, mode='w') as store_out:
        with h5.HDFStore(path_in, mode='r') as store_in:
            by = get_by(store_in, by)
            grouper = store_in.groupby(by)
            for (ix, group) in grouper:
                downsampled = _build_downsampled_group(group, t, t_step, by)
                levels = dict(zip(by, ix))
                prepend_index_levels(downsampled, **levels)
                assert numpy.all(downsampled.notnull().all())
                store_out.put(downsampled, index=False)
        store_out.create_table_index()
        store_out.repack()


def get_path_downsampled(path):
    return _path_stem_append(path, 'downsampled')


def load_downsampled(path):
    path_downsampled = get_path_downsampled(path)
    if not path_downsampled.exists():
        _build_downsampled(path, path_downsampled)
    return h5.HDFStore(path_downsampled, mode='r')


def get_infected(obj):
    '''Get the number of infected at each time.'''
    infected = obj[cols_infected]
    if isinstance(infected, (pandas.Series, dask.dataframe.Series)):
        return infected.sum()
    if isinstance(infected, (pandas.DataFrame, dask.dataframe.DataFrame)):
        return infected.sum(axis='columns') \
                       .rename('infected')
    raise ValueError(f'Unknown {type(obj)=}!')


def _build_infected(path, path_out):
    with h5.HDFStore(path_out, mode='w') as store_out:
        with load_downsampled(path) as store_in:
            chunker = store_in.select(columns=cols_infected,
                                      iterator=True)
            for chunk in chunker:
                infected = get_infected(chunk)
                store_out.put(infected, index=False)
        store_out.create_table_index()
        store_out.repack()


def get_path_infected(path):
    return _path_stem_append(path, 'infected')


def load_infected(path):
    path_infected = get_path_infected(path)
    if not path_infected.exists():
        _build_infected(path, path_infected)
    infected = h5.load(path_infected)
    return infected


def get_extinction_time(dfr, by=None, **kwds):
    '''Get the extinction time for each run.'''
    by = get_by(dfr, by)
    infected = get_infected(dfr)
    # `.groupby()` does not seem to work on index levels.
    grouper = infected.reset_index() \
                      .groupby(list(by), **kwds)

    def get_one(group):
        t = group[t_name]
        (t_start, t_end) = (t.min(), t.max())
        time = t_end - t_start
        (infected_end,) = group['infected'][t == t_end]
        observed = infected_end == 0
        assert observed or (time == TMAX), (observed, time, TMAX)
        return pandas.Series({
            'time': time,
            'observed': observed,
        })

    dtypes = {
        'time': grouper.dfr.dtypes[t_name],
        'observed': bool,
    }
    deferred = grouper.apply(get_one,
                             meta=dtypes)
    extinction_time = deferred.compute() \
                              .sort_index()
    return extinction_time


def _build_extinction_time(path, path_out):
    dfr = h5.load_dask(path,
                       columns=cols_infected)
    extinction_time = get_extinction_time(dfr)
    h5.dump(extinction_time, path_out, mode='w')


def get_path_extinction_time(path):
    return _path_stem_append(path, 'extinction_time')


def build_extinction_time(path):
    '''Build the extinction-time store.'''
    path_extinction_time = get_path_extinction_time(path)
    _build_extinction_time(path, path_extinction_time)


def load_extinction_time(path):
    path_extinction_time = get_path_extinction_time(path)
    if not path_extinction_time.exists():
        _build_extinction_time(path, path_extinction_time)
    extinction_time = h5.load(path_extinction_time)
    return extinction_time


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
