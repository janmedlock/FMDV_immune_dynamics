'''Common plotting code.'''

import os

import matplotlib.collections
import matplotlib.pyplot
import numpy
import pandas
import psutil
import statsmodels.nonparametric.api

import h5
from herd.utility import arange


# Science
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Fonts between 5pt and 7pt.
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...

SATs = (1, 2, 3)

TMAX = 10

t_name = 'time (y)'

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


def _path_stem_append(path, postfix):
    return path.with_stem(path.stem + f'_{postfix}')


def _get_by(store, by):
    if by is None:
        # All the index levels except `t_name`.
        by = store.get_index_names() \
                  .difference({t_name})
    return by


def _build_downsampled_group(group, t, t_step, by):
    # Only keep time index.
    group = group.reset_index(by, drop=True)
    # Shift start to 0.
    group.index -= group.index.min()
    # Only interpolate between start and extinction.
    # Round up to the next multiple of `t_step`.
    t_max = numpy.ceil(group.index.max() / t_step) * t_step
    mask = (t <= t_max)
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def _build_downsampled(path_in, path_out,
                       t_min=0, t_max=TMAX, t_step=1/365,
                       by=None):
    t = arange(t_min, t_max, t_step, endpoint=True)
    with h5.HDFStore(path_out, mode='w') as store_out:
        with h5.HDFStore(path_in, mode='r') as store_in:
            by = _get_by(store_in, by)
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


def get_infected(dfr):
    return dfr[cols_infected].sum(axis='columns') \
                             .rename('infected')


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


def _get_extinction_time_one(dfr):
    infected = get_infected(dfr)
    t = infected.index.get_level_values(t_name)
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    assert observed or (time == TMAX)
    return dict(time=time, observed=observed)


def get_extinction_time(store, by=None, **kwds):
    by = _get_by(store, by)
    grouper = store.groupby(by, columns=cols_infected, **kwds)
    extinction_time = {ix: _get_extinction_time_one(group)
                       for (ix, group) in grouper}
    extinction_time = pandas.DataFrame.from_dict(extinction_time,
                                                 orient='index') \
                                      .rename_axis(by, axis='index') \
                                      .sort_index(level=by)
    return extinction_time


def _build_extinction_time(path, path_out):
    with h5.HDFStore(path, mode='r') as store:
        extinction_time = get_extinction_time(store)
    h5.dump(extinction_time, path_out, mode='w')


def get_path_extinction_time(path):
    return _path_stem_append(path, 'extinction_time')


def load_extinction_time(path):
    path_extinction_time = get_path_extinction_time(path)
    if not path_extinction_time.exists():
        _build_extinction_time(path, path_extinction_time)
    extinction_time = h5.load(path_extinction_time)
    return extinction_time


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
