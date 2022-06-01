'''Common plotting code.'''

import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import statsmodels.nonparametric.api

import h5
from herd.utility import arange
import run


# Science
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Fonts between 5pt and 7pt.
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...


t_name = 'time (y)'
cols_infected = ['exposed', 'infectious', 'chronic']


def _get_infected(dfr):
    return dfr[cols_infected].sum(axis='columns').rename('infected')


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


def build_downsampled(path_in, path_out,
                      t_min=0, t_max=10, t_step=1/365, by=None):
    t = arange(t_min, t_max, t_step, endpoint=True)
    with (h5.HDFStore(path_in, mode='r') as store_in,
          h5.HDFStore(path_out, mode='w') as store_out):
        if by is None:
            by = [n for n in store_in.get_index_names() if n != t_name]
        for (ix, group) in store_in.groupby(by):
            downsampled = _build_downsampled_group(group, t, t_step, by)
            levels = dict(zip(by, ix))
            run.prepend_index_levels(downsampled, **levels)
            assert numpy.all(downsampled.notnull().all())
            store_out.put(downsampled, index=False)
        store_out.create_table_index()
        store_out.repack()


def get_downsampled(path, by=None):
    path_downsampled = path.with_stem(path.stem + '_downsampled')
    if not path_downsampled.exists():
        build_downsampled(path, path_downsampled, by=by)
    return h5.HDFStore(path_downsampled, mode='r')


def _build_infected(path, path_out, by=None):
    with (get_downsampled(path, by=by) as store_in,
          h5.HDFStore(path_out, mode='w') as store_out):
        for chunk in store_in.select(columns=cols_infected, iterator=True):
            infected = _get_infected(chunk)
            store_out.put(infected, index=False)
        store_out.create_table_index()
        store_out.repack()


def get_infected(path, by=None):
    path_infected = path.with_stem(path.stem + '_infected')
    if not path_infected.exists():
        _build_infected(path, path_infected, by=by)
    infected = h5.load(path_infected)
    return infected


def _build_extinction_time_group(infected, tmax=10):
    t = infected.index.get_level_values(t_name)
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    assert observed or (time == tmax)
    return dict(time=time, observed=observed)


def _build_extinction_time(path, path_out, by=None):
    # Only the infected columns.
    extinction = {}
    with h5.HDFStore(path, mode='r') as store:
        if by is None:
            by = [n for n in store.get_index_names() if n != t_name]
        for (ix, group) in store.groupby(by, columns=cols_infected):
            infected = _get_infected(group)
            extinction[ix] = _build_extinction_time_group(infected)
    extinction = pandas.DataFrame.from_dict(extinction, orient='index')
    extinction.index.names = by
    extinction.sort_index(level=by, inplace=True)
    h5.dump(extinction, path_out, mode='w')


def get_extinction_time(path, by=None):
    path_extinction_time = path.with_stem(path.stem + '_extinction_time')
    if not path_extinction_time.exists():
        _build_extinction_time(path, path_extinction_time, by=by)
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
