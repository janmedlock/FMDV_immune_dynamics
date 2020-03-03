#!/usr/bin/python3

import os.path

from matplotlib import pyplot
import numpy
import pandas
import seaborn

import h5
import plot_common
import run_common


filename = f'run_start_times_SATs.h5'


def get_downsampled():
    t_max = 10 + 11 / 12
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    if not os.path.exists(filename_ds):
        plot_common.build_downsampled(filename, t_max=t_max)
    return h5.HDFStore(filename_ds, mode='r')


def _build_infected(filename_out):
    store = get_downsampled()
    where = 'start_time=0'
    columns = ['exposed', 'infectious', 'chronic']
    infected = []
    for chunk in store.select(where=where, columns=columns, iterator=True):
        infected.append(chunk.sum(axis='columns'))
    infected = pandas.concat(infected, copy=False)
    infected.name = 'infected'
    h5.dump(infected, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_infected():
    filename_infected = f'plot_start_times_SATs_infected.h5'
    try:
        infected = h5.load(filename_infected)
    except FileNotFoundError:
        _build_infected(filename_infected)
        infected = h5.load(filename_infected)
    return infected


def plot_infected_facet(infected, color=None, alpha=1, **kwargs):
    # `color` is ignored so each run gets a different color.
    infected = infected.unstack('run')
    t = infected.index.get_level_values(plot_common.t_name).sort_values()
    t -= t.min()
    # Convert to days.
    t *= 365
    pyplot.plot(t, infected, alpha=alpha, **kwargs)
    # `infection.fillna(0)` gives mean including those that have gone extinct.
    # Use `alpha=1` for the mean, not the `alpha` value passed in.
    pyplot.plot(t, infected.fillna(0).mean(axis=1),
                color='black', alpha=1, **kwargs)


def plot_infected():
    infected = get_infected()
    infected.reset_index(inplace=True)
    pyplot.figure()
    # g = seaborn.FacetGrid(data=infected,
    #                       row='SAT', col='start_time',
    #                       sharey='row')
    g = seaborn.FacetGrid(data=infected, col='SAT')
    g.map(plot_infected_facet, 'infected', alpha=0.3)
    g.set_axis_labels('time (days)', 'number infected')
    g.set_titles('{col_var} {col_name}')
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_infected.pdf')


def _build_extinction_time_group(infected):
    if infected.iloc[-1] == 0:
        t = infected.index.get_level_values(plot_common.t_name)
        return t.max() - t.min()
    else:
        return numpy.nan


def _build_extinction_time(filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != plot_common.t_name]
        # Only the first start time.
        where = 'start_time=0'
        # Only the infected columns.
        columns = ['exposed', 'infectious', 'chronic']
        ser = {}
        for (ix, group) in store.groupby(by, where=where, columns=columns):
            infected = group.sum(axis='columns')
            ser[ix] = _build_extinction_time_group(infected)
    ser = pandas.Series(ser, name='extinction time (days)')
    ser.rename_axis(by, inplace=True)
    ser *= 365
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_extinction_time():
    filename_et = f'plot_start_times_SATs_extinction_time.h5'
    try:
        extinction_time = h5.load(filename_et)
    except FileNotFoundError:
        _build_extinction_time(filename_et)
        extinction_time = h5.load(filename_et)
    return extinction_time


def plot_extinction_time():
    extinction_time = get_extinction_time()
    extinction_time.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=extinction_time,
    #                    x='extinction time (days)', y='start_time', col='SAT',
    #                    kind='box', orient='horizontal', sharey=False)
    ax = seaborn.violinplot(data=extinction_time,
                            x='extinction time (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.xlim(left=0)
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_extinction_time.pdf')


def _build_time_to_peak_group(infected):
    t = infected.index.get_level_values(plot_common.t_name)
    m = infected.index.get_loc(infected.idxmax())
    return (t[m] - t.min())


def _build_time_to_peak(filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != plot_common.t_name]
        # Only the first start time.
        where = 'start_time=0'
        # Only the infected columns.
        columns = ['exposed', 'infectious', 'chronic']
        ser = {}
        for (ix, group) in store.groupby(by, where=where, columns=columns):
            infected = group.sum(axis='columns')
            ser[ix] = _build_time_to_peak_group(infected)
    ser = pandas.Series(ser, name='time to peak (days)')
    ser.rename_axis(by, inplace=True)
    ser *= 365
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_time_to_peak():
    filename_ttp = f'plot_start_times_SATs_time_to_peak.h5'
    try:
        extinction_time = h5.load(filename_ttp)
    except FileNotFoundError:
        _build_time_to_peak(filename_ttp)
        time_to_peak = h5.load(filename_ttp)
    return time_to_peak


def plot_time_to_peak():
    time_to_peak = get_time_to_peak()
    time_to_peak.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=time_to_peak,
    #                    x='time to peak (days)', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    ax = seaborn.violinplot(data=time_to_peak,
                            x='time to peak (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_time_to_peak.pdf')


def _build_total_infected_group(df):
    R = df['recovered']
    # This sucks to approximate total infected.
    return R.iloc[-1] - R.iloc[0]


def _build_total_infected(filename_out):
    with h5.HDFStore(filename, mode='r') as store:
        by = [n for n in store.get_index_names() if n != plot_common.t_name]
        # Only the first start time.
        where = 'start_time=0'
        # Only the recovered column.
        columns = ['recovered']
        ser = {}
        for (ix, group) in store.groupby(by, where=where, columns=columns):
            ser[ix] = _build_total_infected_group(group)
    ser = pandas.Series(ser, name='total infected')
    ser.rename_axis(by, inplace=True)
    ser.clip_lower(0, inplace=True)
    h5.dump(ser, filename_out, mode='w',
            min_itemsize=run_common._min_itemsize)


def get_total_infected():
    filename_ti = f'plot_start_times_SATs_total_infected.h5'
    try:
        total_infected = h5.load(filename_ti)
    except FileNotFoundError:
        _build_total_infected(filename_ti)
        total_infected = h5.load(filename_ti)
    return total_infected


def plot_total_infected():
    total_infected = get_total_infected()
    total_infected.reset_index(inplace=True)
    pyplot.figure()
    # seaborn.factorplot(data=total_infected,
    #                    x='total infected', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    ax = seaborn.violinplot(data=total_infected,
                            x='total infected', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig(f'plot_start_times_SATs_total_infected.pdf')


if __name__ == '__main__':
    plot_infected()
    # plot_extinction_time()
    # plot_time_to_peak()
    # plot_total_infected()
    # pyplot.show()
