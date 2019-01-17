#!/usr/bin/python3

import os.path

from matplotlib import pyplot
import seaborn

import h5
import plot_common


def get_filename(chronic=False):
    filename = 'run_start_times_SATs'
    if chronic:
        filename += '_chronic'
    filename += '.h5'
    return filename


def get_downsampled(chronic=False):
    filename = get_filename(chronic=chronic)
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    try:
        data_ds = h5.load(filename_ds)
    except FileNotFoundError:
        data_ds = plot_common.downsample(filename)
        h5.dump(data_ds, filename_ds)
    return data_ds


def get_persistence_time(x):
    t = x.index.get_level_values(plot_common.t_name)
    return t.max() - t.min()


def plot_persistence_time(chronic=False):
    filename = get_filename(chronic=chronic)
    # Only plot the first start time.
    where = 'start_time=0'
    # Persistence time is calculated using only the time values
    # in the index, so no data columns are needed.
    data = h5.load(filename, where=where, columns=[])
    not_t_names = [n for n in data.index.names if n != plot_common.t_name]
    persistence_time = data.groupby(level=not_t_names).apply(
        get_persistence_time)
    persistence_time *= 365
    persistence_time = persistence_time.reset_index(
        name='persistence time (days)')
    # seaborn.factorplot(data=persistence_time,
    #                    x='persistence time (days)', y='start_time', col='SAT',
    #                    kind='box', orient='horizontal', sharey=False)
    pyplot.figure()
    ax = seaborn.violinplot(data=persistence_time,
                            x='persistence time (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    filename = 'plot_start_times_SATs_persistence_time'
    if chronic:
        filename += '_chronic'
    filename += '.pdf'
    pyplot.savefig(filename)


def plot_infected_facet(x, color=None, alpha=1, **kwargs):
    # `color` is ignored so each run gets a different color.
    x = x.unstack('run')
    t = x.index - x.index.min()
    pyplot.plot(365 * t, x, alpha=alpha, **kwargs)
    # `x.fillna(0)` gives mean including those that have gone extinct.
    # Use `alpha=1` for the mean, not the `alpha` value passed in.
    pyplot.plot(365 * t, x.fillna(0).mean(axis=1), color='black', alpha=1,
                **kwargs)


def plot_infected(chronic=False):
    # It might be helpful to reduce what is loaded here to
    # `where='start_time=0'` and
    # `columns=['exposed', 'infectious', 'chronic']`.
    data = get_downsampled(chronic=chronic)
    # Only plot the first start time.
    mask = (data.index.get_level_values('start_time') == 0.)
    data = data[mask]
    infected = data[['exposed', 'infectious', 'chronic']].sum(axis=1)
    infected = infected.reset_index(level=['SAT', 'start_time'],
                                    name='infected')
    # g = seaborn.FacetGrid(data=infected, row='SAT', col='start_time',
    #                       sharey='row')
    g = seaborn.FacetGrid(data=infected, col='SAT')
    g.map(plot_infected_facet, 'infected', alpha=0.3)
    g.set_axis_labels('time (days)', 'number infected')
    g.set_titles('{col_var} {col_name}')
    pyplot.tight_layout()
    filename = 'plot_start_times_SATs_infected'
    if chronic:
        filename += '_chronic'
    filename += '.pdf'
    pyplot.savefig(filename)


def get_time_to_peak(x):
    I = x[['exposed', 'infectious', 'chronic']].sum(axis=1)
    return I.idxmax()[-1] - I.index.get_level_values(plot_common.t_name).min()


def plot_time_to_peak(chronic=False):
    filename = get_filename(chronic=chronic)
    # Only plot the first start time.
    where = 'start_time=0'
    data = h5.load(filename, where=where,
                   columns=['exposed', 'infectious', 'chronic'])
    not_t_names = [n for n in data.index.names if n != plot_common.t_name]
    time_to_peak = data.groupby(level=not_t_names).apply(
        get_time_to_peak)
    time_to_peak *= 365
    time_to_peak = time_to_peak.reset_index(level=['SAT', 'start_time'],
                                            name='time to peak (days)')
    # seaborn.factorplot(data=time_to_peak,
    #                    x='time to peak (days)', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    pyplot.figure()
    ax = seaborn.violinplot(data=time_to_peak,
                            x='time to peak (days)', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    filename = 'plot_start_times_SATs_time_to_peak'
    if chronic:
        filename += '_chronic'
    filename += '.pdf'
    pyplot.savefig(filename)


def get_total_infected(x):
    R = x['recovered']
    # This sucks to approximate total infected.
    return R.iloc[-1] - R.iloc[0]


def plot_total_infected(chronic=False):
    filename = get_filename(chronic=chronic)
    # Only plot the first start time.
    where = 'start_time=0'
    data = h5.load(filename, where=where, columns=['recovered'])
    not_t_names = [n for n in data.index.names if n != plot_common.t_name]
    total_infected = data.groupby(level=not_t_names).apply(get_total_infected)
    total_infected.clip_lower(0, inplace=True)
    total_infected = total_infected.reset_index(level=['SAT', 'start_time'],
                                                name='total infected')
    # seaborn.factorplot(data=total_infected,
    #                    x='total infected', y='start_time', col='SAT',
    #                    sharey=False,
    #                    kind='violin', orient='horizontal',
    #                    cut=0, linewidth=1)
    pyplot.figure()
    ax = seaborn.violinplot(data=total_infected,
                            x='total infected', y='SAT',
                            orient='horizontal', width=0.95,
                            cut=0, linewidth=1)
    plot_common.set_violins_linewidth(ax, 0)
    pyplot.ylabel('')
    locs, labels = pyplot.yticks()
    pyplot.yticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    filename = 'plot_start_times_SATs_total_infected'
    if chronic:
        filename += '_chronic'
    filename += '.pdf'
    pyplot.savefig(filename)


if __name__ == '__main__':
    chronic = True

    plot_infected(chronic)
    # plot_persistence_time(chronic)
    # plot_time_to_peak(chronic)
    # plot_total_infected(chronic)
    # pyplot.show()
