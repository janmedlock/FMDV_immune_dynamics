#!/usr/bin/python3

from matplotlib import pyplot
import numpy
import pandas
import seaborn


t_name = 'time (y)'


def get_data():
    data = pandas.read_pickle('run_start_times_SATs.pkl')
    try:
        data_ds = pandas.read_pickle('run_start_times_SATs_downsampled.pkl')
    except FileNotFoundError:
        data_ds = downsample(data)
    return (data, data_ds)


def downsample(data):
    not_t_names = [n for n in data.index.names if n != t_name]
    # Daily timesteps.
    t = data.index.get_level_values(t_name).unique()
    t = numpy.arange(t.min(), t.max(), 1 / 365)
    not_t = [data.index.get_level_values(n).unique()
             for n in not_t_names]
    midx = pandas.MultiIndex.from_product(not_t + [t],
                                          names=data.index.names)
    data_ds = pandas.DataFrame(index=midx, columns=data.columns)
    # Loop over each run.
    for k, x in data.groupby(level=not_t_names):
        x = x.copy()
        # Only keep time index.
        x.index = x.index.get_level_values(t_name)
        # Only interpolate between start and extinction.
        mask = ((t >= x.index.min()) & (t <= x.index.max()))
        # Interpolate from previous point.
        data_ds.loc[k][mask] = x.reindex(t[mask], method='ffill')
    data_ds.dropna(axis=0, inplace=True)
    data_ds.to_pickle('run_start_times_SATs_downsampled.pkl')
    return data_ds


def get_persistence_time(x):
    t = x.index.get_level_values(t_name)
    return t.max() - t.min()


def plot_persistence_time(data):
    # Only plot the first start time.
    mask = (data.index.get_level_values('start_time') == 0.)
    data = data[mask]
    not_t_names = [n for n in data.index.names if n != t_name]
    persistence_time = data.groupby(level=not_t_names).apply(
        get_persistence_time)
    persistence_time *= 365
    persistence_time = persistence_time.reset_index(
        name='persistence time (days)')
    # seaborn.factorplot(data=persistence_time,
    #                    x='start_time', y='persistence time (days)', col='SAT',
    #                    kind='box', sharey=False)
    pyplot.figure()
    seaborn.violinplot(data=persistence_time,
                       x='SAT', y='persistence time (days)',
                       cut=0, linewidth=1)
    pyplot.xlabel('')
    locs, labels = pyplot.xticks()
    pyplot.xticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig('plot_start_times_SATs_persistence_time.pdf')


def plot_infected_facet(x, color=None, alpha=1, **kwargs):
    # `color` is ignored so each run gets a different color.
    x = x.unstack('run')
    t = x.index - x.index.min()
    pyplot.plot(365 * t, x, alpha=alpha, **kwargs)
    # Use `alpha=1` for the mean, not the value passed in.
    pyplot.plot(365 * t, x.mean(axis=1), color='black', alpha=1, **kwargs)


def plot_infected(data):
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
    pyplot.savefig('plot_start_times_SATs_infected.pdf')


def get_time_to_peak(x):
    I = x[['exposed', 'infectious', 'chronic']].sum(axis=1)
    return I.idxmax()[-1] - I.index.get_level_values(t_name).min()


def plot_time_to_peak(data):
    # Only plot the first start time.
    mask = (data.index.get_level_values('start_time') == 0.)
    data = data[mask]
    not_t_names = [n for n in data.index.names if n != t_name]
    time_to_peak = data.groupby(level=not_t_names).apply(
        get_time_to_peak)
    time_to_peak *= 365
    time_to_peak = time_to_peak.reset_index(level=['SAT', 'start_time'],
                                            name='time to peak (days)')
    # seaborn.factorplot(data=time_to_peak,
    #                    x='start_time', y='time to peak (days)', col='SAT',
    #                    sharey=False,
    #                    kind='violin', cut=0, linewidth=1)
    pyplot.figure()
    seaborn.violinplot(data=time_to_peak,
                       x='SAT', y='time to peak (days)',
                       cut=0, linewidth=1)
    pyplot.xlabel('')
    locs, labels = pyplot.xticks()
    pyplot.xticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig('plot_start_times_SATs_time_to_peak.pdf')


def get_total_infected(x):
    R = x['recovered']
    # This sucks to approximate total infected.
    return R.iloc[-1] - R.iloc[0]


def plot_total_infected(data):
    # Only plot the first start time.
    mask = (data.index.get_level_values('start_time') == 0.)
    data = data[mask]
    not_t_names = [n for n in data.index.names if n != t_name]
    total_infected = data.groupby(level=not_t_names).apply(get_total_infected)
    total_infected.clip_lower(0, inplace=True)
    total_infected = total_infected.reset_index(level=['SAT', 'start_time'],
                                                name='total infected')
    # seaborn.factorplot(data=total_infected,
    #                    x='start_time', y='total infected', col='SAT',
    #                    sharey=False,
    #                    kind='violin', cut=0, linewidth=1)
    pyplot.figure()
    seaborn.violinplot(data=total_infected,
                       x='SAT', y='total infected',
                       cut=0, linewidth=1)
    pyplot.xlabel('')
    locs, labels = pyplot.xticks()
    pyplot.xticks(locs, ['SAT {}'.format(i.get_text()) for i in labels])
    pyplot.tight_layout()
    pyplot.savefig('plot_start_times_SATs_total_infected.pdf')


if __name__ == '__main__':
    data, data_ds = get_data()
    plot_infected(data_ds)
    plot_persistence_time(data)
    plot_time_to_peak(data)
    plot_total_infected(data)
    # pyplot.show()
