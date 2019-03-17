#!/usr/bin/python3

import os.path

from matplotlib import pyplot
import seaborn

import h5
import plot_common


filename = f'run_start_times_SATs.h5'


def get_downsampled(model='acute'):
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    try:
        data_ds = h5.load(filename_ds)
    except FileNotFoundError:
        plot_common.downsample(filename)
        data_ds = h5.load(filename_ds)
    return data_ds.loc[model]


def get_infected(model='acute'):
    filename_infected = f'plot_start_times_SATs_infected.h5'
    try:
        infected = h5.load(filename_infected)
    except FileNotFoundError:
        # It might be helpful to reduce what is loaded here to
        # `where='start_time=0'` and
        # `columns=['exposed', 'infectious', 'chronic']`.
        data = get_downsampled(model=model)
        # Only plot the first start time.
        mask = (data.index.get_level_values('start_time') == 0.)
        data = data[mask]
        infected = data[['exposed', 'infectious', 'chronic']].sum(axis=1)
        infected = infected.to_frame(name='infected')
        h5.dump(infected, filename_infected)
    return infected.loc[model]


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


def plot_infected(model='acute'):
    infected = get_infected(model=model)
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
    pyplot.savefig(f'plot_start_times_SATs_infected_{model}.pdf')


def get_extinction_time_one(infected):
    if infected.iloc[-1] == 0:
        t = infected.index.get_level_values(plot_common.t_name)
        return t.max() - t.min()
    else:
        return None


def get_extinction_time(model='acute'):
    filename_et = f'plot_start_times_SATs_extinction_time.h5'
    try:
        extinction_time = h5.load(filename_et)
    except FileNotFoundError:
        # Only plot the first start time.
        where = 'start_time=0'
        columns = ['exposed', 'infectious', 'chronic']
        data = h5.load(filename, where=where, columns=columns)
        not_t_names = [n for n in data.index.names if n != plot_common.t_name]
        infected = data[['exposed', 'infectious', 'chronic']].sum(axis=1)
        infected.name = 'infected'
        extinction_time = infected.groupby(level=not_t_names).aggregate(
            get_extinction_time_one)
        extinction_time *= 365
        extinction_time = extinction_time.to_frame(
            name='extinction time (days)')
        h5.dump(extinction_time, filename_et)
    return extinction_time.loc[model]


def plot_extinction_time(model='acute'):
    extinction_time = get_extinction_time(model=model)
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
    pyplot.savefig(f'plot_start_times_SATs_extinction_time_{model}.pdf')


def get_time_to_peak_one(infected):
    t = infected.index.get_level_values(plot_common.t_name)
    m = infected.index.get_loc(infected.idxmax())
    return (t[m] - t.min())


def get_time_to_peak(model='acute'):
    filename_ttp = f'plot_start_times_SATs_time_to_peak.h5'
    try:
        extinction_time = h5.load(filename_ttp)
    except FileNotFoundError:
        # Only plot the first start time.
        where = 'start_time=0'
        columns = ['exposed', 'infectious', 'chronic']
        data = h5.load(filename, where=where, columns=columns)
        not_t_names = [n for n in data.index.names if n != plot_common.t_name]
        infected = data[['exposed', 'infectious', 'chronic']].sum(axis=1)
        infected.name = 'infected'
        time_to_peak = infected.groupby(level=not_t_names).aggregate(
            get_time_to_peak_one)
        time_to_peak *= 365
        time_to_peak = time_to_peak.to_frame(name='time to peak (days)')
        h5.dump(time_to_peak, filename_ttp)
    return time_to_peak.loc[model]


def plot_time_to_peak(model='acute'):
    time_to_peak = get_time_to_peak(model=model)
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
    pyplot.savefig(f'plot_start_times_SATs_time_to_peak_{model}.pdf')


def get_total_infected_one(x):
    R = x['recovered']
    # This sucks to approximate total infected.
    return R.iloc[-1] - R.iloc[0]


def get_total_infected(model='acute'):
    filename_ti = f'plot_start_times_SATs_total_infected.h5'
    try:
        total_infected = h5.load(filename_ti)
    except FileNotFoundError:
        # Only plot the first start time.
        where = 'start_time=0'
        columns = ['recovered']
        data = h5.load(filename, where=where, columns=columns)
        not_t_names = [n for n in data.index.names if n != plot_common.t_name]
        total_infected = data.groupby(level=not_t_names).aggregate(
            get_total_infected_one)
        total_infected.clip_lower(0, inplace=True)
        total_infected = total_infected.to_frame(name='total infected')
        h5.dump(total_infected, filename_ti)
    return total_infected.loc[model]


def plot_total_infected(model='acute'):
    total_infected = get_total_infected(model=model)
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
    pyplot.savefig(f'plot_start_times_SATs_total_infected_{model}.pdf')


if __name__ == '__main__':
    model = 'chronic'

    plot_infected(model)
    # plot_extinction_time(model)
    # plot_time_to_peak(model)
    # plot_total_infected(model)
    # pyplot.show()
