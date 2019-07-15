#!/usr/bin/python3

from matplotlib import colors, pyplot, ticker
import numpy
import pandas
import seaborn
import statsmodels.nonparametric.api

import h5
import stats


def _get_extinction(infected):
    t = infected.index.get_level_values('time (y)')
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    return {'extinction_time': time,
            'extinction_observed': observed}


def _load_extinction_times():
    with h5.HDFStore('run_population_sizes.h5', mode='r') as store:
        by = ['model', 'SAT', 'population_size', 'run']
        columns = ['exposed', 'infectious', 'chronic']
        extinction = {}
        for (ix, group) in store.groupby(by, columns=columns):
            infected = group.sum(axis='columns')
            extinction[ix] = _get_extinction(infected)
        extinction = pandas.DataFrame.from_dict(extinction,
                                                orient='index')
        extinction.index.names = by
        extinction.sort_index(inplace=True)
        return extinction


def load_extinction_times():
    try:
        df = h5.load('plot_population_sizes.h5')
    except OSError:
        df = _load_extinction_times()
        h5.dump(df, 'plot_population_sizes.h5')
    return df


class PercentFormatter(ticker.Formatter):
    def __call__(self, x, pos=None):
        return '{:g}%'.format(100 * x)


def plot_survival(df):
    row = dict(enumerate(range(3), 1))
    column = {'acute': 0, 'chronic': 1}
    fig, axes = pyplot.subplots(3, 2, sharex='col', sharey='row')
    for ((model, SAT), group) in df.groupby(['model', 'SAT']):
        i, j = row[SAT], column[model]
        ax = axes[i, j]
        for (p, g) in group.groupby('population_size'):
            survival = stats.get_survival(g,
                                          'extinction_time',
                                          'extinction_observed')
            ax.step(survival.index, survival,
                    where='post', label=f'population size {p}')


def plot_kde(df):
    row = dict(enumerate(range(3), 1))
    column = {'acute': 0, 'chronic': 1}
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 2, sharex='col')
        for ((model, SAT), group) in df.groupby(['model', 'SAT']):
            i, j = row[SAT], column[model]
            ax = axes[i, j]
            for (p, g) in group.groupby('population_size'):
                ser = g.extinction_time[g.extinction_observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    x = kde.support
                    y = proportion_observed * kde.density
                else:
                    x, y = [], []
                label = p if i == j == 0 else ''
                ax.plot(x, y, label=label, alpha=0.7)
            ax.yaxis.set_major_locator(ticker.NullLocator())
            if ax.is_first_row():
                ax.set_title(f'{model.capitalize()} model',
                             fontdict=dict(fontsize='medium'))
            if ax.is_last_row():
                ax.set_xlim(left=0)
                ax.set_xlabel('extinction time (y)')
            if ax.is_first_col():
                ylabel = 'density' if i == 1 else ''
                ax.set_ylabel(f'SAT {SAT}\n{ylabel}')
        leg = fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5),
                         handletextpad=3, title='Population size')
        for text in leg.get_texts():
            text.set_horizontalalignment('right')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def _get_cmap(color):
    '''White to `color`.'''
    return colors.LinearSegmentedColormap.from_list('name',
                                                    ['white', color])


def plot_kde_2d(df):
    persistence_time_max = {'acute': 0.5, 'chronic': 5}
    SAT_colors = {1: '#2271b5', 2: '#ef3b2c', 3: '#807dba'}
    population_sizes = (df.index
                          .get_level_values('population_size')
                          .unique()
                          .sort_values())
    population_size_baseline = 1000
    fig, axes = pyplot.subplots(3, 2 + 1, sharex='col', sharey='row',
                                gridspec_kw=dict(width_ratios=(1, 1, 0.5)))
    for (j, (model, group_model)) in enumerate(df.groupby('model')):
        persistence_time = numpy.linspace(0, persistence_time_max[model], 301)
        for (i, (SAT, group_SAT)) in enumerate(group_model.groupby('SAT')):
            ax = axes[i, j]
            density = numpy.zeros((len(population_sizes),
                                   len(persistence_time)))
            proportion_observed = numpy.zeros_like(population_sizes,
                                                   dtype=float)
            for (k, (p, g)) in enumerate(group_SAT.groupby('population_size')):
                ser = g.extinction_time[g.extinction_observed]
                proportion_observed[k] = len(ser) / len(g)
                if proportion_observed[k] > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    density[k] = kde.evaluate(persistence_time)
                else:
                    density[k] = 0
            cmap = _get_cmap(SAT_colors[SAT])
            # Use raw `density` for color,
            # but plot `density * proportion_observed`.
            norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
            ax.imshow(density * proportion_observed[:, None],
                      cmap=cmap, norm=norm, interpolation='bilinear',
                      extent=(min(persistence_time), max(persistence_time),
                              min(population_sizes), max(population_sizes)),
                      aspect='auto', origin='lower', clip_on=False)
            ax.autoscale(tight=True)
            if model == 'chronic':
                ax_po = axes[i, -1]
                ax_po.plot(1 - proportion_observed, population_sizes,
                           color=SAT_colors[SAT], clip_on=False, zorder=3)
                ax_po.autoscale(tight=True)
                if ax.is_last_row():
                    ax_po.set_xlabel('persisting 10 y')
                    ax_po.xaxis.set_major_formatter(PercentFormatter())
                    ax_po.xaxis.set_minor_locator(
                        ticker.AutoMinorLocator(2))
            if ax.is_last_row():
                ax.set_xlabel('extinction time (y)')
                ax.xaxis.set_major_locator(
                    ticker.MultipleLocator(max(persistence_time) / 5))
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            if ax.is_first_col():
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.LogFormatter())
                ax.yaxis.set_minor_formatter(ticker.LogFormatter())
                ax.set_ylabel('population size')
                ax.annotate(f'SAT {SAT}',
                            (-0.5, 0.5), xycoords='axes fraction',
                            rotation=90, verticalalignment='center')
    for ax in fig.axes:
        ax.axhline(population_size_baseline,
                   color='black', linestyle='dotted', alpha=0.7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    title_y = 0.975
    fig.text(0.31, title_y, 'Acute model',
             horizontalalignment='center')
    fig.text(0.73, title_y, 'Chronic model',
             horizontalalignment='center')
    fig.tight_layout()
    fig.savefig('plot_population_sizes.pdf')


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
