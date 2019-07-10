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


def plot_kde_2d(df):
    xmax = {'acute': 0.5, 'chronic': 5}
    y = df.index.get_level_values('population_size').unique().sort_values()
    fig, axes = pyplot.subplots(3, 2, sharex='col', sharey='row')
    for (j, (model, group_model)) in enumerate(df.groupby('model')):
        x = numpy.linspace(0, xmax[model], 301)
        for (i, (SAT, group_SAT)) in enumerate(group_model.groupby('SAT')):
            ax = axes[i, j]
            Z = numpy.zeros((len(y), len(x)))
            vmax = 0
            for (k, (p, g)) in enumerate(group_SAT.groupby('population_size')):
                ser = g.extinction_time[g.extinction_observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    Z[k] = kde.evaluate(x)
                    vmax = max(vmax, max(Z[k]))
                    Z[k] *= proportion_observed
                else:
                    Z[k] = 0
            norm = colors.Normalize(vmin=0, vmax=vmax)
            ax.imshow(Z, interpolation='bilinear', cmap='Purples',
                      norm=norm, aspect='auto', origin='lower',
                      extent=(min(x), max(x), min(y), max(y)))
            if ax.is_first_row():
                ax.set_title(f'{model.capitalize()} model',
                             fontdict=dict(fontsize='medium'))
            if ax.is_last_row():
                ax.xaxis.set_major_locator(ticker.MultipleLocator(max(x) / 5))
                ax.set_xlabel('extinction time (y)')
            if ax.is_first_col():
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.LogFormatter())
                ax.yaxis.set_minor_formatter(ticker.LogFormatter())
                ax.set_ylabel('population size')
            if ax.is_last_col():
                ax.annotate(f'SAT {SAT}',
                            (1.02, 0.5), xycoords='axes fraction',
                            rotation=90, verticalalignment='center')
    fig.tight_layout()


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
