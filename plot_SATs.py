#!/usr/bin/python3

import csv
import itertools

from matplotlib import patches
from matplotlib import pyplot
from matplotlib import ticker
import numpy
import pandas
import seaborn
from scipy import integrate

import h5

seaborn.set_context('talk')


def load_SIR_data():
    return h5.load('run.h5')


def load_extinction_data():
    # FIXME: doesn't work with `model='chronic'`.
    def translate(row):
        def f(x):
            if x == '':
                return numpy.nan
            else:
                return float(x)
        return list(map(f, row))

    r = csv.reader(open('search_parameters.csv'))
    header = next(r)
    data = numpy.array([translate(row) for row in r])

    cols = ['SAT',
            'population_size',
            'birth_seasonal_coefficient_of_variation']

    ix = {c: header.index(c) for c in cols[1 : ]}

    vals = {}
    SAT = 0
    arrs = []
    for i in range(len(data)):
        val = data[i, 9]
        try:
            SAT = vals[val]
        except KeyError:
            SAT += 1
            vals[val] = SAT
        ps = int(data[i, ix['population_size']])
        bscov = data[i, ix['birth_seasonal_coefficient_of_variation']]
        arrs.append((SAT, ps, bscov))

    idx = pandas.MultiIndex.from_tuples(arrs, names = cols)
    data = pandas.DataFrame(data[:, len(header) - 1 : ], index = idx)
    data.sort_index(inplace = True)
    return data


def make_SIR_plots(show=True):
    data = load_SIR_data()

    (SATs, reps) = data.index.levels[:2]

    # Drop 'Total'
    nrows = len(data.columns)
    ncols = len(SATs)
    (fig, axes) = pyplot.subplots(nrows, ncols,
                                  sharex = 'all', sharey = 'row')
    for (j, SAT) in enumerate(SATs):
        colors = itertools.cycle(seaborn.color_palette('husl', 8))
        for rep in reps:
            x = data.loc(axis=0)[SAT, rep]
            t = x.index
            if rep != 'mean':
                c = next(colors)
                alpha = 0.5
            else:
                c = 'black'
                alpha = 0.6
            for i in range(nrows):
                axes[i, j].step(365 * t, x.iloc[:, i], where='post',
                                color=c, alpha=0.5)
                axes[i, j].xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=4))
                axes[i, j].yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=4))
            axes[0, j].set_title('SAT {}'.format(SAT))

    ax = axes[:, 0]
    ax[0].set_ylabel('maternal\nimmunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('exposed')
    ax[3].set_ylabel('infected')
    ax[4].set_ylabel('chronic')
    ax[5].set_ylabel('recovered')

    mid = (axes.shape[-1] - 1) // 2
    axes[-1, mid].set_xlabel('time (days)')

    for ax in axes[:, 0]:
        yl = ax.get_ylim()
        if yl[0] < 0:
            ax.set_ylim(bottom=0)

    filename = f'plot_SATs_SIR.pdf'
    fig.savefig(filename)
    if show:
        pyplot.show()


def make_extinction_hist(population_size=1000,
                         birth_seasonal_coefficient_of_variation=0.61,
                         show=True):
    data = load_extinction_data()
    data = data.loc(axis = 0)[:,
                              population_size,
                              birth_seasonal_coefficient_of_variation]

    SATs = data.index.levels[0]

    (fig, axes) = pyplot.subplots(1, len(SATs),
                                  sharex = 'all', sharey = 'row')
    colors = itertools.cycle(seaborn.color_palette('deep'))
    for (j, SAT) in enumerate(SATs):
        seaborn.distplot(365 * data.loc[SAT],
                         ax = axes[j],
                         color = next(colors),
                         kde = False)
        axes[j].set_title('SAT {}'.format(SAT))

    mid = (axes.shape[-1] - 1) // 2
    axes[mid].set_xlabel('time (days)')

    filename = f'plot_SATs_hist.pdf'
    fig.savefig(filename)
    if show:
        pyplot.show()


def make_full(show=True):
    data = load_extinction_data()

    palette = 'Set2'
    linewidth = 0.75

    data_ = []
    for (k, v) in data.iterrows():
        for (_, x) in v.items():
            data_.append(k + (365 * x, ))
    cols = list(data.index.names) + ['extinction_time']
    data_ = pandas.DataFrame(data_, columns = cols)

    SATs = data.index.levels[0]
    population_sizes = data.index.levels[1]
    covs = data.index.levels[2]

    fig = pyplot.figure()
    nrows = 1
    ncols = len(SATs)
    (fig, axes) = pyplot.subplots(nrows, ncols,
                                  sharex = 'all',
                                  sharey = 'all')
    for (SAT, ax) in zip(SATs, axes):
        seaborn.violinplot(data = data_[data_['SAT'] == SAT],
                           x = 'extinction_time',
                           y = 'birth_seasonal_coefficient_of_variation',
                           order = reversed(covs),
                           hue = 'population_size',
                           # hue_order = reversed(population_sizes),
                           orient = 'h',
                           palette = palette,
                           cut = 0,
                           linewidth = linewidth,
                           ax = ax)

        ax.set_title('SAT {}'.format(SAT),
                     fontdict = dict(fontsize = 'small'))

        ax.set_xlim(0, data_['extinction_time'].max())

        if SAT == SATs[len(SATs) // 2]:
            ax.set_xlabel('extinction time (days)')
        else:
            ax.set_xlabel('')

        leg = ax.get_legend()
        if ax.is_first_col():
            ax.set_ylabel('birth seasonal coefficient of variation')
            leg.set_title('population size')
        else:
            ax.set_ylabel('')
            leg.set_visible(False)

    fig.tight_layout()
    filename = f'plot_SATs_full.pdf'
    fig.savefig(filename)
    if show:
        pyplot.show()


if __name__ == '__main__':
    make_SIR_plots()
    # make_extinction_hist()
    # make_full()
