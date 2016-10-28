#!/usr/bin/python3

import csv
import itertools

from matplotlib import pyplot
from matplotlib import ticker
import numpy
import pandas
import seaborn
from scipy import integrate

seaborn.set_context('talk')


def load_extinction_data():
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


def make_SIR_plots(show = True):
    data = load_SIR_data()

    (fig, axes) = pyplot.subplots(5, 3, sharex = 'all', sharey = 'row')

    for (j, SAT) in enumerate((1, 2, 3)):
        colors = itertools.cycle(seaborn.color_palette('husl', 8))

        reps = data.index.levels[0]

        for r in reps:
            x = data.loc[r]
            t = x.index
            if r != 'mean':
                c = next(colors)
                alpha = 0.5
            else:
                c = 'black'
                alpha = 0.6
            for i in range(axes.shape[0]):
                axes[i, j].step(365 * t, x.iloc[:, i], where = 'post',
                                color = c, alpha = 0.5)

                axes[i, j].xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins = 4))
                axes[i, j].yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins = 4))

            axes[0, j].set_title('SAT {}'.format(SAT))

    ax = axes[:, 0]
    ax[0].set_ylabel('maternal\nimmunity')
    ax[1].set_ylabel('susceptible')
    ax[2].set_ylabel('exposed')
    ax[3].set_ylabel('infected')
    ax[4].set_ylabel('recovered')

    mid = (axes.shape[-1] - 1) // 2
    axes[-1, mid].set_xlabel('time (days)')

    for ax in axes[:, 0]:
        yl = ax.get_ylim()
        if yl[0] < 0:
            ax.set_ylim(ymin = 0)

    if show:
        pyplot.show()


def make_extinction_histos(population_size = 1000,
                           birth_seasonal_coefficient_of_variation = 0.61,
                           show = True):
    data = load_extinction_data()
    
    data_ = data.loc(axis = 0)[:,
                               population_size,
                               birth_seasonal_coefficient_of_variation]

    SATs = data_.index.get_level_values('SAT')

    (fig, axes) = pyplot.subplots(1, len(SATs),
                                  sharex = 'all', sharey = 'row')

    colors = itertools.cycle(seaborn.color_palette('deep'))

    for (j, SAT) in enumerate(SATs):
        seaborn.distplot(365 * data_.loc[SAT],
                         ax = axes[j],
                         color = next(colors),
                         kde = False)
        axes[j].set_title('SAT {}'.format(SAT))

    mid = (axes.shape[-1] - 1) // 2
    axes[mid].set_xlabel('time (days)')

    if show:
        pyplot.show()


def make_heatmaps(show = True):
    data = load_extinction_data()
    
    if show:
        pyplot.show()


if __name__ == '__main__':
    # make_SIR_plots()
    make_extinction_histos()
    # make_heatmaps()
