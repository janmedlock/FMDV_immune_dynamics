#!/usr/bin/python3
'''Plot variation between SATs.'''

import pathlib

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import seaborn

from context import data as data_
from context import plotting
import estimate
import plotting_


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 4.5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid.axis': 'y',
    'axes.grid.which': 'both',
    'ytick.minor.ndivs': 2,
    'ytick.minor.visible': True,
}


def plot_rates(axs):
    '''Plot the rates.'''
    log_rate = estimate.estimate_by_sat()
    plotting_.rates_by_sat_on(log_rate, axs, alpha=plotting.ALPHA)


def plot_seropositives(ax):
    '''Plot the antibody titers by SAT for seropositives.'''
    seropositives = data_.load_seropositives()
    sats = seropositives.SAT.unique()
    seaborn.violinplot(
        seropositives, x='SAT', y='titer', hue='SAT',
        palette=plotting.SAT_COLORS, alpha=plotting.ALPHA, saturation=1,
        linewidth=0, cut=0, inner=None, legend=False, ax=ax,
    )
    mean = seropositives.groupby('SAT')['titer'].mean()
    x = ax.get_xticks()
    s = 20 ** 2
    ax.scatter(x, mean, color='black', marker='_', s=s)
    ax.set_xticks(x, [f'SAT{sat}' for sat in sats])
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(r'log$_{10}$ antibody titer')
    ax.yaxis.set_major_locator(
        matplotlib.ticker.MultipleLocator(0.2)
    )


AGE_BREAKS = [0, 1, 2, 3, 6, 11, numpy.inf]


def plot_seronegative_by_age(axs):
    '''Plot the proportion seronegative by age.'''
    data_with_age = data_.load_with_age()
    age_bins = pandas.IntervalIndex.from_breaks(
        AGE_BREAKS, closed='left', name='age (y)'
    )
    ages = pandas.cut(data_with_age['age (y)'], bins=age_bins)

    def get_interval_label(interval):
        right = interval.right - 1
        if interval.left == right:
            return f'{interval.left:g}'
        if numpy.isinf(interval.right):
            return f'{interval.left:g}–'
        return f'{interval.left:g}–{right:g}'

    age_labels = list(map(get_interval_label, age_bins))
    grouper = (
        pandas.concat([data_with_age[['SAT', 'negative']], ages],
                      axis='columns')
        .groupby(['SAT', 'age (y)'],
                 observed=True)
    )
    seronegative = (
        grouper['negative'].sum() / grouper.size()
    )
    y_max = seronegative.max()
    grouper = seronegative.groupby('SAT')
    for ((sat, ser), ax) in zip(grouper, axs):
        ax.bar(age_labels, ser,
               color=plotting.SAT_COLORS[sat], alpha=plotting.ALPHA)
        (_, y_margin) = ax.margins()
        ax.set_ylim(0, (1 + y_margin) * y_max)
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel(ages.name)
        ax.set_ylabel(f'SAT{sat} seronegative')
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1)
        )
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.2)
        )


def plot_variation(save=True, show=True):
    '''Make the figure showing variation between SATs.'''
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axs) = matplotlib.pyplot.subplots(nrows=2, ncols=3)
        plot_rates(axs[0, :2])
        plot_seropositives(axs[0, 2])
        plot_seronegative_by_age(axs[1, :])
        plotting.add_part_labels(axs, pad=13)
        fig.align_labels()
        if save:
            path_file = pathlib.Path(__file__)
            path = path_file.with_stem(path_file.stem.replace('_plot', ''))
            for suffix in ('.pdf', '.png'):
                fig.savefig(path.with_suffix(suffix))
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    plot_variation()
