#!/usr/bin/python3
'''Plot variation between SATs.'''

import pathlib

import matplotlib.pyplot
import matplotlib.ticker
import seaborn

from context import data
from context import plotting
import estimate
import plotting_


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 4.5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid.axis': 'y',
}


def plot_seropositives(antibodies_, ax=None, alpha=plotting_.ALPHA):
    '''Plot the antibody titers by SAT for seropositives.'''
    if ax is None:
        ax = matplotlib.pyplot.gca()
    seropositives = antibodies_[antibodies_['positive']]
    sats = seropositives.SAT.unique()
    seaborn.violinplot(
        seropositives, x='SAT', y='titer', hue='SAT',
        palette=plotting.SAT_COLORS, alpha=alpha, saturation=1,
        linewidth=0, cut=0, inner=None, legend=False, ax=ax,
    )
    mean = seropositives.groupby('SAT')['titer'].mean()
    x = ax.get_xticks()
    s = 20 ** 2
    ax.scatter(x, mean, color='black', marker='_', s=s)
    ax.set_xticks(x, [f'SAT{sat}' for sat in sats])
    ax.xaxis.label.set_visible(False)
    ax.set_ylabel(r'log$_{10}$ antibody titer')


def plot_variation(antibodies_, log_rate_, save=True, show=True):
    '''Make the figure showing variation between SATs.'''
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axs) = matplotlib.pyplot.subplots(nrows=2, ncols=3)
        plotting_.rates_by_sat_on(log_rate_, axs[0, :2])
        plot_seropositives(antibodies_, axs[0, 2])
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
    antibodies = data.load()
    log_rate = estimate.estimate_by_sat()
    plot_variation(antibodies, log_rate, show=False)
