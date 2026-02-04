#!/usr/bin/python3
'''Plot examples of categories of titer dynamics.'''

import pathlib

import matplotlib.dates
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas

from context import plotting
import data_
from titers_plot import load


_TICK_LABELSIZE = 5
rc = (
    plotting.rc
    | plotting.SupplementalMaterials.rc
    | {
        'figure.figsize': (1, 1),
        'xtick.labelsize': _TICK_LABELSIZE,
        'ytick.labelsize': _TICK_LABELSIZE,
        'axes.labelsize': 5,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.minor.ndivs': 2,
        'xtick.minor.visible': True,
        'ytick.minor.ndivs': 2,
        'ytick.minor.visible': True,
        'lines.linewidth': 0.5,
        'axes.labelpad': 0,
        'figure.constrained_layout.h_pad': 0,
        'figure.constrained_layout.w_pad': 0,
    }
)


EXAMPLES = {
    'negative': {'ID': 119, 'SAT': 2},
    'positive': {'ID': 11, 'SAT': 2},
    'converted': {'ID': 5, 'SAT': 1},
    'fluctuating': {'ID': 5, 'SAT': 3},
}


def _plot_example(name, group, save):
    (fig, ax) = matplotlib.pyplot.subplots()
    ax.plot('date', 'titer', data=group,
            color='black', marker='o', markersize=1)
    ax.axhline(data_.ANTIBODY_TITER_CUTOFF,
               color='0.3', linestyle='dotted', zorder=1.5)
    ax.set_xlim(pandas.Timestamp('2014-01-01'), None)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.set_ylim(1, 3)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.set_ylabel(r'log$_{10}$ antibody titer')
    if save:
        source_path = pathlib.Path(__file__)
        output_path_stem = source_path.with_stem(
            source_path.stem.replace('_plot', f'_{name}')
        )
        for suffix in ('.pdf', '.png'):
            output_path = output_path_stem.with_suffix(suffix)
            plotting.savefig(fig, output_path)


def plot_examples(data, save=True, show=False):
    '''Plot the examples.'''
    with matplotlib.pyplot.rc_context(rc=rc):
        for (name, which) in EXAMPLES.items():
            keep = numpy.all(
                [data[key] == val for (key, val) in which.items()],
                axis=0,
            )
            _plot_example(name, data[keep], save)
        if show:
            matplotlib.pyplot.show()



if __name__ == '__main__':
    data__ = load()
    plot_examples(data__)
