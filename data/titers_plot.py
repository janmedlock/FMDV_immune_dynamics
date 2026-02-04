#!/usr/bin/python3
'''Plot titer time-series by SAT and animal.'''

import functools
import math
import pathlib

import matplotlib.dates
import matplotlib.pyplot
import pandas
import seaborn

from context import plotting
import data_


_CONSEC_OBS_MIN = 6

_TICK_LABELSIZE = 6
rc = (
    plotting.rc
    | plotting.SupplementalMaterials.rc
    | {
        'figure.figsize': (plotting.SupplementalMaterials.WIDTH_MAXIMUM, 6),
        'font.size': 5,
        'xtick.labelsize': _TICK_LABELSIZE,
        'ytick.labelsize': _TICK_LABELSIZE,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.minor.ndivs': 2,
        'xtick.minor.visible': True,
        'ytick.minor.ndivs': 2,
        'ytick.minor.visible': True,
        'axes.titlepad': 0,
        'lines.linewidth': 0.5,
    }
)


@functools.lru_cache(maxsize=1)
def load():
    '''Load the data.'''
    observations = data_.load_observations()
    keep = observations.consecutive_observations >= _CONSEC_OBS_MIN
    keep_ids = observations[keep].index
    data = data_.load()
    return data[data['ID'].isin(keep_ids)]


_COL_WRAP = 7


def _set_ylabel(fg):
    fg.set_ylabels('')
    # Find the axis in the middle row, first column.
    nrows = math.ceil(len(fg.axes) / _COL_WRAP)
    row = nrows // 2
    ax = fg.facet_axis(0, row * _COL_WRAP)
    ax.set_ylabel(r'log$_{10}$ antibody titer')


def _plot_sat(sat, group, save):
    fg = seaborn.FacetGrid(group,
                           col='ID', col_wrap=_COL_WRAP,
                           ylim=(1, 3))
    fg.map(matplotlib.pyplot.plot, 'date', 'titer',
           color='black', marker='o', markersize=1)
    fg.refline(y=data_.ANTIBODY_TITER_CUTOFF,
               color='0.3', linestyle='dotted', zorder=1.5)
    fg.set_xlabels('')
    _set_ylabel(fg)
    fg.set_titles('{col_var} {col_name}')
    fg.set(xlim=(pandas.Timestamp('2014-01-01'), None))
    for ax in fg.axes:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    fg.figure.set_size_inches(*rc['figure.figsize'])
    fg.figure.set_layout_engine('constrained')
    if save:
        source_path = pathlib.Path(__file__)
        output_path_stem = source_path.with_stem(
            source_path.stem.replace('_plot', f'_sat{sat}')
        )
        output_path = output_path_stem.with_suffix('.pdf')
        plotting.savefig(fg.figure, output_path)


def plot_titers(data, save=True, show=False):
    '''Plot the titers.'''
    grouper = data.groupby('SAT')
    # Avoid a warning by disabling constrained layout here
    # and re-enabling in `_plot_sat()`.
    rc_ = rc | {'figure.constrained_layout.use': False}
    with matplotlib.pyplot.rc_context(rc=rc_):
        for (sat, group) in grouper:
            _plot_sat(sat, group, save)
        if show:
            matplotlib.pyplot.show()


if __name__ == '__main__':
    data__ = load()
    plot_titers(data__)
