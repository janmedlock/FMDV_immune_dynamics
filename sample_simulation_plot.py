#!/usr/bin/python3
'''Plot one simulation for each SAT.'''

import math
import pathlib

import matplotlib.pyplot
import matplotlib.ticker
import seaborn

import common
import h5
import sample_simulation
import supplemental_materials


rc = common.rc | supplemental_materials.rc | common.rc_text_small | {
    'figure.figsize': (supplemental_materials.WIDTH_MAXIMUM, 7),
}


def load():
    return h5.load(sample_simulation.store_path)


def plot_one(ax, SAT, group):
    time = group.index.get_level_values(common.t_name)
    t = time - time.min()
    for (name, ser) in group.items():
        ax.plot(t, ser, label=name.capitalize(),
                drawstyle='steps-pre',
                alpha=0.9, linewidth=1)
    ax.annotate(f'SAT{SAT}',
                (-0.1625, 0.5), xycoords='axes fraction',
                fontsize=rc['axes.titlesize'],
                rotation='vertical',
                verticalalignment='center')
    ax.set_ylabel('Number')
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(common.t_label)
        ax.margins(x=0)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(1)
        )
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(100)
        )
        (handles, labels) = ax.get_legend_handles_labels()
        nrow = 2
        ncol = math.ceil(len(handles) / nrow)
        common.legend_multicolumn(ax, handles, labels, ncol,
                                  loc='upper center',
                                  bbox_to_anchor=(0.5, -0.25))


def plot_SATs(data, save=True):
    grouper = data.groupby('SAT')
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axes) = matplotlib.pyplot.subplots(
            nrows=len(grouper),
            sharex=True, sharey=True,
        )
        for (ax, (SAT, group)) in zip(axes, grouper):
            plot_one(ax, SAT, group)
        seaborn.despine(fig)
        if save:
            fig.savefig(sample_simulation.store_path.with_suffix('.pdf'))
    return fig


if __name__ == '__main__':
    data = load()
    plot_SATs(data)
    matplotlib.pyplot.show()
