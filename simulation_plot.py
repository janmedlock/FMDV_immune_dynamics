#!/usr/bin/python3
'''Plot one simulation for each SAT.'''

import math

import matplotlib.pyplot
import matplotlib.ticker
import seaborn

import h5
import plotting
import simulation


rc = (
    plotting.rc
    | plotting.SupplementalMaterials.rc
    | plotting.rc_text_small
    | {
        'figure.figsize': (plotting.SupplementalMaterials.WIDTH_MAXIMUM, 7),
        'axes.spines.top': False,
    }
)


def load():
    return h5.load(simulation.store_path, 'simulation')


def plot_one(ax, SAT, group):
    time = group.index.get_level_values('time')
    t = time - time.min()
    for (name, ser) in group.items():
        ax.plot(t, ser, label=plotting.get_state_label(name),
                drawstyle='steps-pre',
                alpha=0.9, linewidth=1, zorder=3)
    ax.annotate(f'SAT{SAT}',
                (-0.1625, 0.5), xycoords='axes fraction',
                fontsize=rc['axes.titlesize'],
                rotation='vertical',
                verticalalignment='center')
    ax.set_ylabel('Number')
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(plotting.TIME_LABEL)
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
        plotting.legend_multicolumn(ax, handles, labels, ncol,
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
        if save:
            fig.savefig(simulation.store_path.with_suffix('.pdf'))
    return fig


if __name__ == '__main__':
    data = load()
    plot_SATs(data)
    matplotlib.pyplot.show()
