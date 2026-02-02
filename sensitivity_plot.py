#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size and susceptibility.  This requires the files
`population_size.h5` and `susceptibility.h5`.'''

import pathlib

import matplotlib.pyplot
import numpy
import seaborn

import common
import plotting
import population_size
import sensitivity
import susceptibility


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 4),
}


PERSISTENCE_LABEL = (
    plotting.PERSISTENCE_LABEL
    .replace('FMDV ', 'FMDV\n')
)


MODULES = (population_size, susceptibility)


def load():
    return [
        sensitivity.load(module)
        for module in MODULES
    ]


def plot_persistence(extinction_times, save=True, show=False):
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axes) = (None, None)
        ncols = len(extinction_times)
        for (col, (extinction_time, module)) in enumerate(zip(extinction_times,
                                                              MODULES)):
            grouper = extinction_time.groupby('SAT')
            if fig is None:
                nrows = len(grouper)
                (fig, axes) = matplotlib.pyplot.subplots(
                    nrows, ncols,
                    sharex='col', sharey='row',
                )
            vals = extinction_time.index \
                                  .get_level_values(module.var) \
                                  .unique() \
                                  .sort_values()
            for ((SAT, group), ax) in zip(grouper, axes[:, col]):
                persistence = (
                    group.groupby(module.var)
                    .apply(common.get_persistence)
                )
                ax.plot(persistence,
                        color=plotting.SAT_COLORS[SAT],
                        clip_on=False, zorder=3)
                subplotspec = ax.get_subplotspec()
                if subplotspec.is_last_row():
                    ax.set_xlim(min(vals), max(vals))
                    ax.set_xlabel(module.label)
                if subplotspec.is_first_col():
                    ax.annotate(f'SAT{SAT}',
                                (-0.32, 0.5), xycoords='axes fraction',
                                fontsize=rc['axes.titlesize'],
                                rotation='vertical',
                                verticalalignment='center')
                    ax.set_ylabel(PERSISTENCE_LABEL)
                    ax.set_ylim(0, 1)
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.PercentFormatter(xmax=1)
                    )
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(0.2)
                    )
                    ax.yaxis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator(2)
                    )
                if module.log:
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.LogFormatter()
                    )
                    # Add the tick label at the maximum x.
                    # Add all of the minor-tick labels.
                    ax.xaxis.set_minor_formatter(
                        matplotlib.ticker.LogFormatter(
                            minor_thresholds=(numpy.inf, numpy.inf)
                        )
                    )
                    # Make all but the one at the max not visible.
                    (_, x_max) = ax.xaxis.get_data_interval()
                    for label in ax.xaxis.get_minorticklabels():
                        (x, _) = label.get_position()
                        if x < x_max:
                            label.set_visible(False)
                    # Align the minor-tick labels with the major-tick labels.
                    x_minor_pad = (
                        matplotlib.pyplot.rcParams['xtick.major.pad']
                        + matplotlib.pyplot.rcParams['xtick.major.size']
                        - matplotlib.pyplot.rcParams['xtick.minor.size']
                    )
                    ax.tick_params(axis='x', which='minor', pad=x_minor_pad)
                else:
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(0.2)
                    )
                    ax.xaxis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator(2)
                    )
                ax.axvline(module.default,
                           color='black', linestyle='dotted', alpha=0.7,
                           clip_on=False)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        plotting.add_part_labels(axes[0, :], pad=13)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        if save:
            source_path = pathlib.Path(__file__)
            output_path_stem = source_path.with_stem(
                source_path.stem.removesuffix('_plot')
            )
            for suffix in ('.pdf', '.png'):
                output_path = output_path_stem.with_suffix(suffix)
                plotting.savefig(fig, output_path)
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    extinction_times = load()
    plot_persistence(extinction_times)
