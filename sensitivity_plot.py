#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size and susceptibility.  This requires the files
`population_size.h5` and `susceptibility.h5`.'''

import pathlib

import matplotlib.pyplot
import numpy

import common
import population_size
import sensitivity
import susceptibility


rc = common.rc | common.rc_text_small | {
    'figure.figsize': (common.WIDTH_MAXIMUM['double_column'], 4),
    'axes.titlesize': 11,
}


MODULES = (population_size, susceptibility)


def load():
    return [
        sensitivity.load_extinction_time(module)
        for module in MODULES
    ]


def plot_persistence(dfs, save=True, show=True):
    with matplotlib.pyplot.rc_context(rc=rc):
        fig = None
        ncols = len(dfs)
        for (col, (df, module)) in enumerate(zip(dfs, MODULES)):
            grouper_SAT = df.groupby('SAT')
            if fig is None:
                nrows = len(grouper_SAT)
                (fig, axes) = matplotlib.pyplot.subplots(
                    nrows, ncols,
                    sharex='col', sharey='row',
                )
            vals = df.index \
                     .get_level_values(module.var) \
                     .unique() \
                     .sort_values()
            for ((SAT, group_SAT), ax) in zip(grouper_SAT, axes[:, col]):
                proportion_observed = sensitivity.get_proportion_observed(
                    group_SAT, module.var
                )
                ax.plot(1 - proportion_observed,
                        color=common.SAT_colors[SAT],
                        clip_on=False, zorder=3)
                subplotspec = ax.get_subplotspec()
                if subplotspec.is_last_row():
                    ax.set_xlim(min(vals), max(vals))
                    ax.set_xlabel(module.label.replace('\n', ' '))
                if subplotspec.is_first_col():
                    ax.annotate(f'SAT{SAT}',
                                (-0.275, 0.5), xycoords='axes fraction',
                                fontsize=rc['axes.titlesize'],
                                rotation='vertical',
                                verticalalignment='center')
                    ax.set_ylabel('Persisting 10 y')
                    ax.set_ylim(0, 1)
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.PercentFormatter(xmax=1)
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
                ax.axvline(module.default,
                           color='black', linestyle='dotted', alpha=0.7,
                           clip_on=False)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        if save:
            source_path = pathlib.Path(__file__)
            output_path_stem = source_path.with_name(
                source_path.name.replace('_plot.py', '')
            )
            fig.savefig(output_path_stem.with_suffix('.pdf'))
            fig.savefig(output_path_stem.with_suffix('.png'), dpi=300)
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    dfs = load()
    plot_persistence(dfs)
