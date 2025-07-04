#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size and susceptibility.  This requires the files
`population_size.h5` and `susceptibility.h5`.'''

import pathlib

import matplotlib.pyplot

import common
import population_size
import sensitivity
import susceptibility


MODULES = (population_size, susceptibility)


def load():
    return [
        sensitivity.load_extinction_time(module)
        for module in MODULES
    ]


def plot_persistence(dfs, save=True):
    rc = common.rc.copy()
    width = 183 / 25.4  # convert mm to in
    height = 4  # in
    rc['figure.figsize'] = (width, height)
    # Between 5pt and 7pt.
    rc['font.size'] = 6
    rc['axes.titlesize'] = 11
    rc['axes.labelsize'] = 8
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
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
                    layout='constrained'
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
                if module.log:
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(
                        matplotlib.ticker.LogFormatter()
                    )
                subplotspec = ax.get_subplotspec()
                if subplotspec.is_last_row():
                    ax.set_xlim(min(vals), max(vals))
                    ax.set_xlabel(module.label.replace('\n', ' '))
                if subplotspec.is_first_col():
                    ax.annotate(f'SAT{SAT}',
                                (-0.25, 0.5), xycoords='axes fraction',
                                fontsize=rc['axes.titlesize'],
                                rotation='vertical',
                                verticalalignment='center')
                    ax.set_ylabel('persisting 10 y')
                    ax.set_ylim(0, 1)
                    ax.yaxis.set_major_formatter(
                        matplotlib.ticker.PercentFormatter(xmax=1)
                    )
                ax.axvline(module.default,
                           color='black', linestyle='dotted', alpha=0.7,
                           clip_on=False)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        if save:
            filepath = pathlib.Path(__file__)
            fig.savefig(filepath.with_suffix('.pdf'))
            fig.savefig(filepath.with_suffix('.png'),
                        dpi=300)
        return fig


if __name__ == '__main__':
    dfs = load()
    plot_persistence(dfs)
    matplotlib.pyplot.show()
