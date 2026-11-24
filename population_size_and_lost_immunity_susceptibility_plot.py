#!/usr/bin/env python
'''Analyze and plot the results of varying both the population size
and the susceptibility of the lost-immunity state. This requires the
file `population_size_and_lost_immunity_susceptibility.h5`, which is
built by `population_size_and_lost_immunity_susceptibility_run.py`.'''

import matplotlib.pyplot
import matplotlib.ticker
import pandas
import seaborn

import common
import herd.utility
import lost_immunity_susceptibility
import plotting
import population_size
import population_size_and_lost_immunity_susceptibility
import sensitivity

rc = (
    plotting.rc
    | plotting.SupplementalMaterials.rc
    | plotting.rc_text_small
    | {
        'figure.figsize': (3.2, 6),
        'pcolor.shading': 'gouraud',
        'contour.algorithm': 'threaded',
        'contour.linewidth': 1,
        'axes.spines.right': False,
        'axes.spines.top': False,
    }
)


def _fill_missing_persistence(persistence):
    assert herd.utility.is_increasing(persistence.columns, strict=True)
    # Starting from the left, where there is a missing value, if the
    # value in the previous column is 1, set the current value to 1.
    # Check back 2 columns if the previous one was the default. Skip
    # the first column since it has no previous column.
    for (i, row) in enumerate(persistence.index):
        for (j, col) in enumerate(persistence.columns[1:], 1):
            if (
                    pandas.isna(persistence.loc[row, col])
                    & (
                        (
                            persistence.iloc[i, j-1] == 1
                        ) | (
                            (
                                j > 1
                            )
                            & (
                                persistence.columns[j-1]
                                == population_size.default
                            ) & (
                                persistence.iloc[i, j-2] == 1
                            )
                        )
                    )
            ):
                persistence.loc[row, col] = 1


def _prepend_to_text(s, text):
    text.set_text(s + text.get_text())


def plot_persistence(extinction_time, save=True, show=False):
    contour_levels = [0.01, 0.5, 0.99]
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        grouper = extinction_time.groupby('SAT')
        nrows = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(
            nrows=nrows, sharex='col',
        )
        for ((SAT, group), ax) in zip(grouper, axes):
            # `persistence` has index 'lost_immunity_susceptibility'
            # and columns 'population_size'.
            persistence = (
                group.groupby(['lost_immunity_susceptibility',
                               'population_size'])
                .apply(common.get_persistence)
                .unstack()
            )
            _fill_missing_persistence(persistence)
            cmap = plotting.get_cmap_SAT(SAT)
            epsilon = 0.01
            vmin = 0 + epsilon
            vmax = 1 - epsilon
            norm = plotting.LogitNorm(vmin=vmin, vmax=vmax, clip=True)
            img = ax.pcolormesh(persistence.columns,
                                persistence.index,
                                persistence,
                                cmap=cmap, norm=norm)
            contours = ax.contour(persistence.columns,
                                  persistence.index,
                                  persistence,
                                  contour_levels,
                                  colors='black')
            contours.clabel(inline=True,
                            fmt=lambda x: f'{100*x:g}%')
            ax.axvline(population_size.default,
                       color='black', linestyle='dotted', alpha=0.7,
                       clip_on=False)
            ax.axhline(lost_immunity_susceptibility.default,
                       color='black', linestyle='dotted', alpha=0.7,
                       clip_on=False)
            ax.annotate(f'SAT{SAT}',
                        (-0.4, 0.5), xycoords='axes fraction',
                        fontsize=rc['axes.titlesize'],
                        rotation='vertical',
                        verticalalignment='center')
            ax.margins(0)
            subplotspec = ax.get_subplotspec()
            if subplotspec.is_last_row():
                ax.set_xscale('log')
                ax.set_xlabel(
                    population_size.label
                )
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.LogFormatter()
                )
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(
                    lost_immunity_susceptibility.label
                    .replace('of ', 'of\n')
                )
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(0.2)
                )
                ax.yaxis.set_minor_locator(
                    matplotlib.ticker.AutoMinorLocator(2)
                )
            cbar = fig.colorbar(
                img,
                ax=ax,
                location='right',
                label=plotting.PERSISTENCE_LABEL,
                format=matplotlib.ticker.PercentFormatter(xmax=1),
            )
            cbar.outline.set_edgecolor('none')
            cbar.ax.spines['right'].set_visible(True)
            cbar.minorformatter = matplotlib.ticker.NullFormatter()
            cticklabels = cbar.long_axis.get_ticklabels()
            _prepend_to_text('≤ ', cticklabels[0])
            _prepend_to_text('≥ ', cticklabels[-1])
            cbar.set_ticks(
                cbar.get_ticks(), labels=cticklabels
            )
            cbar.long_axis.labelpad = 0
        fig.align_ylabels()
        if save:
            output_path = (
                population_size_and_lost_immunity_susceptibility.store_path
                .with_suffix('.pdf')
            )
            plotting.savefig(fig, output_path)
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    extinction_time = sensitivity.load(
        population_size_and_lost_immunity_susceptibility)
    plot_persistence(extinction_time)
