#!/usr/bin/python3
'''Analyze and plot the results of varying both the population size
and the susceptibility of the lost-immunity state. This requires the
file `population_size_and_susceptibility.h5`, which is built by
`population_size_and_susceptibility_run.py`.'''

import matplotlib.colors
import matplotlib.pyplot
import matplotlib.scale
import matplotlib.ticker
import seaborn

import common
import population_size
import population_size_and_susceptibility
import supplemental_materials
import susceptibility


rc = common.rc | supplemental_materials.rc | common.rc_text_small | {
    'figure.figsize': (3.5, 6.5),
    'pcolor.shading': 'gouraud',
    'contour.algorithm': 'threaded',
    'contour.linewidth': 1,
    'axes.spines.right': False,
    'axes.spines.top': False,
}

population_size_label = population_size.label.replace('\n', ' ')

susceptibility_label = susceptibility.label.replace('\n', ' ') \
                                           .replace('of ', 'of\n')

persistence_label = (
    f'Proportion persisting {common.TIME_MAX} {common.TIME_UNIT}s'
)


def load_extinction_time():
    return common.load_extinction_time(
        population_size_and_susceptibility.store_path)


def get_persistence(dfr):
    grouper = dfr.groupby(['SAT',
                           susceptibility.var,
                           population_size.var])
    return grouper.apply(common.get_persistence)


def fill_missing_persistence(dfr):
    assert common.is_increasing(dfr.columns, strict=True)
    # Starting from the left, where there is a missing value, if the
    # value in the previous column is 1, set the current value to 1.
    # Skip the first column since it has no previous column.
    for (col_prev, col_curr) in zip(dfr.columns[:-1], dfr.columns[1:]):
        to_update = (dfr[col_curr].isnull()
                     & (dfr[col_prev] == 1))
        dfr.loc[to_update, col_curr] = 1


_LogitNorm = matplotlib.colors.make_norm_from_scale(
    matplotlib.scale.LogitScale
)(
    matplotlib.colors.Normalize
)
_LogitNorm.__name__ = _LogitNorm.__qualname__ = 'LogitNorm'
_LogitNorm.__doc__ = 'Logit norm.'


def prepend_to_text(s, text):
    text.set_text(s + text.get_text())


def plot_persistence(dfr, save=True):
    contour_levels = [0.01, 0.5, 0.99]
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        persistence = get_persistence(dfr)
        grouper = persistence.groupby('SAT')
        nrows = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(
            nrows=nrows,
            sharex='col',
        )
        for ((SAT, group), ax) in zip(grouper, axes):
            # Move population_size from an index level to columns.
            arr = group.unstack()
            fill_missing_persistence(arr)
            x = arr.columns
            y = arr.index.droplevel('SAT')
            cmap = common.get_cmap_SAT(SAT)
            epsilon = 0.01
            vmin = 0 + epsilon
            vmax = 1 - epsilon
            norm = _LogitNorm(vmin=vmin, vmax=vmax, clip=True)
            img = ax.pcolormesh(x, y, arr,
                                cmap=cmap, norm=norm)
            contours = ax.contour(x, y, arr, contour_levels,
                                  colors='black')
            contours.clabel(inline=True,
                            fmt=lambda x: f'{100*x:g}%')
            ax.axvline(population_size.default,
                       color='black', linestyle='dotted', alpha=0.7,
                       clip_on=False)
            ax.axhline(susceptibility.default,
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
                ax.set_xlabel(population_size_label)
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.LogFormatter()
                )
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(susceptibility_label)
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
                label=persistence_label,
                format=matplotlib.ticker.PercentFormatter(xmax=1),
            )
            cbar.outline.set_edgecolor('none')
            cbar.ax.spines['right'].set_visible(True)
            cbar.minorformatter = matplotlib.ticker.NullFormatter()
            cticklabels = cbar.long_axis.get_ticklabels()
            prepend_to_text('≤ ', cticklabels[0])
            prepend_to_text('≥ ', cticklabels[-1])
            cbar.set_ticks(
                cbar.get_ticks(), labels=cticklabels
            )
            cbar.long_axis.labelpad = 0
        fig.align_ylabels()
        if save:
            store_path = population_size_and_susceptibility.store_path
            fig.savefig(store_path.with_suffix('.pdf'))
        return fig


if __name__ == '__main__':
    dfr = load_extinction_time()
    plot_persistence(dfr)
    matplotlib.pyplot.show()
