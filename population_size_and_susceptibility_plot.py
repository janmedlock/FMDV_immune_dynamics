#!/usr/bin/python3

import matplotlib.colors
import matplotlib.pyplot
import matplotlib.scale
import matplotlib.ticker

import common
import population_size
import population_size_and_susceptibility
import susceptibility


population_size_label = population_size.label.replace('\n', ' ')

susceptibility_label = susceptibility.label.replace('\n', ' ') \
                                           .replace('of ', 'of\n')

persistence_label = f'Proportion persisting {common.TMAX} y'


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
    rc = common.rc.copy()
    width = 183 / 25.4  # convert mm to in
    height = 4  # in
    rc['figure.figsize'] = (width, height)
    # Between 5pt and 7pt.
    rc['font.size'] = 6
    rc['axes.titlesize'] = 9
    rc['axes.labelsize'] = 8
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
    with matplotlib.pyplot.rc_context(rc=rc):
        persistence = get_persistence(dfr)
        grouper = persistence.groupby('SAT')
        ncols = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(
            2, ncols,
            sharey='row', layout='constrained',
            gridspec_kw=dict(hspace=0.15,
                             height_ratios=(10, 1)))
        for ((SAT, group), (ax, ax_cbar)) in zip(grouper, axes.T):
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
                                cmap=cmap, norm=norm,
                                shading='gouraud')
            contour_levels = [0.01, 0.5, 0.99]
            contours = ax.contour(x, y, arr, contour_levels,
                                  algorithm='threaded',
                                  colors='black', linewidths=1)
            pct_fmt = lambda x: f'{100*x:g}%'
            contour_fontsize = matplotlib.pyplot.rcParams['xtick.labelsize']
            contours.clabel(inline=True,
                            fmt=pct_fmt,
                            fontsize=contour_fontsize)
            ax.axvline(population_size.default,
                       color='black', linestyle='dotted', alpha=0.7,
                       clip_on=False)
            ax.axhline(susceptibility.default,
                       color='black', linestyle='dotted', alpha=0.7,
                       clip_on=False)
            ax.set_title(f'SAT{SAT}')
            ax.set_xscale('log')
            ax.set_xlim(min(population_size.values),
                        max(population_size.values))
            ax.set_xlabel(population_size_label)
            ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatter())
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylim(min(susceptibility.values),
                            max(susceptibility.values))
                ax.set_ylabel(susceptibility_label)
            cbar = fig.colorbar(
                img,
                cax=ax_cbar,
                orientation='horizontal',
                label=persistence_label,
                format=matplotlib.ticker.PercentFormatter(xmax=1))
            ax_cbar.tick_params(which='minor', labelbottom=False)
            cticklabels = ax_cbar.get_xticklabels()
            prepend_to_text('≤', cticklabels[0])
            prepend_to_text('≥', cticklabels[-1])
            ax_cbar.set_xticks(ax_cbar.get_xticks(), cticklabels)
        fig.align_xlabels()
        if save:
            store_path = population_size_and_susceptibility.store_path
            fig.savefig(store_path.with_suffix('.pdf'))
            fig.savefig(store_path.with_suffix('.png'), dpi=300)
        return fig


if __name__ == '__main__':
    dfr = load_extinction_time()
    plot_persistence(dfr)
    matplotlib.pyplot.show()
