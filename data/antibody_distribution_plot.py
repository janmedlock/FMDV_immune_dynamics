#!/usr/bin/python3
'''Plot the distribution of antibodies.'''

import pathlib

import matplotlib.collections
import matplotlib.pyplot
import seaborn

from context import plotting
import data_


rc = (
    plotting.rc
    | plotting.SupplementalMaterials.rc
    | plotting.rc_text_small
    | {
        'figure.figsize': (plotting.SupplementalMaterials.WIDTH_MAXIMUM, 3),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid.axis': 'y',
        'axes.grid.which': 'both',
        'ytick.minor.ndivs': 2,
        'ytick.minor.visible': True,
    }
)


def _remove_violin_borders(ax):
    '''Remove violin borders. Assumes violins are instances of
    `matplotlib.collections.FillBetweenPolyCollection()`, and that
    there are no non-violin instances of that type.'''
    for collection in ax.collections:
        if isinstance(collection,
                      matplotlib.collections.FillBetweenPolyCollection):
            collection.set_linewidth(0)


def plot_antibody_distribution(data, save=True, show=False):
    '''Plot the antibody titers by SAT.'''
    sats = data.SAT.unique()
    xticklabels = [f'SAT{sat}' for sat in sats]
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, ax) = matplotlib.pyplot.subplots()
        seaborn.violinplot(
            data, x='SAT', y='titer', hue='SAT',
            palette=plotting.SAT_COLORS, alpha=plotting.ALPHA, saturation=1,
            inner='box', inner_kws={'whis_width': 0, 'solid_capstyle': 'butt'},
            cut=0, legend=False, ax=ax,
        )
        _remove_violin_borders(ax)
        ax.set_xticks(ax.get_xticks(), xticklabels)
        ax.xaxis.label.set_visible(False)
        ax.set_ylabel(r'log$_{10}$ antibody titer')
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.2)
        )
        ax.axhline(
            data_.ANTIBODY_TITER_CUTOFF,
            linestyle='dotted', color='black', zorder=0.9,
        )
        if save:
            source_path = pathlib.Path(__file__)
            output_path_stem = source_path.with_stem(
                source_path.stem.removesuffix('_plot')
            )
            output_path = output_path_stem.with_suffix('.pdf')
            plotting.savefig(fig, output_path)
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    data__ = data_.load()
    plot_antibody_distribution(data__)
