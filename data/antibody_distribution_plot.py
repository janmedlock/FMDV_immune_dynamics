#!/usr/bin/python3
'''Plot the distribution of antibodies.'''

import pathlib

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


def plot_antibody_distribution(data, save=True, show=True):
    '''Plot the antibody titers by SAT.'''
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, ax) = matplotlib.pyplot.subplots()
        sats = data.SAT.unique()
        seaborn.violinplot(
            data, x='SAT', y='titer', hue='SAT',
            palette=plotting.SAT_COLORS, alpha=plotting.ALPHA, saturation=1,
            cut=0,
            inner='box', inner_kws={'whis_width': 0, 'solid_capstyle': 'butt'},
            legend=False, ax=ax,
        )
        for collection in ax.collections:
            collection.set_linewidth(0)
        ax.set_xticks(ax.get_xticks(), [f'SAT{sat}' for sat in sats])
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
            path_file = pathlib.Path(__file__)
            path = path_file.with_stem(path_file.stem.replace('_plot', ''))
            fig.savefig(path.with_suffix('.pdf'))
        if show:
            matplotlib.pyplot.show()


if __name__ == '__main__':
    data__ = data_.load()
    plot_antibody_distribution(data__)
