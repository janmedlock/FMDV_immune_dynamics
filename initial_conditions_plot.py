#!/usr/bin/python3
'''Plot the model initial condititions.'''

import math
import pathlib

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import seaborn

import common
import herd
import herd.initial_conditions
import herd.utility
import supplemental_materials


rc = common.rc | supplemental_materials.rc | common.rc_text_small | {
    'figure.figsize': (supplemental_materials.WIDTH_MAXIMUM, 6),
}


def plot_age_density(ICs, ages, ax):
    '''Plot the density of the stable age distribution.'''
    p = ICs.ages.pdf(ages)
    ax.plot(ages, p, color='black')
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Age density (year$^{-1}$)')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.1)
        )


def plot_status_probability_unconditional(ICs, ages, ax):
    '''Plot probability of being in each status *not* conditioned on
    being alive vs. age.'''
    p = ICs.immune_status_pdf(ages)
    ax.stackplot(ages, p.T, labels=p.columns.str.capitalize())
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Status probability')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.2)
        )


def plot_status_probability(ICs, ages, ax):
    '''Plot the probability of being in each status conditioned on
    being alive vs. age.'''
    p = ICs.immune_status_conditional_pdf(ages)
    ax.stackplot(ages, p.T, labels=p.columns.str.capitalize())
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Status probability')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.2)
        )


def plot_joint_density(ICs, ages, ax):
    '''Plot the joint density.'''
    p = ICs.pdf(ages)
    ax.stackplot(ages, p.T, labels=p.columns.str.capitalize())
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Joint density (year$^{-1}$)')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(0.1)
        )


plot_fcns = (
    plot_age_density,
    plot_status_probability,
    plot_joint_density,
)


def plot_SAT(axes, SAT, ages):
    parameters = herd.Parameters(SAT=SAT)
    ICs = herd.initial_conditions.gen(parameters)
    for (ax, plot_fcn) in zip(axes, plot_fcns):
        plot_fcn(ICs, ages, ax)
    for ax in axes:
        ax.margins(x=0)
        ax.grid(which='minor', visible=True)
        ax.grid(which='both', clip_on=False)
        subplotspec = ax.get_subplotspec()
        if subplotspec.is_first_row():
            ax.set_title(f'SAT{SAT}')
        # is next-to-last row.
        gridspec = subplotspec.get_gridspec()
        if subplotspec.rowspan.stop == gridspec.nrows - 1:
            ax.set_xlabel('Age (year)')
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(5)
            )
            ax.xaxis.set_minor_locator(
                matplotlib.ticker.MultipleLocator(1)
            )
        else:
            ax.xaxis.set_tick_params(which='both',
                                     labelleft=False, labelright=False)
            ax.xaxis.offsetText.set_visible(False)
        if subplotspec.is_first_col():
            ax.yaxis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(2)
            )
        else:
            ax.yaxis.set_tick_params(which='both',
                                     labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def plot_SATs(save=True):
    ages = herd.utility.arange(0, 20, 0.1, endpoint=True)
    with seaborn.axes_style('whitegrid'), matplotlib.rc_context(rc=rc):
        fig = matplotlib.pyplot.figure()
        nrows = len(plot_fcns)
        ncols = len(common.SATs)
        # Add an extra row for the legend.
        gs = fig.add_gridspec(
            nrows + 1, ncols,
            height_ratios=((1,) * nrows + (0.3,)),
        )
        axes = numpy.empty((nrows, ncols), dtype=object)
        axes[0, 0] = None  # `sharex` & `sharey` for `axes[0, 0]`.
        for col in range(ncols):
            for row in range(nrows):
                axes[row, col] = fig.add_subplot(
                    gs[row, col],
                    sharex=axes[0, col],
                    sharey=axes[row, 0],
                )
        for (col, SAT) in enumerate(common.SATs):
            plot_SAT(axes[:, col], SAT, ages)
        fig.align_ylabels()
        (handles, labels) = axes[1, 0].get_legend_handles_labels()
        nrow = 2
        ncol = math.ceil(len(handles) / nrow)
        common.legend_multicolumn(fig, handles, labels, ncol,
                                  loc='lower right',
                                  bbox_to_anchor=(0.95, 0))
        seaborn.despine(fig)
    if save:
        source_path = pathlib.Path(__file__)
        output_path_stem = source_path.with_name(
            source_path.name.replace('_plot.py', '')
        )
        fig.savefig(output_path_stem.with_suffix('.pdf'))
    return fig


if __name__ == '__main__':
    plot_SATs()
    matplotlib.pyplot.show()
