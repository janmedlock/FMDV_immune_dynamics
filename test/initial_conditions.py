#!/usr/bin/env python
'''Plot the model initial condititions.'''

import matplotlib.pyplot
import numpy

from _context import common, herd, plotting
import herd.initial_conditions
import herd.utility

AGE_MAX = 20
AGE_STEP = 0.05


def plot_probability_constant_birth(ics, ages, ax):
    '''With the assumption of constant-time birth hazard,
    plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    p = ics.immune_status_pdf(ages)
    ax.stackplot(ages, p.T, labels=p.columns)
    ax.set_ylim(0, 1)
    ax.set_ylabel('constant-birth\ndensity')


def plot_conditional_probability(ics, ages, ax):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    p = ics.immune_status_conditional_pdf(ages)
    ax.stackplot(ages, p.T)
    ax.set_ylabel('probability\ngiven alive')


def plot_probability(ics, ages, ax):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    p = ics.pdf(ages)
    ax.stackplot(ages, p.T)
    ax.set_ylabel('density')


def plot_samples(ics, ages, ax):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    samples = ics.rvs()
    bins = ages
    left = bins[:-1]
    width = ages[1] - ages[0]
    bottom = numpy.zeros_like(left)
    grouper = samples.index.groupby(samples)
    for ages_in_immune_status in grouper.values():
        (height, _) = numpy.histogram(ages_in_immune_status, bins=bins)
        ax.bar(left, height, width, bottom, align='edge')
        bottom += height
    ax.set_ylabel('count in\nsample')


def plot_ics(SAT, seed=1):
    parameters = herd.Parameters(SAT=SAT)
    ics = herd.initial_conditions.InitialConditions(parameters, seed=seed)
    ages = herd.utility.arange(0, AGE_MAX, AGE_STEP, endpoint=True)
    plot_fcns = (plot_probability_constant_birth,
                 plot_conditional_probability,
                 plot_probability,
                 plot_samples)
    (fig, axes) = matplotlib.pyplot.subplots(len(plot_fcns), 1,
                                             sharex=True)
    for (ax, plot_fcn) in zip(axes, plot_fcns):
        plot_fcn(ics, ages, ax)
    axes[-1].set_xlabel(f'age ({common.TIME_UNIT})')
    for ax in axes:
        ax.margins(0)
    fig.align_ylabels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (handles, labels) = axes[0].get_legend_handles_labels()
    nrow = 3
    ncol = (len(labels) + nrow - 1) // nrow
    plotting.legend_multicolumn(fig, handles, labels, ncol,
                                loc='lower center')


if __name__ == '__main__':
    for SAT in common.SATs:
        plot_ics(SAT)
    matplotlib.pyplot.show()
