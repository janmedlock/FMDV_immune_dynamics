#!/usr/bin/python3

import sys

import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters, utility
sys.path.pop()


def plot_probability_constant_birth(ICs, ages, ax):
    '''With the assumption of constant-time birth hazard,
    plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    p = ICs.immune_status_pdf(ages)
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(p.items()):
        total += v
        z = len(p) - i
        ax.fill_between(ages, total, label=k, zorder=z)
    ax.set_ylabel('constant-birth\ndensity')


def plot_conditional_probability(ICs, ages, ax):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    p = ICs.immune_status_conditional_pdf(ages)
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(p.items()):
        total += v
        z = p.shape[1] - i
        ax.fill_between(ages, total, label=None, zorder=z)
    ax.set_ylabel('probability\ngiven alive')


def plot_probability(ICs, ages, ax):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    p = ICs.pdf(ages)
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(p.items()):
        total += v
        z = len(p) - i
        ax.fill_between(ages, total, label=None, zorder=z)
    ax.set_ylabel('density')


def plot_samples(ICs, ages, ax):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    numpy.random.seed(1)  # Make `ICs.rvs()` reproducible.
    samples = ICs.rvs()
    left = ages
    width = ages[-1] - ages[-2]
    bins = numpy.hstack((ages, ages[-1] + width))
    bottom = numpy.zeros_like(left)
    for (_, ages) in samples.items():
        height, _ = numpy.histogram(ages, bins=bins)
        ax.bar(left, height, width, bottom, label=None, align='edge')
        bottom += height
    ax.set_ylabel('count in\nsample')


def plot_ICs(parameters, ages):
    from matplotlib import pyplot
    ICs = initial_conditions.gen(parameters)
    (fig, axes) = pyplot.subplots(4, 1, sharex=True)
    plot_probability_constant_birth(ICs, ages, axes[0])
    plot_conditional_probability(ICs, ages, axes[1])
    plot_probability(ICs, ages, axes[2])
    plot_samples(ICs, ages, axes[3])
    axes[-1].set_xlabel('age')
    axes[0].set_xlim(ages.min(), ages.max())
    for ax in axes:
        ax.margins(0)
    fig.align_ylabels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (_, labels) = axes[0].get_legend_handles_labels()
    nrow = 2
    ncol = (len(labels) + nrow - 1) // nrow
    fig.legend(loc='lower center', ncol=ncol)
    pyplot.show()


if __name__ == '__main__':
    parameters = Parameters(SAT=1)
    ages = utility.arange(0, 35, 0.01, endpoint=True)
    plot_ICs(parameters, ages)
