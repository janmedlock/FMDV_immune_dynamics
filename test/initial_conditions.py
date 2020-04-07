#!/usr/bin/python3

import sys

import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters
sys.path.pop()


def plot_conditional_probability(ax, conditional_probability):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    ages = conditional_probability.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(conditional_probability.items()):
        total += v
        z = conditional_probability.shape[1] - i
        ax.fill_between(ages, total, label=k, zorder=z)
    ax.set_ylabel('probability\ngiven age')


def plot_probability(ax, probability):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    ages = probability.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(probability.items()):
        total += v
        z = len(probability) - i
        ax.fill_between(ages, total, label=None, zorder=z)
    ax.set_ylabel('joint\ndensity')


def plot_sample(ax, sample, width=0.1):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    age_max = max(max(ages) for ages in sample.values()) + width
    left = numpy.arange(0, age_max, width)
    bins = numpy.hstack((left, age_max))
    bottom = numpy.zeros_like(left)
    for (_, ages) in sample.items():
        height, _ = numpy.histogram(ages, bins=bins)
        ax.bar(left, height, width, bottom, label=None, align='edge')
        bottom += height
    ax.set_ylabel('number in\nsample')


def plot(parameters, ages):
    from matplotlib import pyplot
    ICs = initial_conditions.gen(parameters)
    fig, axes = pyplot.subplots(3, 1, sharex=True)
    conditional_probability = ICs.immune_status_probability_interpolant(ages)
    plot_conditional_probability(axes[0], conditional_probability)
    probability = ICs.pdf(ages)
    plot_probability(axes[1], probability)
    numpy.random.seed(1)  # Make `ICs.rvs()` cache friendly.
    sample = ICs.rvs()
    plot_sample(axes[2], sample)
    axes[-1].set_xlabel('age')
    axes[0].set_xlim(ages.min(), ages.max())
    for ax in axes:
        ax.margins(0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (_, labels) = axes[0].get_legend_handles_labels()
    nrow = 2
    ncol = (len(labels) + nrow - 1) // nrow
    fig.legend(loc='lower center', ncol=ncol)
    pyplot.show()


if __name__ == '__main__':
    parameters = Parameters(SAT=1)
    ages = numpy.linspace(0, 20, 101)
    plot(parameters, ages)
