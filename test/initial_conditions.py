#!/usr/bin/python3


import sys

import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters
sys.path.pop()


def plot_prob_cond(ax, status_prob_cond):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    ages = status_prob_cond.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(status_prob_cond.items()):
        total += v
        z = status_prob_cond.shape[1] - i
        ax.fill_between(ages, total, label=k, zorder=z)
    ax.set_ylabel('probability\ngiven age')
    ax.set_ylim(-0.05, 1.05)


def plot_prob(ax, status_prob):
    '''Plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    ages = status_prob.index
    total = numpy.zeros_like(ages)
    for (i, (k, v)) in enumerate(status_prob.items()):
        total += v
        z = len(status_prob) - i
        ax.fill_between(ages, total, label=None, zorder=z)
    ax.set_ylabel('joint\ndensity')


def plot_sample(ax, status_ages, width=0.1):
    '''For a sample initial condition,
    plot the number in each class vs. age.'''
    age_max = max(max(ages) for ages in status_ages.values()) + width
    left = numpy.arange(0, age_max, width)
    bins = numpy.hstack((left, age_max))
    bottom = numpy.zeros_like(left)
    for (_, ages) in status_ages.items():
        height, _ = numpy.histogram(ages, bins=bins)
        ax.bar(left, height, width, bottom, label=None, align='edge')
        bottom += height
    ax.set_ylabel('number in\nsample')


# TODO: Remove this.
def get_status_prob_cond(parameters, ages):
    sys.path.append('..')
    from herd.initial_conditions import status
    sys.path.pop()
    hazard_infection = 1
    return status.probability(ages, hazard_infection, parameters)


# TODO: Remove this.
def get_status_prob(parameters, ages, status_prob_cond):
    sys.path.append('..')
    from herd import age_structure
    sys.path.pop()
    age_prob = age_structure.gen(parameters).pdf(ages)
    return status_prob_cond.mul(age_prob, axis='index')


# TODO: Remove this.
def get_status_ages(parameters):
    sys.path.append('..')
    from herd import age_structure
    from herd.initial_conditions import status
    sys.path.pop()
    from scipy.stats import multinomial
    hazard_infection = 1
    # Pick `parameters.population_size` random ages.
    ages = age_structure.gen(parameters).rvs(size=parameters.population_size)
    # Determine the status for each age.
    status_probability =  status.probability(ages,
                                             hazard_infection,
                                             parameters)
    statuses = status_probability.columns
    status_ages = {k: [] for k in statuses}
    # `scipy.stats.multinomial.rvs()` can't handle multiple `p`s,
    # so we need to loop.
    for (age, row) in status_probability.iterrows():
        # Randomly pick a status.
        rv = multinomial.rvs(1, row)
        # `rv` is an array with `1` in the position
        # picked and `0`s in the remaining positions.
        # Convert that to the name.
        s = statuses[rv == 1][0]
        # Add this `age` to the status list.
        status_ages[s].append(age)
    return status_ages


def plot(parameters, ages):
    from matplotlib import pyplot
    # ICs = initial_conditions.gen(parameters)
    fig, axes = pyplot.subplots(3, 1, sharex=True)
    # status_prob_cond = ICs._status_probability(ages)
    status_prob_cond = get_status_prob_cond(parameters, ages)
    plot_prob_cond(axes[0], status_prob_cond)
    # status_prob = ICs.pdf(ages)
    status_prob = get_status_prob(parameters, ages, status_prob_cond)
    plot_prob(axes[1], status_prob)
    numpy.random.seed(1)  # Make `ICs.rvs()` cache friendly.
    # status_ages = ICs.rvs(parameters.population_size)
    status_ages = get_status_ages(parameters)
    plot_sample(axes[2], status_ages)
    axes[-1].set_xlabel('age', labelpad=-9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    (_, labels) = axes[0].get_legend_handles_labels()
    nrow = 2
    ncol = (len(labels) + nrow - 1) // nrow
    fig.legend(loc='lower center', ncol=ncol)
    pyplot.show()


if __name__ == '__main__':
    parameters = Parameters()
    ages = numpy.linspace(0, 20, 101)
    plot(parameters, ages)
