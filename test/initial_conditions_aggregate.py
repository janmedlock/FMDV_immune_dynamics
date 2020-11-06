#!/usr/bin/python3

import itertools
import sys

from matplotlib import pyplot
import numpy
import pandas
from scipy import integrate

sys.path.append('..')
from herd import initial_conditions, mortality, Parameters, utility
sys.path.pop()


def get_params():
    return {SAT: Parameters(SAT)
            for SAT in (1, 2, 3)}


def get_solutions(params):
    ages = utility.arange(0, 20, 0.005, endpoint=True)
    p = {}
    for (SAT, params_SAT) in params.items():
        ICs = initial_conditions.gen(params_SAT)
        p[SAT] = ICs.immune_status_pdf(ages)
    return pandas.concat(p, axis='columns')


def integrate_over_age(X):
    '''Integrate over age for each SAT & immune status.'''
    ages = X.index
    v = X.apply(integrate.trapz, args=(ages, ))
    if isinstance(v.index, pandas.MultiIndex):
        v = v.unstack(0)
    return v


def sum_over_immune_state(X):
    return X.sum(axis='columns', level=0)


def plot_integral_over_age(p):
    P = integrate_over_age(p)
    (fig, ax) = pyplot.subplots(constrained_layout=True)
    y = P / P.sum()
    x = range(len(y.columns))
    bottom = numpy.zeros(len(y.columns))
    for (immune_status, val) in y.iterrows():
        ax.bar(x, val, bottom=bottom, label=immune_status)
        bottom += val
    ax.set_xticks(x)
    ax.set_xticklabels(f'SAT {SAT}' for SAT in y.columns)
    ax.set_ylabel('Proportion of population')
    (handles, labels) = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])


def plot_sum_over_immune_state(p):
    ages = p.index
    P = sum_over_immune_state(p)
    ax = P.plot(alpha=0.6)
    ax.set_xlabel('age')
    ax.set_ylabel('survival')
    survival = mortality.from_param_values().sf(ages)
    ax.plot(ages, survival, color='black', linestyle='dashed', alpha=0.6)
    return ax


def reorder_for_lr(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def plot_probability_constant_birth(p):
    '''With the assumption of constant-time birth hazard,
    plot probability of being in each class vs. age,
    *not* conditioned on being alive.'''
    ages = p.index
    # Group by SAT
    grouper = p.groupby(axis='columns', level=0)
    (fig, axes) = pyplot.subplots(len(grouper),
                                  sharex=True)
    for (ax, (SAT, group)) in zip(axes, grouper):
        collection = ax.stackplot(ages, group.T, labels=p.columns)
        ax.set_ylabel(f'SAT{SAT}\nconstant-birth\ndensity')
        ax.margins(0)
    axes[-1].set_xlabel('age')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    immune_statuses = p.columns.levels[1]
    nrow = 2
    ncol = (len(immune_statuses) + nrow - 1) // nrow
    fig.legend(reorder_for_lr(collection, ncol),
               reorder_for_lr(immune_statuses, ncol),
               ncol=ncol, loc='lower center')
    return fig


def plot_conditional_probability(p):
    '''Plot probability of being in each class vs. age,
    conditioned on being alive.'''
    ages = p.index
    # Group by SAT
    grouper = p.groupby(axis='columns', level=0)
    (fig, axes) = pyplot.subplots(len(grouper),
                                  sharex=True)
    for (ax, (SAT, group)) in zip(axes, grouper):
        Y = group.divide(sum_over_immune_state(group),
                         axis='columns', level=0)
        collection = ax.stackplot(ages, Y.T, labels=p.columns)
        ax.set_ylabel(f'SAT{SAT}\nprobability\ngiven alive')
        ax.margins(0)
    axes[-1].set_xlabel('age')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    immune_statuses = p.columns.levels[1]
    nrow = 2
    ncol = (len(immune_statuses) + nrow - 1) // nrow
    fig.legend(reorder_for_lr(collection, ncol),
               reorder_for_lr(immune_statuses, ncol),
               ncol=ncol, loc='lower center')
    return fig


def get_hazard_infection(p, params):
    '''Compute the hazard of infection from the solutions.'''
    P = integrate_over_age(p)
    transmission_rate = pandas.Series({k: v.transmission_rate
                                       for (k, v) in params.items()})
    chronic_transmission_rate = pandas.Series({k: v.chronic_transmission_rate
                                               for (k, v) in params.items()})
    return (transmission_rate * P.loc['infectious']
            + chronic_transmission_rate * P.loc['chronic'])


def get_incidence(p, params):
    hazard = get_hazard_infection(p, params)
    P = integrate_over_age(p)
    # The fraction susceptible to infection.
    S = P.loc[['susceptible', 'lost immunity']].sum()
    return hazard * S


if __name__ == '__main__':
    params = get_params()
    p = get_solutions(params)
    # plot_integral_over_age(p)
    # plot_sum_over_immune_state(p)
    plot_probability_constant_birth(p)
    plot_conditional_probability(p)
    pyplot.show()
