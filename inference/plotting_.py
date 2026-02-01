'''Plotting.'''

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import seaborn

from context import plotting
import estimate
import _ci


RATE_LABEL = 'rate (y$^{-1}$)'


def _rate(ax, rate_, rate_max, alpha, elinewidth=10):
    (sats, (parameter,)) = (
        rate_.index
        .remove_unused_levels()
        .levels
    )
    s = elinewidth ** 2
    zorder_scatter = 3
    ax.scatter(
        sats, rate_.MLE,
        marker='_', s=s, c='black', zorder=zorder_scatter,
    )
    err = (
        rate_.MLE - rate_.CI_lower,
        rate_.CI_upper - rate_.MLE
    )
    ecolor = [plotting.SAT_COLORS[sat] for sat in sats]
    zorder_errorbar = zorder_scatter - 1
    ax.errorbar(
        sats, rate_.MLE, yerr=err,
        elinewidth=elinewidth, ecolor=ecolor, alpha=alpha,
        linestyle='None', marker='None', zorder=zorder_errorbar,
    )
    ax.set_xticks(sats, [f'SAT{sat}' for sat in sats])
    ax.set_xlim(sats.min() - 0.5, sats.max() + 0.5)
    (_, y_margin) = ax.margins()
    ax.set_ylim(0, (1 + y_margin) * rate_max)
    ax.set_ylabel(parameter.replace('annual rate', RATE_LABEL))


def rates_by_sat_on(log_rate, axs, alpha=plotting.ALPHA):
    '''Plot the rates onto `axs`.'''
    rates = estimate.to_annual_rate(log_rate)
    rate_max = rates.max().max()
    grouper = rates.groupby('parameter', observed=True)
    for ((_, rate_), ax) in zip(grouper, axs):
        _rate(ax, rate_, rate_max, alpha)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(1)
        )


def rates_by_sat(log_rate, show=True):
    '''Plot the rates.'''
    parameters = log_rate.index.levels[
        log_rate.index.names.index('parameter')
    ]
    with seaborn.axes_style('darkgrid'):
        (fig, axs) = matplotlib.pyplot.subplots(ncols=len(parameters),
                                                layout='constrained')
        rates_by_sat_on(log_rate, axs)
        if show:
            matplotlib.pyplot.show()
        return fig


def _to_waiting_time(log_rate):
    rate = numpy.exp(log_rate)
    waiting_time = 1 / rate
    # Swap upper and lower CI.
    waiting_time[_ci.COLUMNS] = waiting_time[_ci.COLUMNS[::-1]]
    waiting_time.name = 'mean waiting time (d)'
    return waiting_time


def _waiting_time(ax, sat, waiting_time_sat, y):
    waiting_time_sat_err = (
        waiting_time_sat.MLE - waiting_time_sat.CI_lower,
        waiting_time_sat.CI_upper - waiting_time_sat.MLE
    )
    ax.errorbar(
        waiting_time_sat.MLE, y, xerr=waiting_time_sat_err,
        label=f'SAT{sat}',
        linestyle='None', marker='|',
        markersize=5, markeredgewidth=2,
        markerfacecolor='black', markeredgecolor='black',
        ecolor=plotting.SAT_COLORS[sat], elinewidth=5,
    )


def waiting_times_by_sat(log_rate, dist=0.1, show=True):
    '''Plot the waiting times.'''
    waiting_time = _to_waiting_time(log_rate)
    (sats, parameters) = waiting_time.index.levels
    n_sats = len(sats)
    n_parameters = len(parameters)
    grouper = waiting_time.groupby('SAT')
    (_, ax) = matplotlib.pyplot.subplots(constrained_layout=True)
    for (i, (sat, waiting_time_sat)) in enumerate(grouper):
        y = numpy.arange(n_parameters)[::-1] + dist * (n_sats - 1 - i)
        _waiting_time(ax, sat, waiting_time_sat, y)
    ax.set_yticks(numpy.arange(n_parameters) + dist * (n_sats - 1) / 2)
    ax.set_yticklabels(
        parameters.str.replace(' log rate', '')
        .str.replace('-', ' ')
        [::-1]
    )
    ax.set_xlim(left=0)
    ax.set_xlabel(waiting_time.name)
    ax.legend(loc='center right', labelspacing=1, frameon=True)
    seaborn.despine(ax=ax)
    if show:
        matplotlib.pyplot.show()
    return ax
