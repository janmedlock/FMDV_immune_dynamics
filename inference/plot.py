'''Plotting.'''

import matplotlib.pyplot
import numpy
import seaborn

import _ci
import estimate
from context import plotting


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 3),
}

RATE_LABEL = 'rate (y$^{-1}$)'


def _rate(ax, rate_):
    (sats, (parameter,)) = (
        rate_.index
        .remove_unused_levels()
        .levels
    )
    colors = [plotting.SAT_COLORS[sat] for sat in sats]
    ax.scatter(
        sats, rate_.MLE,
        c=colors,
        marker='_', s=100,
    )
    err = (
        rate_.MLE - rate_.CI_lower,
        rate_.CI_upper - rate_.MLE
    )
    ax.errorbar(
        sats, rate_.MLE, yerr=err,
        linestyle='None', marker='None',
        ecolor=colors,
    )
    ax.set_xticks(sats, [f'SAT{sat}' for sat in sats])
    ax.set_xlim(sats.min() - 0.5, sats.max() + 0.5)
    ax.set_ylim(bottom=0)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(RATE_LABEL)
    title = (
        parameter.replace('annual rate', '')
        .replace('-', ' ')
    )
    ax.set_title(title)


def rates_by_sat(log_rate, show=True):
    '''Plot the rates.'''
    rates = estimate.to_annual_rate(log_rate)
    grouper = rates.groupby('parameter', observed=True)
    with seaborn.axes_style('darkgrid'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axs) = matplotlib.pyplot.subplots(ncols=len(grouper),
                                                sharey='row')
        for ((_, rate_), ax) in zip(grouper, axs):
            _rate(ax, rate_)
        seaborn.despine(fig)
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
