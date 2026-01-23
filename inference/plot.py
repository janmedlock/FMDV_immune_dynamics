'''Plotting.'''

import matplotlib.pyplot
import numpy
import seaborn


# Erin's colors by SAT.
COLORS = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba',
}


def _waiting_time(ax, sat, waiting_time_sat, y):
    waiting_time_sat_err = [
        waiting_time_sat.MLE - waiting_time_sat.CI_upper,
        waiting_time_sat.CI_lower - waiting_time_sat.MLE
    ]
    ax.errorbar(waiting_time_sat.MLE, y, xerr=waiting_time_sat_err,
                label=f'SAT{sat}',
                linestyle='None', marker='|',
                markersize=5, markeredgewidth=2,
                markerfacecolor='black', markeredgecolor='black',
                ecolor=COLORS[sat], elinewidth=5)


def waiting_times_by_sat(log_rate, dist=0.1, show=True):
    '''Plot the waiting times.'''
    rate = numpy.exp(log_rate)
    waiting_time = 1 / rate
    waiting_time.name = 'mean waiting time (d)'
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
