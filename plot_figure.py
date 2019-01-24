#!/usr/bin/python3
#
# TODO
# * Label alignment.
# * Add/plan for model diagram.
# * Consider tweaking height ratios.
# * Consider changing width ratios from 1:1 to
#   show difference in time scales between columns.
# * Consider plotting fewer runs.

from matplotlib import gridspec
from matplotlib import pyplot
from matplotlib import ticker
import numpy
import seaborn
import statsmodels
import pandas

import plot_start_times_SATs


# Erin's colors.
SAT_colors = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}

# Nature.
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
width = 183 / 25.4  # inches
height = 1.2 * width
rc['figure.figsize'] = [width, height]
# Sans-serif, preferably Helvetica or Arial.
rc['font.family'] = 'sans-serif'
rc['font.sans-serif'] = 'DejaVu Sans'
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 7
rc['axes.labelsize'] = 6
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 5
# Separate panels in multi-part figures should be labelled with 8
# pt bold, upright (not italic) a, b, c...


def load():
    infected = []
    extinction_time = []
    for chronic in (False, True):
        i = plot_start_times_SATs.get_infected(chronic=chronic)
        i['chronic'] = chronic
        infected.append(i)
        e = plot_start_times_SATs.get_extinction_time(chronic=chronic)
        e['chronic'] = chronic
        extinction_time.append(e)
    infected = pandas.concat(infected)
    extinction_time = pandas.concat(extinction_time)
    return (infected, extinction_time)


def plot_infected(ax, infected, SAT, chronic):
    mask = ((infected.SAT == SAT)
            & (infected.chronic == chronic))
    i = infected[mask].infected.unstack('run')
    t = i.index - i.index.min()
    ax.plot(365 * t, i, alpha=0.25)
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(365 * t, i.fillna(0).mean(axis=1), color='black', alpha=1)
    ax.xaxis.set_tick_params(which='both',
                             labelbottom=False, labeltop=False)
    ax.xaxis.offsetText.set_visible(False)
    if ax.is_first_col():
        ax.set_ylabel('Number\ninfected')
        ax.annotate(f'SAT {SAT}', (-0.18, 0.2),
                    xycoords='axes fraction',
                    rotation=90, fontsize=rc['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.is_first_row():
        ax.set_title(('Chronic' if chronic else 'Acute') + ' model',
                     loc='center')


def plot_extinction_time(ax, extinction_time, SAT, chronic):
    col = 'extinction time (days)'
    mask = ((extinction_time.SAT == SAT)
            & (extinction_time.chronic == chronic))
    e = extinction_time[col][mask]
    color = SAT_colors[SAT]
    if len(e.dropna()) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(e)
        kde.fit(gridsize=100, cut=0)
        x = kde.support
        y = kde.density
    else:
        # No extinctions: make line at y=0.
        x = (0, extinction_time[col].max())
        y = (0, 0)
    ax.plot(x, y, color=color)
    ax.fill_between(x, 0, y, facecolor=color, alpha=0.25)
    not_extinct = len(e[e.isnull()]) / len(e)
    if not_extinct > 0:
        # 0.6 -> 0.3, 1 -> 1.
        pad = (1 - 0.3) / (1 - 0.6) * (not_extinct - 0.6) + 0.3
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    (0.98, 0.8), xycoords='axes fraction',
                    bbox=bbox, color='white',
                    verticalalignment='bottom',
                    horizontalalignment='right')
    # No y ticks.
    ax.yaxis.set_major_locator(ticker.NullLocator())
    if ax.is_first_col():
        ax.set_ylabel('Extinction\ntime')
    if ax.is_last_row():
        ax.set_xlabel('Time (d)')
    else:
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)


def plot(infected, extinction_time):
    SATs = infected.SAT.unique()
    chronics = infected.chronic.unique()
    nrows = len(SATs) * 2
    ncols = len(chronics)
    height_ratios = (1, 0.2) * (nrows // 2)
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                               height_ratios=height_ratios)
        axes = numpy.empty((nrows, ncols), dtype=object)
        for row in range(nrows):
            for col in range(ncols):
                # Columns share the x scale.
                sharex = axes[0, col]
                # The infection plots share the y scale.
                # The extinction-time plots do *not* share the y scale.
                sharey = axes[0, 0] if (row % 2 == 0) else None
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=sharex,
                                                 sharey=sharey)
        for (i, SAT) in enumerate(SATs):
            for (col, chronic) in enumerate(chronics):
                ax_i = axes[2 * i, col]
                ax_e = axes[2 * i + 1, col]
                plot_infected(ax_i, infected, SAT, chronic)
                plot_extinction_time(ax_e, extinction_time, SAT, chronic)
        for row in range(nrows):
            for col in range(ncols):
                axes[row, col].set_xlim(left=0)
                axes[row, col].set_ylim(bottom=0)
        seaborn.despine(fig=fig, top=True, right=True, bottom=False, left=False)
        fig.align_labels()
        fig.savefig('plot_figure.pdf')


if __name__ == '__main__':
    infected, extinction_time = load()
    plot(infected, extinction_time)
    # pyplot.show()
