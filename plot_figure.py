#!/usr/bin/python3
#
# TODO
# * Label alignment.
# * Check width of figure with diagram.

from matplotlib import gridspec
from matplotlib import pyplot
from matplotlib import ticker
import numpy
import seaborn

import plot_common
import plot_start_times_SATs


# From `pdfinfo notes/diagram_standalone.pdf'.
diagram_width = 184.763 / 72  # inches
diagram_height = 279.456 / 72  # inches

# Nature.
rc = {}
# Widths: 89mm, 183mm, 120mm, 136mm.
total_width = 183 / 25.4  # inches
fig_width = total_width - diagram_width
fig_height = 6  # inches
rc['figure.figsize'] = [fig_width, fig_height]
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
# I'm gonna try to avoid this.


def load():
    infected = plot_start_times_SATs.get_infected()
    extinction_time = plot_start_times_SATs.get_extinction_time()
    return (infected, extinction_time)


def plot_infected(ax, infected, SAT):
    nruns = 1000
    ix = (SAT, slice(None), slice(0, nruns))
    # Remove all indices except 'run' and 'time'.
    to_drop = infected.index.names[:-2]
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[ix].unstack('run').reset_index(to_drop, drop=True)
    # Start time at 0.
    t = i.index - i.index.min()
    ax.plot(365 * t, i, color=plot_common.SAT_colors[SAT],
            alpha=0.15, linewidth=0.5, drawstyle='steps-pre')
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(365 * t, i.fillna(0).mean(axis='columns'), color='black',
            alpha=1)
    # Tighten y-axis limits.
    ax.margins(y=0)
    # Shared x-axis with extinction time.
    ax.xaxis.set_tick_params(which='both',
                             labelbottom=False, labeltop=False)
    ax.xaxis.offsetText.set_visible(False)
    if ax.is_first_col():
        ax.set_ylabel('Number\ninfected')
        ax.annotate(f'SAT {SAT}', (-0.35, 0.2),
                    xycoords='axes fraction',
                    rotation=90, fontsize=rc['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)


def plot_extinction_time(ax, extinction_time, SAT):
    e = extinction_time.loc[SAT]
    color = plot_common.SAT_colors[SAT]
    if len(e.dropna()) > 0:
        seaborn.kdeplot(e.dropna(), ax=ax, color=color,
                        shade=True, legend=False, cut=0)
    not_extinct = len(e[e.isnull()]) / len(e)
    if not_extinct > 0:
        # 0.6 -> 0.3, 1 -> 1.
        pad = (1 - 0.3) / (1 - 0.6) * (not_extinct - 0.6) + 0.3
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    (0.95, 0.8), xycoords='axes fraction',
                    bbox=bbox, color='white',
                    verticalalignment='bottom',
                    horizontalalignment='right')
    # No y ticks.
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # Shared x-axes between SATs.
    if ax.is_last_row():
        ax.set_xlabel('Time (d)')
    else:
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
    if ax.is_first_col():
        ax.set_ylabel('Extinction\ntime')


def plot(infected, extinction_time):
    SATs = infected.index.get_level_values('SAT').unique()
    nrows = len(SATs) * 2
    height_ratios = (3, 1) * (nrows // 2)
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = gridspec.GridSpec(nrows, 1, figure=fig,
                               height_ratios=height_ratios)
        axes = numpy.empty(nrows, dtype=object)
        for row in range(nrows):
            # Columns share the x scale.
            sharex = axes[0] if (row > 0) else None
            # The infection plots share the y scale.
            # The extinction-time plots do *not* share the y scale.
            sharey = axes[0] if ((row > 0) and (row % 2 == 0)) else None
            axes[row] = fig.add_subplot(gs[row, 1],
                                        sharex=sharex,
                                        sharey=sharey)
        for (i, SAT) in enumerate(SATs):
            row_i = 2 * i
            row_e = 2 * i + 1
            plot_infected(axes[row_i], infected, SAT)
            plot_extinction_time(axes[row_e], extinction_time, SAT)
        # I get weird results if I set these limits individually.
        for row in range(nrows):
            axes[row].set_xlim(left=0)
            axes[row].set_ylim(bottom=0)
        seaborn.despine(fig=fig, top=True, right=True, bottom=False, left=False)
        fig.align_labels()
        fig.savefig('plot_figure.pdf')


if __name__ == '__main__':
    infected, extinction_time = load()
    plot(infected, extinction_time)
    pyplot.show()
