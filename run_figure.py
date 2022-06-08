#!/usr/bin/python3
'''Build a figure comparing the runs of the SATs. This requires the
file `run.h5`, which is built by `run.py`.'''

from matplotlib import pyplot, ticker
import numpy
import seaborn

import plot_common
import run


# Science
rc = plot_common.rc.copy()
width = 183 / 25.4  # convert mm to in
height = 3  # in
rc['figure.figsize'] = (width, height)
# Between 5pt and 7pt.
rc['font.size'] = 6
rc['axes.titlesize'] = 9
rc['axes.labelsize'] = 8
rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7


def load():
    infected = plot_common.get_infected(run.store_path)
    extinction_time = plot_common.get_extinction_time(run.store_path)
    return (infected, extinction_time)


def plot_infected(ax, infected, SAT, draft=False):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[SAT].unstack('run')
    if draft:
        # Only plot the first 100 runs for speed.
        i = i.iloc[:, :100]
    # Start time at 0.
    t = i.index - i.index.min()
    ax.plot(t, i, color=plot_common.SAT_colors[SAT],
            alpha=0.15, linewidth=0.5,
            drawstyle='steps-pre', clip_on=False, zorder=4)
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(t, i.fillna(0).mean(axis='columns'),
            color='black', alpha=1,
            drawstyle='steps-pre', clip_on=False, zorder=4)
    # Tighten y-axis limits.
    ax.margins(y=0)
    # Shared x-axis with extinction time.
    ax.xaxis.set_tick_params(which='both',
                             labelbottom=False, labeltop=False)
    ax.xaxis.offsetText.set_visible(False)
    # Shared y-axis between SATs.
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Number\ninfected')
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.get_subplotspec().is_first_row():
        ax.set_title(f'SAT{SAT}', loc='center')


def plot_extinction_time(ax, extinction_time, SAT):
    et = extinction_time.loc[SAT]
    e = et.time.copy()
    e[~et.observed] = numpy.nan
    color = plot_common.SAT_colors[SAT]
    plot_common.kdeplot(e, ax=ax, color=color, shade=True,
                        clip_on=False, zorder=4)
    not_extinct = len(e[e.isnull()]) / len(e)
    if not_extinct > 0:
        (ne_min, p_min) = (0.6, 0.3)
        (ne_max, p_max) = (1, 1)
        pad = ((p_max - p_min) / (ne_max - ne_min) * (not_extinct - ne_min)
               + p_min)
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    (0.92, 0.8), xycoords='axes fraction',
                    bbox=bbox, color='white',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    zorder=4)
    # No y ticks.
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # Shared x-axes between SATs.
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(plot_common.t_name.capitalize())
    else:
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
    # Shared y-axis between SATs.
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Extinction\ntime')


def plot(infected, extinction_time, draft=False):
    SATs = infected.index.get_level_values('SAT').unique()
    nrows = 2
    ncols = len(SATs)
    height_ratios = (4, 1)
    row_inf = 0
    row_ext = 1
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(layout='constrained')
        gs = fig.add_gridspec(nrows, ncols,
                              height_ratios=height_ratios,
                              wspace=0.1, hspace=0.1)
        axes = numpy.empty((nrows, ncols), dtype=object)
        axes[0, 0] = None  # Make sharex & sharey work for axes[0, 0].
        for (col, SAT) in enumerate(SATs):
            for row in range(nrows):
                # Columns share the x scale.
                sharex = axes[0, col]
                # The infection plots share the y scale.
                # The extinction-time plots do *not* share the y scale.
                if row == row_inf:
                    sharey = axes[0, 0]
                elif row == row_ext:
                    sharey = None
                else:
                    raise ValueError(f'{row=}')
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=sharex,
                                                 sharey=sharey)
        for (col, SAT) in enumerate(SATs):
            plot_infected(axes[row_inf, col], infected, SAT,
                          draft=draft)
            plot_extinction_time(axes[row_ext, col], extinction_time, SAT)
        t_max = infected.index.get_level_values(plot_common.t_name).max()
        # I get weird results if I set these limits individually.
        for (col, SAT) in enumerate(SATs):
            for row in (row_inf, row_ext):
                ax = axes[row, col]
                ax.set_xlim(left=0, right=t_max)
                ax.set_ylim(bottom=0)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                if row == row_inf:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        seaborn.despine(fig=fig, top=True,
                        right=False, bottom=False, left=False)
        # For some reason, aligning the rows and columns works better
        # than aligning all axes.
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[[row_inf, row_ext], 0])
        fig.savefig('run_figure.pdf')
        fig.savefig('run_figure.png', dpi=300)


if __name__ == '__main__':
    draft = False
    infected, extinction_time = load()
    plot(infected, extinction_time, draft=draft)
    pyplot.show()
