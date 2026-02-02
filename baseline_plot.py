#!/usr/bin/python3
'''Build a figure comparing the runs of the SATs. This requires the
file `baseline.h5`, which is built by `baseline_run.py`.'''

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import seaborn

import baseline
import h5
import plotting


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 3),
    'axes.spines.top': False,
}


def load(_module=baseline):
    infected = h5.load(_module.store_path, 'infected_daily')
    extinction_time = h5.load(_module.store_path, 'extinction_time')
    return (infected, extinction_time)


def plot_infected(ax, infected, SAT, draft=False):
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[SAT].unstack('run')
    if draft:
        # Only plot the first 100 runs for speed.
        i = i.iloc[:, :100]
    # Start time at 0.
    t = i.index - i.index.min()
    plot_kwds = {
        'drawstyle': 'steps-post',
        'clip_on': False,
        'zorder': 4,
    }
    ax.plot(t, i, color=plotting.SAT_COLORS[SAT],
            alpha=0.15, linewidth=0.5,
            **plot_kwds)
    # `i.fillna(0)` gives mean including those that
    # have gone extinct.
    ax.plot(t, i.fillna(0).mean(axis='columns'),
            color='black', alpha=1,
            **plot_kwds)
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
    color = plotting.SAT_COLORS[SAT]
    plotting.kdeplot(e, ax=ax, color=color, shade=True,
                   clip_on=False, zorder=4)
    not_extinct = len(e[e.isnull()]) / len(e)
    if not_extinct > 0:
        (ne_min, p_min) = (0.6, 0.3)
        (ne_max, p_max) = (1, 0.5)
        pad = ((p_max - p_min) / (ne_max - ne_min) * (not_extinct - ne_min)
               + p_min)
        bbox = dict(boxstyle=f'rarrow, pad={pad}',
                    facecolor=color, alpha=0.7, linewidth=0)
        ax.annotate('{:g}%'.format(not_extinct * 100),
                    (0.94, 0.89), xycoords='axes fraction',
                    bbox=bbox, color='white',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    zorder=4)
    # No y ticks.
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    # Shared x-axes between SATs.
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(plotting.TIME_LABEL)
    else:
        ax.xaxis.set_tick_params(which='both',
                                 labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)
    # Shared y-axis between SATs.
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Extinction\ntime')


def plot(infected, extinction_time,
         draft=False, save=True, show=False,
         _module=baseline):
    SATs = infected.index.get_level_values('SAT').unique()
    nrows = 2
    ncols = len(SATs)
    height_ratios = (4, 1)
    row_inf = 0
    row_ext = 1
    with seaborn.axes_style('whitegrid'), matplotlib.pyplot.rc_context(rc=rc):
        fig = matplotlib.pyplot.figure()
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
        t_max = infected.index.get_level_values('time').max()
        # I get weird results if I set these limits individually.
        for (col, SAT) in enumerate(SATs):
            for row in (row_inf, row_ext):
                ax = axes[row, col]
                ax.set_xlim(left=0, right=t_max)
                ax.set_ylim(bottom=0)
                ax.xaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(2)
                )
                ax.xaxis.set_minor_locator(
                    matplotlib.ticker.AutoMinorLocator(2)
                )
                ax.grid(axis='x', which='minor', visible=True)
                if row == row_inf:
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(100)
                    )
        # For some reason, aligning the rows and columns works better
        # than aligning all axes.
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[[row_inf, row_ext], 0])
        if save:
            for suffix in ('.pdf', '.png'):
                output_path = _module.store_path.with_suffix(suffix)
                plotting.savefig(fig, output_path)
        if show:
            matplotlib.pyplot.show()
        return fig


if __name__ == '__main__':
    DRAFT = False
    (infected, extinction_time) = load()
    plot(infected, extinction_time, draft=DRAFT)
