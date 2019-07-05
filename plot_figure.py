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
import statsmodels
import pandas

import plot_start_times_SATs
import run_common


# Erin's colors.
SAT_colors = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}

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
    infected = []
    extinction_time = []
    for model in ('acute', 'chronic'):
        i = plot_start_times_SATs.get_infected(model=model)
        run_common._prepend_index_levels(i, model=model)
        infected.append(i)
        e = plot_start_times_SATs.get_extinction_time(model=model)
        run_common._prepend_index_levels(e, model=model)
        extinction_time.append(e)
    infected = pandas.concat(infected)
    extinction_time = pandas.concat(extinction_time)
    return (infected, extinction_time)


def plot_infected(ax, infected, model, SAT):
    nruns = 1000
    ix = (model, SAT, slice(None), slice(0, nruns))
    # Remove all indices except 'run' and 'time'.
    to_drop = infected.index.names[:-2]
    # .unstack('run') puts 'run' on columns, time on rows.
    i = infected.loc[ix].unstack('run').reset_index(to_drop, drop=True)
    # Start time at 0.
    t = i.index - i.index.min()
    ax.plot(365 * t, i, color=SAT_colors[SAT],
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
    # Shared y-axis between models.
    if ax.is_first_col():
        ax.set_ylabel('Number\ninfected')
        ax.annotate(f'SAT {SAT}', (-0.35, 0.2),
                    xycoords='axes fraction',
                    rotation=90, fontsize=rc['axes.titlesize'])
    else:
        ax.yaxis.set_tick_params(which='both',
                                 labelleft=False, labelright=False)
        ax.yaxis.offsetText.set_visible(False)
    if ax.is_first_row():
        ax.set_title(f'{model.capitalize()} model', loc='center')


def plot_extinction_time(ax, extinction_time, model, SAT):
    e = extinction_time.loc[(model, SAT)]
    color = SAT_colors[SAT]
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
    # Shared y-axis between models.
    if ax.is_first_col():
        ax.set_ylabel('Extinction\ntime')


def plot(infected, extinction_time):
    SATs = infected.index.get_level_values('SAT').unique()
    models = infected.index.get_level_values('model').unique()
    nrows = len(SATs) * 2
    ncols = len(models)
    height_ratios = (3, 1) * (nrows // 2)
    width_ratios = (1, 1)
    with seaborn.axes_style('whitegrid'), pyplot.rc_context(rc=rc):
        fig = pyplot.figure(constrained_layout=True)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                               height_ratios=height_ratios,
                               width_ratios=width_ratios)
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
            for (col, model) in enumerate(models):
                row_i = 2 * i
                row_e = 2 * i + 1
                plot_infected(axes[row_i, col], infected, model, SAT)
                plot_extinction_time(axes[row_e, col], extinction_time,
                                     model, SAT)
        # Shade time region from acute-model column
        # in chronic-model column.
        col_chronic = numpy.where(models == 'chronic')[0][0]
        ix = ('acute', )
        e_acute = extinction_time.loc[('acute', )]
        assert e_acute.notnull().all()
        e_acute_mask = e_acute.max()
        color = pyplot.rcParams['grid.color']
        for (i, SAT) in enumerate(SATs):
            for row in range(2 * i, 2 * i + 2):
                _, margin = axes[row, col_chronic].margins()
                axes[row, col_chronic].axvspan(0, e_acute_mask * (1 + margin),
                                               color=color, alpha=0.5,
                                               linewidth=0)
        # I get weird results if I set these limits individually.
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
    pyplot.show()
