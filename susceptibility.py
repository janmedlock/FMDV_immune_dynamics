#!/usr/bin/python3

from matplotlib import colors, pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import plot_common
import stats
import susceptibility_run


def load_extinction_times():
    return plot_common.get_extinction_time(susceptibility_run.store_path)


def plot_median(df, CI=0.5):
    row = dict(enumerate(range(3), 1))
    levels = [CI / 2, 1 - CI / 2]
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(len(row), 1, sharex=True)
        for (SAT, group) in df.groupby('SAT'):
            i = row[SAT]
            ax = axes[i]
            by = 'lost_immunity_susceptibility'
            times = group.groupby(by).time
            median = times.median()
            ax.plot(median, median.index,
                    color=plot_common.SAT_colors[SAT])
            CI_ = times.quantile(levels).unstack()
            ax.fill_betweenx(CI_.index, CI_[levels[0]], CI_[levels[1]],
                             color=plot_common.SAT_colors[SAT],
                             alpha=0.5)
            ax.set_xlim(left=0)
            ax.set_xlabel(f'extinction {plot_common.t_name}')
            if ax.is_first_col():
                if i == 1:
                    ylabel = 'Susceptibility\nof lost-immunity\nstate'
                else:
                    ylabel = '\n\n'
                ax.set_ylabel(f'SAT{SAT}\n{ylabel}')
        fig.suptitle('')
        fig.tight_layout()


def plot_survival(df):
    row = dict(enumerate(range(3), 1))
    fig, axes = pyplot.subplots(len(row), 1, sharex=True)
    for (SAT, group) in df.groupby('SAT'):
        i = row[SAT]
        ax = axes[i]
        for (s, g) in group.groupby('lost_immunity_susceptibility'):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.plot(survival.index, survival,
                    label=f'lost_immunity_susceptibility {s}',
                    drawstype='steps-post')


def plot_kde(df):
    row = dict(enumerate(range(3), 1))
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(len(row), 1)
        for (SAT, group) in df.groupby('SAT'):
            i = row[SAT]
            ax = axes[i]
            for (s, g) in group.groupby('lost_immunity_susceptibility'):
                ser = g.time[g.observed]
                proportion_observed = len(ser) / len(g)
                if proportion_observed > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    x = kde.support
                    y = proportion_observed * kde.density
                else:
                    x, y = [], []
                label = s if i == 0 else ''
                ax.plot(x, y, label=label, alpha=0.7)
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.set_xlim(left=0)
            ax.set_xlabel(f'extinction {plot_common.t_name}')
            if ax.is_first_col():
                ylabel = 'density' if i == 1 else ''
                ax.set_ylabel(f'SAT{SAT}\n{ylabel}')
        leg = fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5),
                         title='Susceptibility\nof lost-immunity\nstate')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def _get_cmap(color):
    '''White to `color`.'''
    return colors.LinearSegmentedColormap.from_list('name',
                                                    ['white', color])


def plot_kde_2d(df):
    persistence_time_max = 5
    sigmas = df.index \
               .get_level_values('lost_immunity_susceptibility') \
               .unique() \
               .sort_values()
    sigma_baseline = 1.
    fig, axes = pyplot.subplots(3, 1 + 1, sharex='col', sharey='row',
                                gridspec_kw=dict(width_ratios=(1, 0.5)))
    persistence_time = numpy.linspace(0, persistence_time_max, 301)
    for (i, (SAT, group_SAT)) in enumerate(df.groupby('SAT')):
        ax = axes[i, 0]
        density = numpy.zeros((len(sigmas), len(persistence_time)))
        proportion_observed = numpy.zeros_like(sigmas, dtype=float)
        for (k, (s, g)) in enumerate(group_SAT.groupby(
                'lost_immunity_susceptibility')):
            ser = g.time[g.observed]
            nruns = len(g)
            proportion_observed[k] = len(ser) / nruns
            if proportion_observed[k] > 0:
                kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                kde.fit(cut=0)
                density[k] = kde.evaluate(persistence_time)
            else:
                density[k] = 0
        cmap = _get_cmap(plot_common.SAT_colors[SAT])
        # Use raw `density` for color,
        # but plot `density * proportion_observed`.
        norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
        ax.imshow(density * proportion_observed[:, None],
                  cmap=cmap, norm=norm, interpolation='bilinear',
                  extent=(min(persistence_time), max(persistence_time),
                          min(sigmas), max(sigmas)),
                  aspect='auto', origin='lower', clip_on=False)
        ax.autoscale(tight=True)
        ax_po = axes[i, -1]
        ax_po.plot(1 - proportion_observed, sigmas,
                   color=plot_common.SAT_colors[SAT],
                   clip_on=False, zorder=3)
        ax_po.autoscale(tight=True)
        subplotspec = ax.get_subplotspec()
        if subplotspec.is_last_row():
            ax_po.set_xlabel('persisting 10 y')
            ax_po.xaxis.set_major_formatter(
                ticker.PercentFormatter(xmax=1))
            ax_po.xaxis.set_minor_locator(
                ticker.AutoMinorLocator(2))
        if subplotspec.is_last_row():
            ax.set_xlabel(f'extinction {plot_common.t_name}')
            ax.xaxis.set_major_locator(
                ticker.MultipleLocator(max(persistence_time) / 5))
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if subplotspec.is_first_col():
            ax.set_ylabel('Susceptibility\nof lost-immunity\nstate')
            ax.annotate(f'SAT{SAT}',
                        (-0.65, 0.5), xycoords='axes fraction',
                        rotation=90, verticalalignment='center')
    for ax in fig.axes:
        ax.axhline(sigma_baseline,
                   color='black', linestyle='dotted', alpha=0.7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('susceptibility.pdf')
    fig.savefig('susceptibility.png', dpi=300)


if __name__ == '__main__':
    df = load_extinction_times()
    # plot_median(df)
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
