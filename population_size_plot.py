#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size. This requires the file `population_size.h5`, which is
built by `population_size_run.py`.'''

from matplotlib import colors, pyplot, ticker
import numpy
import seaborn
import statsmodels.nonparametric.api

import common
import stats
import population_size


def load_extinction_time():
    return common.load_extinction_time(population_size.store_path)


def plot_median(df, CI=0.5):
    levels = [CI / 2, 1 - CI / 2]
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 1, sharex=True)
        idx_mid = len(axes) // 2
        for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
            by = 'population_size'
            times = group.groupby(by).time
            median = times.median()
            ax.plot(median, median.index,
                    color=common.SAT_colors[SAT])
            CI_ = times.quantile(levels).unstack()
            ax.fill_betweenx(CI_.index, CI_[levels[0]], CI_[levels[1]],
                             color=common.SAT_colors[SAT],
                             alpha=0.5)
            ax.set_xlim(left=0)
            ax.set_xlabel(f'extinction {common.t_name}')
            subplotspec = ax.get_subplotspec()
            if subplotspec.is_first_col():
                (_, _, idx, _) = subplotspec.get_geometry()
                if idx == idx_mid:
                    ylabel = 'Population\nsize'
                else:
                    ylabel = '\n\n'
                ax.set_ylabel(f'SAT{SAT}\n{ylabel}')
        fig.suptitle('')
        fig.tight_layout()


def plot_survival(df):
    fig, axes = pyplot.subplots(3, 1, sharex=True)
    for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
        for (popsize, g) in group.groupby('population_size'):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.plot(survival.index, survival,
                    label=f'population_size {popsize}',
                    drawstype='steps-post')


def plot_kde(df):
    with seaborn.axes_style('darkgrid'):
        fig, axes = pyplot.subplots(3, 1, sharex='col')
        for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
            subplotspec = ax.get_subplotspec()
            for (s, g) in group.groupby('population_size'):
                e = g.time.copy()
                e[~g.observed] = numpy.nan
                label = f'{s:g}' if subplotspec.is_first_row() else ''
                common.kdeplot(e, label=label, ax=ax,
                               shade=False, clip_on=False)
            if subplotspec.is_last_row():
                ax.set_xlabel(f'extinction {common.t_name}')
                ax.set_xlim(left=0)
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.set_ylabel(f'SAT{SAT}\ndensity')
        leg = fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5),
                         title='Population\nsize')
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def _get_cmap(color):
    '''White to `color`.'''
    return colors.LinearSegmentedColormap.from_list('name',
                                                    ['white', color])


def plot_kde_2d(df, save=True):
    rc = common.rc.copy()
    width = 183 / 25.4  # convert mm to in
    height = 4  # in
    rc['figure.figsize'] = (width, height)
    # Between 5pt and 7pt.
    rc['font.size'] = 6
    rc['axes.titlesize'] = 9
    rc['axes.labelsize'] = 8
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
    extinction_time_max = 10
    sigmas = df.index \
               .get_level_values('population_size') \
               .unique() \
               .sort_values()
    sigma_baseline = 1.
    with pyplot.rc_context(rc=rc):
        fig, axes = pyplot.subplots(1 + 1, 3, sharex='col', sharey='row',
                                    gridspec_kw=dict(height_ratios=(3, 1)))
        extinction_time = numpy.linspace(0, extinction_time_max, 301)
        for (i, (SAT, group_SAT)) in enumerate(df.groupby('SAT')):
            ax = axes[0, i]
            density = numpy.zeros((len(extinction_time), len(sigmas)))
            proportion_observed = numpy.zeros_like(sigmas, dtype=float)
            for (k, (p, g)) in enumerate(group_SAT.groupby('population_size')):
                ser = g.time[g.observed]
                nruns = len(g)
                proportion_observed[k] = len(ser) / nruns
                if proportion_observed[k] > 0:
                    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
                    kde.fit(cut=0)
                    density[:, k] = kde.evaluate(extinction_time)
                else:
                    density[:, k] = 0
            cmap = _get_cmap(common.SAT_colors[SAT])
            # Use raw `density` for color,
            # but plot `density * proportion_observed`.
            norm = colors.Normalize(vmin=0, vmax=numpy.max(density))
            ax.imshow(density * proportion_observed,
                      cmap=cmap, norm=norm, interpolation='bilinear',
                      extent=(min(sigmas), max(sigmas),
                              min(extinction_time), max(extinction_time)),
                      aspect='auto', origin='lower', clip_on=False)
            ax.autoscale(tight=True)
            ax.set_title(f'SAT{SAT}')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(f'extinction {common.t_name}')
                ax.yaxis.set_major_locator(
                    ticker.MultipleLocator(max(extinction_time) / 5))
            ax_po = axes[-1, i]
            ax_po.plot(sigmas, 1 - proportion_observed,
                       color=common.SAT_colors[SAT],
                       clip_on=False, zorder=3)
            ax_po.autoscale(tight=True)
            ax_po.set_xlabel('Population\nsize')
            if ax_po.get_subplotspec().is_first_col():
                ax_po.set_ylabel('persisting 10 y')
                ax_po.set_ylim(0, 1)
                ax_po.yaxis.set_major_formatter(
                    ticker.PercentFormatter(xmax=1))
        for ax in fig.axes:
            ax.axvline(sigma_baseline,
                       color='black', linestyle='dotted', alpha=0.7)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        if save:
            fig.savefig(population_size.store_path.with_suffix('.pdf'))
            fig.savefig(population_size.store_path.with_suffix('.png'),
                        dpi=300)
        return fig


if __name__ == '__main__':
    df = load_extinction_time()
    # plot_median(df)
    # plot_survival(df)
    # plot_kde(df)
    plot_kde_2d(df)
    pyplot.show()
