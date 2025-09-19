'''Common code for running, analyzing, and plotting simulations with
varying parameters.'''

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas
import seaborn
import statsmodels.nonparametric.api

import baseline
import common
import h5
import herd
import stats


rc = common.rc | common.rc_text_small | {
    'figure.figsize': (common.WIDTH_MAXIMUM['double_column'], 4),
}


def _copy_runs(hdfstore_out, nruns, SAT, **kwds):
    '''Copy the data from 'baseline.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'run<{nruns}'))
    with h5.HDFStore(baseline.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(chunk, 2, **kwds)
            hdfstore_out.put(chunk)


def _run(module, SAT, val, nruns, hdfstore, *args, **kwargs):
    parameters_kwds = {
        'SAT': SAT,
        module.var: val,
    }
    if val == module.default:
        _copy_runs(hdfstore, nruns, **parameters_kwds)
    else:
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(dfr, **parameters_kwds)
            hdfstore.put(dfr)


def run(module, nruns, *args, **kwargs):
    '''Run simulations varying the parameter in `module`.'''
    with h5.HDFStore(module.store_path) as hdfstore:
        for SAT in common.SATs:
            for val in module.values:
                _run(module, SAT, val, nruns, hdfstore, *args, **kwargs)
        hdfstore.repack()
    common.set_read_only(module.store_path)


def _get_proportion_observed_one(grp):
    return sum(grp.observed) / len(grp)


def get_proportion_observed(dfr, by_var):
    grouper = dfr.groupby(by_var)
    return grouper.apply(_get_proportion_observed_one)


def _get_density_one(grp, time):
    ser = grp.time[grp.observed]
    # If all of `grp.observed` are `False`, the density is 0.
    # If all but one of `grp.observed` are `False`, the method below
    # returns NaNs.
    if len(ser) <= 1:
        return numpy.zeros_like(time)
    kde = statsmodels.nonparametric.api.KDEUnivariate(ser)
    kde.fit(cut=0)
    return kde.evaluate(time)


def get_density(dfr, by_var, time):
    grouper = dfr.groupby(by_var)
    ser = grouper.apply(_get_density_one, time)
    # Unforunately, we have an array-valued `pandas.Series()`,
    # so convert that to a `pandas.DataFrame()`.
    return pandas.DataFrame(ser.to_list(),
                            index=ser.index,
                            columns=time)


def load_extinction_time(module):
    return common.load_extinction_time(module.store_path)


def plot_median(module, df, CI=0.5):
    levels = [CI / 2, 1 - CI / 2]
    with seaborn.axes_style('darkgrid'):
        (fig, axes) = matplotlib.pyplot.subplots(3, 1, sharex=True)
        idx_mid = len(axes) // 2
        for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
            times = group.groupby(module.var).time
            median = times.median()
            ax.plot(median, median.index,
                    color=common.SAT_colors[SAT])
            CI_ = times.quantile(levels).unstack()
            ax.fill_betweenx(CI_.index, CI_[levels[0]], CI_[levels[1]],
                             color=common.SAT_colors[SAT],
                             alpha=0.5)
            ax.set_xlim(left=0)
            ax.set_xlabel(f'extinction {common.TIME_LABEL.lower()}')
            if module.log:
                ax.set_yscale('log')
            subplotspec = ax.get_subplotspec()
            if subplotspec.is_first_col():
                (_, _, idx, _) = subplotspec.get_geometry()
                if idx == idx_mid:
                    ylabel = module.label
                else:
                    ylabel = '\n\n'
                ax.set_ylabel(f'SAT{SAT}\n{ylabel}')
        fig.suptitle('')
        fig.tight_layout()


def plot_survival(module, df):
    (fig, axes) = matplotlib.pyplot.subplots(3, 1, sharex=True)
    for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
        for (idx, g) in group.groupby(module.var):
            survival = stats.get_survival(g, 'time', 'observed')
            ax.plot(survival.index, survival,
                    label=f'{module.var} {idx}',
                    drawstyle='steps-post')


def plot_kde(module, df):
    with seaborn.axes_style('darkgrid'):
        (fig, axes) = matplotlib.pyplot.subplots(3, 1, sharex='col')
        for ((SAT, group), ax) in zip(df.groupby('SAT'), axes):
            subplotspec = ax.get_subplotspec()
            for (s, g) in group.groupby(module.var):
                e = g.time.copy()
                e[~g.observed] = numpy.nan
                label = f'{s:g}' if subplotspec.is_first_row() else ''
                common.kdeplot(e, label=label, ax=ax,
                               shade=False, clip_on=False)
            if subplotspec.is_last_row():
                ax.set_xlabel(f'extinction {common.TIME_LABEL.lower()}')
                ax.set_xlim(left=0)
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.set_ylabel(f'SAT{SAT}\ndensity')
        leg = fig.legend(loc='center left', bbox_to_anchor=(0.8, 0.5),
                         title=module.label)
        fig.tight_layout(rect=(0, 0, 0.82, 1))


def plot_kde_2d(module, df, save=True):
    vals = df.index \
             .get_level_values(module.var) \
             .unique() \
             .sort_values()
    extinction_time = numpy.linspace(0, common.TIME_MAX, 301)
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        grouper_SAT = df.groupby('SAT')
        ncols = len(grouper_SAT)
        (fig, axes) = matplotlib.pyplot.subplots(
            2, ncols,
            sharex='col', sharey='row',
            gridspec_kw=dict(height_ratios=(3, 1)))
        for ((SAT, group_SAT), axes_col) in zip(grouper_SAT, axes.T):
            proportion_observed = get_proportion_observed(group_SAT,
                                                          module.var)
            density = get_density(group_SAT, module.var,
                                  extinction_time)
            ax = axes_col[0]
            cmap = common.get_cmap_SAT(SAT)
            extent=(min(vals), max(vals),
                    min(extinction_time), max(extinction_time)),
            # Use `density` to set the color range.
            vmax = density.max().max()
            # Plot `density * proportion_observed`.
            arr = density.T * proportion_observed
            ax.pcolormesh(arr.columns, arr.index, arr,
                          cmap=cmap, vmin=0, vmax=vmax,
                          shading='gouraud')
            if module.log:
                ax.set_xscale('log')
            ax.set_xlim(min(vals), max(vals))
            ax.set_title(f'SAT{SAT}')
            if ax.get_subplotspec().is_first_col():
                ax.set_xlabel(f'extinction {common.TIME_LABEL.lower()}')
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(
                        max(extinction_time) / 5))
            ax_po = axes_col[-1]
            ax_po.plot(1 - proportion_observed,
                       color=common.SAT_colors[SAT],
                       clip_on=False, zorder=3)
            if module.log:
                ax_po.set_xscale('log')
                ax_po.xaxis.set_major_formatter(
                    matplotlib.ticker.LogFormatter())
            ax_po.set_xlim(min(vals), max(vals))
            ax_po.set_xlabel(module.label)
            if ax_po.get_subplotspec().is_first_col():
                ax_po.set_ylabel(
                    f'Persisting {common.TIME_MAX} {common.TIME_UNIT}s'
                )
                ax_po.set_ylim(0, 1)
                ax_po.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(xmax=1))
        for ax in fig.axes:
            ax.axvline(module.default,
                       color='black', linestyle='dotted', alpha=0.7)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        if save:
            for suffix in ('.pdf', '.png'):
                fig.savefig(module.store_path.with_suffix(suffix))
        return fig
