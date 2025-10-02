'''Common code for running, analyzing, and plotting simulations with
varying parameters.'''

import math

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
import plotting
import stats


rc = plotting.rc | plotting.rc_text_small | {
    'figure.figsize': (plotting.WIDTH_MAXIMUM['double_column'], 4),
    'axes.spines.top': False,
    'axes.spines.right': False,
}


EXTINCTION_TIME_LABEL = f'Extinction {plotting.TIME_LABEL.lower()}'

PERSISTENCE_LABEL = (
    plotting.PERSISTENCE_LABEL
    .replace('FMDV ', 'FMDV\n')
)

YLABEL_VERTICALALIGNMENT = 'baseline'


def _save_result(store, result):
    '''Save extinction time.'''
    # If you change this, you must change `_copy_baseline()` to
    # save matching output.
    common.save_result(store, result,
                       extinction_time=True)


def _copy_baseline(store, nruns, SAT, **kwds):
    '''Copy the data from 'baseline.h5'.'''
    extinction_time = h5.load(baseline.store_path, 'extinction_time')
    index = extinction_time.index.to_frame()
    mask = (
        (index['SAT'] == SAT)
        & (index['run'] < nruns)
    )
    extinction_time_masked = extinction_time[mask]
    common.insert_index_levels(extinction_time_masked, 2, **kwds)
    store.put('extinction_time', extinction_time_masked)


def _run(module, SAT, val, nruns, store, *args, **kwargs):
    parameters_kwds = {
        'SAT': SAT,
        module.var: val,
    }
    if val == module.default:
        _copy_baseline(store, nruns, **parameters_kwds)
    else:
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for chunk in chunks:
            common.prepend_index_levels(chunk, **parameters_kwds)
            _save_result(store, chunk)


def run(module, nruns, *args, **kwargs):
    '''Run simulations varying the parameter in `module`.'''
    with h5.HDFStore(module.store_path) as store:
        for SAT in common.SATs:
            for val in module.values:
                _run(module, SAT, val, nruns, store, *args, **kwargs)
        store.repack()
        store.set_read_only()


def load(module):
    return h5.load(module.store_path, 'extinction_time')


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
    # `ser` is a array-valued `pandas.Series()`:
    # convert it to a `pandas.DataFrame()`.
    return pandas.DataFrame(ser.to_list(),
                            index=ser.index,
                            columns=time)


def plot_median(module, extinction_time, CI=0.5):
    vals = (
        extinction_time.index
        .get_level_values(module.var)
        .unique()
        .sort_values()
    )
    levels = [CI / 2, 1 - CI / 2]
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        grouper = extinction_time.groupby('SAT')
        nrows = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(nrows, sharex=True)
        idx_mid = len(axes) // 2
        for ((SAT, group), ax) in zip(grouper, axes):
            times = group.groupby(module.var).time
            median = times.median()
            ax.plot(median, median.index,
                    color=plotting.SAT_COLORS[SAT])
            CI_ = times.quantile(levels).unstack()
            ax.fill_betweenx(CI_.index, CI_[levels[0]], CI_[levels[1]],
                             color=plotting.SAT_COLORS[SAT],
                             alpha=0.5)
            ax.set_ylim(min(vals), max(vals))
            if module.log:
                ax.set_yscale('log')
            subplotspec = ax.get_subplotspec()
            if subplotspec.is_last_row():
                ax.set_xlim(0, common.TIME_MAX)
                ax.set_xlabel(EXTINCTION_TIME_LABEL)
            if subplotspec.is_first_col():
                (_, _, idx, _) = subplotspec.get_geometry()
                if idx == idx_mid:
                    ylabel = module.label
                else:
                    ylabel = ''
                ax.set_ylabel(f'SAT{SAT}\n{ylabel}')
        fig.align_ylabels()


def plot_survival(module, extinction_time):
    vals = (
        extinction_time.index
        .get_level_values(module.var)
        .unique()
        .sort_values()
    )
    with seaborn.axes_style('ticks'), \
         seaborn.color_palette('husl', len(vals), desat=1), \
         matplotlib.pyplot.rc_context(rc=rc):
        grouper = extinction_time.groupby('SAT')
        nrows = len(grouper)
        fig = matplotlib.pyplot.figure()
        # Add an extra row for the legend.
        gs = fig.add_gridspec(
            nrows + 1, 1,
            height_ratios=((1,) * nrows + (0.8,)),
        )
        axes = numpy.empty(nrows, dtype=object)
        axes[0] = None  # `sharex` for `axes[0]`.
        for row in range(nrows):
            axes[row] = fig.add_subplot(
                gs[row],
                sharex=axes[0],
            )
        for ((SAT, group), ax) in zip(grouper, axes):
            for (idx, g) in group.groupby(module.var):
                survival = stats.get_survival(g, 'time', 'observed')
                ax.plot(survival.index, survival,
                        label=f'{idx:g}',
                        drawstyle='steps-post',
                        alpha=0.7,
                        clip_on=False)
            ax.set_ylim(0, 1)
        for ax in axes[:-1]:
            ax.xaxis.set_tick_params(which='both',
                                     labelleft=False, labelright=False)
            ax.xaxis.offsetText.set_visible(False)
        axes[-1].set_xlabel(EXTINCTION_TIME_LABEL)
        axes[-1].set_xlim(0, common.TIME_MAX)
        (handles, labels) = axes[0].get_legend_handles_labels()
        nrow = 2
        ncol = math.ceil(len(handles) / nrow)
        plotting.legend_multicolumn(fig, handles, labels, ncol,
                                    title=module.label,
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0))


def plot_kde(module, extinction_time):
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        grouper = extinction_time.groupby('SAT')
        nrows = len(grouper)
        fig = matplotlib.pyplot.figure()
        # Add an extra row for the legend.
        gs = fig.add_gridspec(
            nrows + 1, 1,
            height_ratios=((1,) * nrows + (0.8,)),
        )
        axes = numpy.empty(nrows, dtype=object)
        axes[0] = None  # `sharex` for `axes[0]`.
        for row in range(nrows):
            axes[row] = fig.add_subplot(
                gs[row],
                sharex=axes[0],
            )
        for ((SAT, group), ax) in zip(grouper, axes):
            for (s, g) in group.groupby(module.var):
                e = g.time.copy()
                e[~g.observed] = numpy.nan
                plotting.kdeplot(e, label=f'{s:g}', ax=ax,
                                 shade=False, clip_on=False)
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.set_ylabel(f'SAT{SAT}\ndensity')
            ax.margins(y=0)
        for ax in axes[:-1]:
            ax.xaxis.set_tick_params(which='both',
                                     labelleft=False, labelright=False)
            ax.xaxis.offsetText.set_visible(False)
        axes[-1].set_xlabel(EXTINCTION_TIME_LABEL)
        axes[-1].set_xlim(0, common.TIME_MAX)
        (handles, labels) = axes[0].get_legend_handles_labels()
        nrow = 2
        ncol = math.ceil(len(handles) / nrow)
        plotting.legend_multicolumn(fig, handles, labels, ncol,
                                    title=module.label,
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, 0))


def plot_kde_2d(module, extinction_time, save=True, show=False):
    vals = (
        extinction_time.index
        .get_level_values(module.var)
        .unique()
        .sort_values()
    )
    times = numpy.linspace(0, common.TIME_MAX, 301)
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        grouper = extinction_time.groupby('SAT')
        ncols = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(
            2, ncols,
            sharex='col', sharey='row',
            gridspec_kw={'height_ratios': (3, 1)})
        for ((SAT, group), axes_col) in zip(grouper, axes.T):
            persistence = (
                group.groupby(module.var)
                .apply(common.get_persistence)
            )
            density = get_density(group, module.var, times)
            ax = axes_col[0]
            cmap = plotting.get_cmap_SAT(SAT)
            # Use `density` to set the color range.
            vmax = density.max().max()
            # Plot `density * proportion_observed`.
            arr = density.T * (1 - persistence)
            ax.pcolormesh(arr.columns, arr.index, arr,
                          cmap=cmap, vmin=0, vmax=vmax,
                          shading='gouraud')
            if module.log:
                ax.set_xscale('log')
            ax.set_title(f'SAT{SAT}')
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel(EXTINCTION_TIME_LABEL,
                              verticalalignment=YLABEL_VERTICALALIGNMENT)
                ax.yaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(common.TIME_MAX / 5)
                )
            ax_po = axes_col[-1]
            ax_po.plot(persistence,
                       color=plotting.SAT_COLORS[SAT],
                       clip_on=False, zorder=3)
            if module.log:
                ax_po.set_xscale('log')
                ax_po.xaxis.set_major_formatter(
                    matplotlib.ticker.LogFormatter())
            ax_po.set_xlim(min(vals), max(vals))
            ax_po.set_xlabel(module.label)
            if ax_po.get_subplotspec().is_first_col():
                ax_po.set_ylabel(PERSISTENCE_LABEL,
                                 verticalalignment=YLABEL_VERTICALALIGNMENT)
                ax_po.set_ylim(0, 1)
                ax_po.yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(xmax=1))
        for ax in fig.axes:
            ax.axvline(module.default,
                       color='black', linestyle='dotted', alpha=0.7)
        fig.align_xlabels(axes[-1, :])
        fig.align_ylabels(axes[:, 0])
        if save:
            for suffix in ('.pdf', '.png'):
                fig.savefig(module.store_path.with_suffix(suffix))
        if show:
            matplotlib.pyplot.show()
        return fig
