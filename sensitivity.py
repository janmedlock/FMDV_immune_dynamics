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
import plotting


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
                output_path = module.store_path.with_suffix(suffix)
                plotting.savefig(fig, output_path)
        if show:
            matplotlib.pyplot.show()
        return fig
