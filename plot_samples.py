#!/usr/bin/python3

from matplotlib import pyplot, ticker
import pandas
import seaborn

import herd
from herd.samples import samples
import stats


def _get_persistence_time(x):
    t = x.index.get_level_values('time (y)')
    return t.max() - t.min()


def load_persistence_times():
    df = pandas.read_pickle('run_samples.pkl')
    groups = df.groupby(['SAT', 'sample'])
    persistence_times = groups.apply(_get_persistence_time)
    return persistence_times


def _get_labels(base, rank):
    if not rank:
        label = base
        label_resid = 'Residual ' + base.lower()
    else:
        label = 'Rank ' + base.lower()
        label_resid = 'Residual rank ' + base.lower()
    return (label, label_resid)


def plot_parameters(persistence_times, rank=True, marker='.', s=1, alpha=0.6):
    SATs = persistence_times.index.get_level_values('SAT').unique()
    (ylabel, ylabel_resid) = _get_labels('Persistence time', rank)
    for SAT in SATs:
        X = samples[SAT]
        y = persistence_times[SAT]
        if rank:
            X = (X.rank() - 1) / (len(X) - 1)
            y = (y.rank() - 1) / (len(y) - 1)
        fig, axes = pyplot.subplots(X.shape[1], 2)
        for (i, (col, x)) in enumerate(X.items()):
            color = 'C{}'.format(i)
            (xlabel, xlabel_resid) = _get_labels(col.replace('_', ' '), rank)
            axes[i, 0].scatter(x, y,
                               color=color, marker=marker, s=s, alpha=alpha)
            Z = X.drop(columns=col)
            x_res = stats.get_residuals(Z, x)
            y_res = stats.get_residuals(Z, y)
            axes[i, 1].scatter(x_res, y_res,
                               color=color, marker=marker, s=s, alpha=alpha)
            for j in (0, 1):
                axes[i, j].tick_params(pad=0, labelsize='small')
            axes[i, 1].yaxis.set_ticks_position('right')
            axes[i, 1].yaxis.set_label_position('right')
            for (j, l) in enumerate([xlabel, xlabel_resid]):
                axes[i, j].set_xlabel(l, labelpad=0, size='small')
        # Single ylabel per column.
        for (x, l, ha) in [[0, ylabel, 'left'], [1, ylabel_resid, 'right']]:
            fig.text(x, 0.5, l, size='small', rotation=90,
                     horizontalalignment=ha, verticalalignment='center')
        fig.suptitle('SAT {}'.format(SAT), y=1, size='medium')
        w = 0.02
        fig.tight_layout(pad=0, h_pad=0, w_pad=0,
                         rect=[w, 0, 1 - w, 0.98])


def plot_tornados(persistence_times, errorbars=False):
    n_samples = len(persistence_times)
    SATs = persistence_times.index.get_level_values('SAT').unique()
    colors = None
    with seaborn.axes_style('whitegrid'):
        fig, axes = pyplot.subplots(1, len(SATs), sharex='row')
        for (SAT, ax) in zip(SATs, axes):
            n_params = samples[SAT].shape[-1]
            rho = stats.prcc(samples[SAT], persistence_times[SAT])
            CI = stats.prcc_CI(rho, n_samples)
            xerr = numpy.row_stack((rho - CI['lower'], CI['upper'] - rho))
            ix = numpy.argsort(numpy.abs(rho))
            labels = samples[SAT].columns[ix]
            # Colors are defined by the order in the first SAT.
            if colors is None:
                colors = dict(zip(reversed(labels),
                                  seaborn.color_palette('tab10', n_params)))
            c = [colors[l] for l in labels]
            if errorbars:
                kwds = dict(xerr=xerr[:, ix],
                            error_kw=dict(ecolor='black',
                                          elinewidth=1.5,
                                          capthick=1.5,
                                          capsize=5,
                                          alpha=0.6))
            else:
                kwds = dict()
            patches = ax.barh(range(n_params), rho[ix],
                              height=1, left=0,
                              align='center',
                              color=c,
                              edgecolor=c,
                              **kwds)
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
            ax.tick_params(axis='y', pad=35)
            ax.set_yticks(range(n_params))
            ax.set_ylim(- 0.5, n_params - 0.5)
            ylabels = [l.replace('_', '\n') for l in labels]
            ax.set_yticklabels(ylabels, horizontalalignment='center')
            ax.set_xlabel('PRCC')
            ax.set_title('SAT {}'.format(SAT))
            ax.grid(False, axis='y', which='both')
        seaborn.despine(fig, top=True, bottom=False, left=True, right=True)
        fig.tight_layout()


if __name__ == '__main__':
    persistence_times = load_persistence_times()
    plot_parameters(persistence_times)
    plot_tornados(persistence_times)
    pyplot.show()
