#!/usr/bin/python3
#
# To do:
# * Update for `model='chronic'`.

from matplotlib import pyplot, ticker
import numpy
import pandas
import seaborn

import h5
import herd
import herd.samples
import stats


def _get_persistence_time(x):
    t = x.index.get_level_values('time (y)')
    return t.max() - t.min()


def _load_persistence_times(model):
    # Update here for `model='chronic'`.
    assert model == 'acute'
    results = h5.load('run_samples.h5')
    groups = results.groupby(['SAT', 'sample'])
    pt = groups.apply(_get_persistence_time)
    pt.name = 'persistence_time'
    # Move 'SAT' from row MultiIndex to columns.
    pt = pt.reset_index('SAT').pivot(columns='SAT')
    pt = pt.reorder_levels([1, 0], axis='columns')
    samples = herd.samples.load(model=model)
    # Put parameter values and persistence time together.
    df = pandas.concat([samples, pt], axis='columns', copy=False)
    df.columns.set_names('value', level=1, inplace=True)
    return df


def load_persistence_times(model='acute'):
    # Update here for `model='chronic'`.
    assert model == 'acute'
    try:
        df = h5.load('plot_samples.h5')
    except FileNotFoundError:
        df = _load_persistence_times(model=model)
        h5.dump(df, 'plot_samples.h5')
    return df


def _get_labels(name, rank):
    base = name.replace('_', ' ')
    if not rank:
        label = base
        label_resid = 'Residual ' + base.lower()
    else:
        label = 'Rank ' + base.lower()
        label_resid = 'Residual rank ' + base.lower()
    return (label, label_resid)


def plot_times(df):
    outcome = 'persistence_time'
    df = df.reorder_levels(order=[1, 0], axis='columns')
    fig, ax = pyplot.subplots()
    for SAT, col in df[outcome].items():
        x = numpy.hstack([0, col.sort_values()])
        cdf = numpy.linspace(0, 1, len(x))
        ax.step(x, 1 - cdf, where='post', label='SAT {}'.format(SAT))
    ax.set_xlabel('time (y)')
    ax.set_ylabel('Survival')
    ax.set_yscale('log')
    # Next smaller power of 10.
    # a = numpy.ceil(numpy.log10(1 / len(df)) - 1)
    # ax.set_ylim(10 ** a, 1)
    ax.legend()
    fig.savefig('plot_samples_times.pdf')


def plot_parameters(df, rank=True, marker='.', s=1, alpha=0.6):
    outcome = 'persistence_time'
    SATs = df.columns.get_level_values('SAT').unique()
    params = df.columns.get_level_values('value').unique().drop(outcome)
    (ylabel, ylabel_resid) = _get_labels(outcome, rank)
    for SAT in SATs:
        X = df.loc[:, (SAT, params)]
        y = df.loc[:, (SAT, outcome)]
        if rank:
            X = (X.rank() - 1) / (len(X) - 1)
            y = (y.rank() - 1) / (len(y) - 1)
        fig, axes = pyplot.subplots(X.shape[1], 2)
        for (i, (col, x)) in enumerate(X.items()):
            _, param = col
            color = 'C{}'.format(i)
            (xlabel, xlabel_resid) = _get_labels(param, rank)
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
        fig.savefig('plot_samples_parameters.pdf')


def plot_sensitivity(df, rank=True, errorbars=False):
    outcome = 'persistence_time'
    SATs = df.columns.get_level_values('SAT').unique()
    params = df.columns.get_level_values('value').unique().drop(outcome)
    n_samples = len(df)
    n_params = len(params)
    y = range(n_params)
    colors = None  # Set the first time through.
    with seaborn.axes_style('whitegrid'):
        fig, axes = pyplot.subplots(1, len(SATs), sharex='row')
        for (SAT, ax) in zip(SATs, axes):
            p = df.loc[:, (SAT, params)]
            o = df.loc[:, (SAT, outcome)]
            if rank:
                rho = stats.prcc(p, o)
                xlabel = 'PRCC'
                if errorbars:
                    rho_CI = stats.prcc_CI(rho, n_samples)
            else:
                rho = stats.pcc(p, o)
                xlabel = 'PCC'
                if errorbars:
                    rho_CI = stats.pcc_CI(rho, n_samples)
            ix = rho.abs().sort_values().index
            x = rho[ix]
            labels = ix.get_level_values('value')
            # Colors are defined by the order in the first SAT.
            if colors is None:
                colors = dict(zip(reversed(labels),
                                  seaborn.color_palette('tab10', n_params)))
            c = [colors[l] for l in labels]
            if errorbars:
                rho_err = pandas.DataFrame({'lower': rho - rho_CI['lower'],
                                            'upper': rho_CI['upper'] - rho}).T
                xerr = rho_err[ix].values
                kwds = dict(xerr=xerr,
                            error_kw=dict(ecolor='black',
                                          elinewidth=1.5,
                                          capthick=1.5,
                                          capsize=5,
                                          alpha=0.6))
            else:
                kwds = dict()
            ax.barh(y, x, height=1, left=0,
                    align='center', color=c, edgecolor=c,
                    **kwds)
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=2))
            ax.tick_params(axis='y', pad=35)
            ax.set_yticks(y)
            ax.set_ylim(- 0.5, n_params - 0.5)
            ylabels = [l.replace('_', '\n') for l in labels]
            ax.set_yticklabels(ylabels, horizontalalignment='center')
            ax.set_xlabel(xlabel)
            ax.set_title('SAT {}'.format(SAT))
            ax.grid(False, axis='y', which='both')
        seaborn.despine(fig, top=True, bottom=False, left=True, right=True)
        fig.tight_layout()
    fig.savefig('plot_samples_sensitivity.pdf')


if __name__ == '__main__':
    df = load_persistence_times()
    plot_times(df)
    # plot_parameters(df)
    plot_sensitivity(df)
    pyplot.show()
