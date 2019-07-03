#!/usr/bin/python3

import itertools

from matplotlib import pyplot, ticker
from matplotlib.backends import backend_pdf
import numpy
import pandas
import seaborn

import h5
import herd
import herd.samples
import stats


def _get_extinction(infected):
    t = infected.index.get_level_values('time (y)')
    time = t.max() - t.min()
    observed = (infected.iloc[-1] == 0)
    return [time, observed]


def _load_extinction_times():
    with h5.HDFStore('run_samples.h5', mode='r') as store:
        df = []
        for model in ('acute', 'chronic'):
            for SAT in (1, 2, 3):
                print(f'model={model} & SAT={SAT}')
                infected = store.select(
                    f'model={model} & SAT={SAT}',
                    columns=['exposed', 'infectious', 'chronic'])
                infected = infected.sum(axis='columns')
                groups = infected.groupby(['model', 'SAT', 'sample'])
                extinction = groups.apply(_get_extinction)
                extinction.columns = ['extinction_time',
                                      'extinction_observed']
                samples = herd.samples.load(model=model, SAT=SAT)
                samples.index = extinction.index
                df.append(pandas.concat([extinction, samples],
                                        axis='columns', copy=False))
        return pandas.concat(df, axis='index', copy=False)


def load_extinction_times():
    try:
        df = h5.load('plot_samples.h5')
    except OSError:
        df = _load_extinction_times()
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
    fig, ax = pyplot.subplots()
    groups = df['extinction_time'].groupby(['model', 'SAT'])
    for ((model, SAT), ser) in groups:
        x = numpy.hstack([0, ser.sort_values()])
        survival = numpy.linspace(1, 0, len(x))
        ax.step(x, survival, where='post',
                label=f'{model.capitalize()} model, SAT {SAT}')
    ax.set_xlabel('time (y)')
    ax.set_ylabel('Survival')
    ax.set_yscale('log')
    # Next smaller power of 10.
    # a = numpy.ceil(numpy.log10(1 / len(df)) - 1)
    # ax.set_ylim(10 ** a, 1)
    ax.legend()
    fig.savefig('plot_samples_times.pdf')


def plot_parameters(df, rank=True, marker='.', s=1, alpha=0.6):
    outcome = 'extinction_time'
    models = df.index.get_level_values('model').unique()
    SATs = df.index.get_level_values('SAT').unique()
    params = df.columns.drop(outcome)
    (ylabel, ylabel_resid) = _get_labels(outcome, rank)
    with backend_pdf.PdfPages('plot_samples_parameters.pdf') as pdf:
        for model in models:
            for SAT in SATs:
                X = df.loc[(model, SAT, slice(None)), params]
                y = df.loc[(model, SAT, slice(None)), outcome]
                if rank:
                    X = (X.rank() - 1) / (len(X) - 1)
                    y = (y.rank() - 1) / (len(y) - 1)
                fig, axes = pyplot.subplots(X.shape[1], 2)
                for (i, (param, x)) in enumerate(X.items()):
                    color = f'C{i}'
                    (xlabel, xlabel_resid) = _get_labels(param, rank)
                    axes[i, 0].scatter(x, y,
                                       color=color, marker=marker, s=s,
                                       alpha=alpha)
                    Z = X.drop(columns=param)
                    x_res = stats.get_residuals(Z, x)
                    y_res = stats.get_residuals(Z, y)
                    axes[i, 1].scatter(x_res, y_res,
                                       color=color, marker=marker, s=s,
                                       alpha=alpha)
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
                fig.suptitle(f'{model.capitalize()} model, SAT {SAT}',
                             y=1, size='medium')
                w = 0.02
                fig.tight_layout(pad=0, h_pad=0, w_pad=0,
                                 rect=[w, 0, 1 - w, 0.98])
                pdf.savefig(fig)


def plot_sensitivity(df, rank=True, errorbars=False):
    outcome = 'extinction_time'
    models = df.index.get_level_values('model').unique()
    SATs = df.index.get_level_values('SAT').unique()
    models_SATs = list(itertools.product(models, SATs))
    samples = df.index.get_level_values('sample').unique()
    params = df.columns.drop(outcome)
    n_samples = len(samples)
    n_params = len(params)
    y = range(n_params)
    colors = None  # Set the first time through.
    with seaborn.axes_style('whitegrid'):
        fig, axes = pyplot.subplots(1, len(models) * len(SATs), sharex='row')
        for ((model, SAT), ax) in zip(models_SATs, axes):
            p = df.loc[(model, SAT, slice(None)), params]
            o = df.loc[(model, SAT, slice(None)), outcome]
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
            # Colors are defined by the order in the first SAT.
            if colors is None:
                colors = dict(zip(reversed(params),
                                  seaborn.color_palette('tab10', n_params)))
            c = [colors[p] for p in params]
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
            ylabels = [p.replace('_', '\n') for p in params]
            ax.set_yticklabels(ylabels, horizontalalignment='center')
            ax.set_xlabel(xlabel)
            ax.set_title(f'{model.capitalize()} model, SAT {SAT}')
            ax.grid(False, axis='y', which='both')
        seaborn.despine(fig, top=True, bottom=False, left=True, right=True)
        fig.tight_layout()
    fig.savefig('plot_samples_sensitivity.pdf')


if __name__ == '__main__':
    df = load_extinction_times()
    plot_times(df)
    plot_parameters(df)
    plot_sensitivity(df)
    pyplot.show()
