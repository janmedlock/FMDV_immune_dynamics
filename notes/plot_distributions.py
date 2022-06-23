#!/usr/bin/python3

import numpy
from matplotlib import lines, pyplot, ticker
import seaborn

import sys
sys.path.append('..')
from herd import Parameters, RandomVariables
import common
sys.path.pop()


# The of order these determines the order of the legend.
SATs = ('all', 1, 2, 3)

colors = {SAT: 'black' if (SAT == 'all') else common.SAT_colors[SAT]
          for SAT in SATs}
labels = {SAT: 'All' if (SAT == 'all') else f'SAT{SAT}'
          for SAT in SATs}

# For 'all', use the first non-'all' SAT.
SAT_map = ((SAT, SATs[1] if (SAT == 'all') else SAT)
           for SAT in SATs)
RVs = {SAT: RandomVariables(Parameters(SAT=v))
       for (SAT, v) in SAT_map}

# Common to all SATs.
common = ('mortality', 'birth', 'maternal_immunity_waning')


def get_RV(RVs, name):
    if name in common:
        which = ('all', )
    else:
        which = (SAT for SAT in SATs if (SAT != 'all'))
    return {SAT: getattr(RVs[SAT], name) for SAT in which}


width = 390 / 72.27
height = 0.6 * width
rc = {'figure.figsize': (width, height),
      'font.size': 7,
      'axes.titlesize': 'large',
      'legend.frameon': False}
ncols = 6
nrows = 2
with pyplot.rc_context(rc=rc):
    fig, axes = pyplot.subplots(nrows, ncols, sharex='col')
    fig.align_labels()
    axes_hazards = axes[0, :]
    axes_survivals = axes[1, :]
    j = -1

    RV = get_RV(RVs, 'mortality')
    title = 'Death'
    xlabel = 'Age ($\mathrm{y}$)'
    t_max = 20
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT],
                            linestyle='steps-post')
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'birth')
    title = 'Birth'
    xlabel = 'Time ($\mathrm{y}$)'
    t_max = 3
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t, 4 + t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t, 0, 4), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'maternal_immunity_waning')
    title = 'Waning'
    xlabel = 'Age ($\mathrm{y}$)'
    t_max = 1
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'progression')
    title = 'Progression'
    xlabel = 'Time ($\mathrm{d}$)'
    t_max = 10
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t / 365), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t / 365), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'recovery')
    title = 'Recovery'
    xlabel = 'Time ($\mathrm{d}$)'
    t_max = 15
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t / 365), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t / 365), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    RV = get_RV(RVs, 'chronic_recovery')
    title = 'Chronic\nRecovery'
    xlabel = 'Time ($\mathrm{y}$)'
    t_max = 1
    t = numpy.linspace(0, t_max, 1001)
    j += 1
    for SAT, v in RV.items():
        axes_hazards[j].plot(t, v.hazard(t), color=colors[SAT])
        axes_survivals[j].plot(t, v.sf(t), color=colors[SAT])
    axes_hazards[j].set_title(title)
    axes_survivals[j].set_xlabel(xlabel)

    for ax in axes.flat:
        for l in ('top', 'right'):
            ax.spines[l].set_visible(False)
            ax.autoscale(tight=True)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        for line in ax.lines:
            line.set_clip_on(False)

    for ax in axes_hazards:
        ax.set_ylim(0, )
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
    for ax in axes_survivals:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: '{:g}%'.format(100 * x)))

    axes_hazards[0].set_ylabel(r'Hazard ($\mathrm{y}^{-1}$)')
    axes_survivals[0].set_ylabel('Survival')

    handles = [lines.Line2D([], [], color=color, label=labels[SAT])
               for (SAT, color) in colors.items()]
    fig.legend(handles=handles, markerfirst=False, loc='lower center',
               ncol=len(handles))

    fig.tight_layout(rect=(0, 0.07, 1, 1))

    fig.savefig('distributions.pgf')
    pyplot.show()
