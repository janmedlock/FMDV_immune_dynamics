#!/usr/bin/python3
import numpy
from matplotlib import lines, pyplot, ticker
import seaborn

import sys
sys.path.append('..')
from herd import Parameters
from herd import birth


tau = 1 / 4
mu = 1
CVs = (1 / 3, 1)

parameters = Parameters()
parameters.birth_peak_time_of_year = tau

t = numpy.linspace(0, 2, 1001)
colors = 'C3', 'C9'

width = 390 / 72.27
height = 0.6 * width
rc = {'figure.figsize': (width, height),
      'legend.frameon': False}
with pyplot.rc_context(rc=rc):
    fig, axes = pyplot.subplots()
    yticks = [0, mu]
    yticklabels = [r'$0$', r'$\mu$']
    for (CV, color) in zip(CVs, colors):
        parameters.birth_seasonal_coefficient_of_variation = CV
        birthRV = birth.gen(parameters, _scaling=mu)
        sign = '<' if (CV < 1 / numpy.sqrt(3)) else '>'
        label = r'$c_{{\mathrm{{v}}}} {} 1 / \sqrt{{3}}$'.format(sign)
        axes.plot(t, birthRV.hazard(t, 4), color=color, label=label,
                  clip_on=False)
        if CV < 1 / numpy.sqrt(3):
            yticks.extend([mu * birthRV._alpha * (1 - birthRV._beta),
                           mu * birthRV._alpha])
            yticklabels.extend(
                [r'$\left(1 - c_{\mathrm{v}} \sqrt{3}\right) \mu$',
                 r'$\left(1 + c_{\mathrm{v}} \sqrt{3}\right) \mu$'])
            axes.vlines(tau + 0.5, 0,
                        numpy.clip(mu * birthRV._alpha * (1 - birthRV._beta),
                                   0, numpy.inf),
                        linestyle='dotted', clip_on=False)
        else:
            yticks.extend([mu * birthRV._alpha])
            yticklabels.append(
                r'$\frac{3}{2} \left(1 + c_{\mathrm{v}}^2\right) \mu$')
            axes.vlines(tau,
                        0, mu * birthRV._alpha,
                        linestyle='dotted', clip_on=False)
    xticks = list(range(int(t[-1]) + 1))
    xticklabels = [r'${}$'.format(x) for x in xticks]
    xticks += [tau, tau + 0.5]
    xticklabels += [r'$\tau$', r'$\tau + \frac{1}{2}$']
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels, verticalalignment='center')
    axes.set_xlabel(r'Time ($\mathrm{y}$)')
    axes.tick_params(axis='x', pad=12)
    axes.set_ylabel(r'Birth hazard ($\mathrm{y}^{-1}$)')
    axes.set_yticks(yticks)
    axes.set_yticklabels(yticklabels)
    axes.hlines(yticks[1 : ], t[0], t[-1], linestyle='dotted',
                clip_on=False)
    axes.legend()
    for l in ('top', 'right'):
        axes.spines[l].set_visible(False)
        axes.autoscale(tight=True)
    fig.tight_layout()
    fig.savefig('birth_hazard.pgf')
    pyplot.show()
