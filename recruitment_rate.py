#!/usr/bin/python3

import itertools

from matplotlib import pyplot
from matplotlib import ticker
import numpy
import pandas
from scipy import integrate
from scipy import stats
import seaborn

import herd


def integrand(s, t, birth_pdf, maternal_immunity_waning):
    return birth_pdf(s) * maternal_immunity_waning.pdf(t - s)


@numpy.vectorize
def convolution(t, birth_pdf, maternal_immunity_waning):
    if isinstance(maternal_immunity_waning, herd.rv.deterministic):
        return birth_pdf(t - maternal_immunity_waning._scale)
    else:
        I, _ = integrate.quad(integrand, 0, t,
                              args = (t, birth_pdf,
                                      maternal_immunity_waning))
        return I


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def xticks_months(ax, t):
    every = 3 # months
    xticks = numpy.arange(t[0], t[-1] + every / 12, every / 12)
    ax.set_xticks(xticks)
    xticklabels = itertools.cycle(months[::every])
    xticklabels = [next(xticklabels) for _ in range(len(xticks))]
    ax.set_xticklabels(xticklabels)


if __name__ == '__main__':
    n_points = 301
    t_birth = numpy.linspace(0, 1, n_points)
    t_waning = numpy.linspace(0, 1, n_points)
    t_susceptible = numpy.linspace(0, 2, n_points)

    parameters = herd.parameters.Parameters()
    rvs = herd.RandomVariables(parameters)

    t0 = 0.5
    a0 = 4
    _scale = 1
    def birth_pdf(t):
        '''
        Remove age dependence.
        '''
        return numpy.where((t >= 0) & (t < 1),
                           rvs.birth.pdf(t, t0, a0),
                           0) / _scale
    # Scale so that birth_pdf integrates to 1 over [0, 1y].
    _scale, _ = integrate.quad(birth_pdf, 0, 1)

    fig, ax = pyplot.subplots(1, 3)

    birth = birth_pdf(t_birth)
    ax[0].plot(t_birth, birth)
    ax[0].set_ylabel('Birth')
    xticks_months(ax[0], t_birth)

    waning = rvs.maternal_immunity_waning.pdf(t_waning)
    ax[1].plot(t_waning, waning)
    ax[1].set_ylabel('Maternal-immunity waning')
    # `\u2009` is a thin space.
    ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}\u2009y'))

    susceptible = convolution(t_susceptible,
                              birth_pdf,
                              rvs.maternal_immunity_waning)
    ax[2].plot(t_susceptible, susceptible)
    ax[2].set_ylabel('Susceptible recruitment')
    xticks_months(ax[2], t_susceptible)

    for ax_ in ax:
        ax_.yaxis.set_major_locator(ticker.NullLocator())
        ax_.set_ylim(bottom = 0)
    seaborn.despine(fig, top = True, right = True)

    fig.tight_layout()
    fig.savefig('recruitment_rate.png', dpi = 300)
    fig.savefig('recruitment_rate.pdf')
    fig.savefig('recruitment_rate.pgf')
    pyplot.show()
