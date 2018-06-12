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


def integrand_pdf(s, t, maternal_immunity_waning, birth, t0, a0):
    return birth.pdf(s, t0, a0) * maternal_immunity_waning.pdf(t - s)

def integrand_cdf(s, t, maternal_immunity_waning, birth, t0, a0):
    return birth.pdf(s, t0, a0) * maternal_immunity_waning.cdf(t - s)

@numpy.vectorize
def convolution(t, maternal_immunity_waning, birth, t0, a0):
    pdf, _ = integrate.quad(integrand_pdf, 0, t,
                            args = (t, maternal_immunity_waning,
                                    birth, t0, a0))
    cdf, _ = integrate.quad(integrand_cdf, 0, t,
                            args = (t, maternal_immunity_waning,
                                    birth, t0, a0))
    return numpy.ma.divide(pdf, 1 - cdf).filled(0)


def hazard(t, pdf):
    cdf = integrate.cumtrapz(pdf, t / 12, initial = 0)
    return pdf / (1 - cdf)


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def xticks_months(ax, t, t0):
    every = 3 # months
    xticks = numpy.arange(t[0], t[-1], every)
    if not numpy.isclose(xticks[-1], t[-1]):
        xticks = numpy.hstack((xticks, t[-1]))
    ax.set_xticks(xticks)
    month0 = int(t0 % 12)
    months_ = (months[month0:] + months[:month0])[::every]
    xticklabels = itertools.cycle(months_)
    xticklabels = [next(xticklabels) for _ in range(len(xticks))]
    ax.set_xticklabels(xticklabels)


if __name__ == '__main__':
    n_points = 301
    t0 = 6  # Start on 01 July.
    t_birth = numpy.linspace(0, 12, n_points)
    t_waning = numpy.linspace(0, 12, n_points)
    t_susceptible = numpy.linspace(0, 24, n_points)

    df = pandas.DataFrame()

    parameters = herd.parameters.Parameters()
    rvs = herd.RandomVariables(parameters)

    # For births, to remove age dependence.
    a0 = 4

    fig, ax = pyplot.subplots(1, 3)

    birth = pandas.Series(rvs.birth.hazard(t_birth / 12, t0 / 12, a0),
                          index = t_birth)
    ax[0].plot(t_birth, birth)
    ax[0].set_ylabel('Birth')
    xticks_months(ax[0], t_birth, t0)

    waning = pandas.Series(rvs.maternal_immunity_waning.hazard(t_waning / 12),
                           index = t_waning)
    ax[1].plot(t_waning / 12, waning)
    ax[1].set_ylabel('Maternal-immunity waning')
    # `\u2009` is a thin space.
    ax[1].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}\u2009y'))

    susceptible = pandas.Series(convolution(t_susceptible / 12,
                                            rvs.maternal_immunity_waning,
                                            rvs.birth,
                                            t0 / 12, a0),
                                index = t_susceptible)
    ax[2].plot(t_susceptible, susceptible)
    ax[2].set_ylabel('Susceptible recruitment')
    xticks_months(ax[2], t_susceptible, t0)

    for ax_ in ax:
        ax_.yaxis.set_major_locator(ticker.NullLocator())
        ax_.set_ylim(bottom = 0)
    seaborn.despine(fig, top = True, right = True)

    fig.tight_layout()
    fig.savefig('recruitment_rate.png', dpi = 300)
    fig.savefig('recruitment_rate.pdf')
    fig.savefig('recruitment_rate.pgf')

    df = pandas.DataFrame(dict(birth = birth,
                               waning = waning,
                               susceptible = susceptible))
    df.to_csv('recruitment_rate.csv')

    pyplot.show()
