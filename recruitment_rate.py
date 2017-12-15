#!/usr/bin/python3

import itertools

from matplotlib import pyplot
from matplotlib import ticker
import numpy
import pandas
from scipy import integrate
from scipy import stats

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


def gamma(mean, std):
    return stats.gamma((mean / std) ** 2,
                       scale = std ** 2 / mean)


if __name__ == '__main__':
    parameters = herd.parameters.Parameters()
    rvs = herd.RandomVariables(parameters)

    t0 = 0.5
    a0 = 4
    _scale = 1
    def birth_pdf(t):
        return numpy.where((t >= 0) & (t < 1),
                           rvs.birth.pdf(t, t0, a0),
                           0) / _scale
    # Scale so that birth_pdf integrates to 1 over [0, 1y].
    _scale, _ = integrate.quad(birth_pdf, 0, 1)

    # maternal_immunity_waning = rvs.maternal_immunity_waning
    maternal_immunity_waning = gamma(0.35, 0.2)

    t = numpy.linspace(0, 2, 301)
    df = pandas.DataFrame(index = t)
    df['Birth'] = birth_pdf(t)
    # df['Waning'] = maternal_immunity_waning.pdf(t)
    df['Susceptible'] = convolution(t, birth_pdf, maternal_immunity_waning)
    df.to_csv('recruitment_rate.csv')

    fig, ax = pyplot.subplots()
    ax.plot(t, df['Birth'], label = 'Birth')
    # ax.plot(t, df['Waning'], label = 'Waning')
    ax.plot(t, df['Susceptible'], label = 'Susceptible')
    ax.set_ylabel('Probability density')
    every = 3 # months
    xticks = numpy.arange(t[0], t[-1] + every / 12, every / 12)
    ax.set_xticks(xticks)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    xticklabels = itertools.cycle(months[::every])
    xticklabels = [next(xticklabels) for _ in range(len(xticks))]
    ax.set_xticklabels(xticklabels)
    ax.legend()
    fig.tight_layout()
    fig.savefig('recruitment_rate.png', dpi = 300)
    fig.savefig('recruitment_rate.pdf')
    fig.savefig('recruitment_rate.pgf')
    pyplot.show()
