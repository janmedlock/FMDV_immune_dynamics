#!/usr/bin/python3

import joblib
from matplotlib import dates, pyplot, ticker
import numpy
import pandas
from scipy import integrate
import seaborn

import herd
import herd.age_structure
import herd.birth
from herd.floquet.floquet import _agemax_default
import herd.maternal_immunity_waning
import herd.mortality
import herd.rv


class SusceptibleRecruitment(herd.rv.RV):
    '''The mixture of birth and maternal-immunity waning.'''
    def __init__(self, parameters, *args, **kwargs):
        self._age_structure = herd.age_structure.gen(parameters)
        self._birth = herd.birth.gen(parameters)
        self._maternal_immunity_waning = herd.maternal_immunity_waning.gen(
            parameters)
        self._mortality = herd.mortality.gen(parameters)
        self._t0 = parameters.start_time
        self._a0 = 4  # Remove age dependence from `self._birth`

    def _birth_rate_integrand(self, a, t):
        return (self._age_structure.pdf(a - t)
                * numpy.exp(self._mortality.logsf(a)
                            - self._mortality.logsf(a - t)))

    def _birth_rate(self, t):
        '''The birth rate is
        b(t)
        = \int_0^{\infty} n(t, a) p_{birth}(t, a) da
        = \int_0^{\infty} n(a - t)
                          S_{mortality}(a) / S_{mortality}(a - t)
                          p_{birth}(t, a) da.'''
        # Remove age dependence in `self._birth()` so that
        # b(t) = p_{birth}(t, a0)
        #        \int_{a0}^{\infty} n(a - t)
        #                           S_{mortality}(a) / S_{mortality}(a - t) da.
        a1 = _agemax_default
        # `self._age_structure` is 1-y periodic, so
        # \int_a0^{\infty} n(a - t)
        #                  S_{mortality}(a) / S_{mortality}(a - t)
        #                  da
        # = \int_a0^{\infty} n(a - \fracpart{t})
        #                    S_{mortality}(a) / S_{mortality}(a - \fracpart{t})
        #                    da.
        # This assumes a0 >= 1 y.
        t_frac = numpy.mod(t, 1)
        # This is the total reproductive-age population at t_0.
        n, _ = integrate.quadrature(self._birth_rate_integrand,
                                    self._a0, a1,
                                    args=(t_frac, ),
                                    maxiter=10000)
        # This is the total birth rate.
        b = n * self._birth.pdf(t, self._t0, self._a0)
        assert (b >= 0)
        return b

    def _pdf_integrand(self, s, t):
        return (self._birth_rate(s)
                * self._maternal_immunity_waning.pdf(t - s))

    def _pdf(self, t):
        pdf, _ = integrate.quadrature(self._pdf_integrand, self._t0, t,
                                      args=(t, ),
                                      vec_func=False,
                                      maxiter=10000)
        assert (pdf >= 0)
        return pdf

    def pdf(self, t):
        '''The PDF is
        p_{susceptible}(t)
        = \int_{t0}^t b(s) p_{waning}(t - s) ds,
        where the birth rate is
        b(t)
        = \int_0^{\infty} n(t, a) p_{birth}(t, a) da
        = \int_0^{\infty} n(a - t) S_{mortality}(t) p_{birth}(t, a) da.'''
        n_years = 10
        b = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self._birth_rate)(ti)
            for ti in numpy.ravel(t))
        t_long = numpy.hstack([t + y for y in range(n_years)])
        w = self._maternal_immunity_waning.pdf(t_long)
        dt = t[1] - t[0]
        v = numpy.convolve(b, w) * dt
        # Pad with 0's to get a shape that is a multiple of len(t).
        n = int(numpy.ceil(len(v) / len(t)))
        v = numpy.hstack([v, numpy.zeros(n  * len(t) - len(v))])
        # Sum over years.
        v = v.reshape((n, -1)).sum(axis=0)
        # Scale to sum to 1.
        v /= integrate.trapz(v, t)
        return v


def get_susceptible_recruitment():
    filename = 'susceptible_recruitment.csv'
    try:
        ser = pandas.read_csv(filename,
                              index_col=0, parse_dates=True,
                              header=None, squeeze=True)
    except FileNotFoundError:
        parameters = herd.Parameters()
        t0 = parameters.start_time
        susceptible_recruitment = SusceptibleRecruitment(parameters)
        # Susceptible recruitment depends on time of year.
        # 1 year of dates, points every day.
        year = pandas.Timedelta(days=dates.DAYS_PER_YEAR)
        start = (pandas.Timestamp(year=2001, month=1, day=1)
                 + t0 * year)
        end = start + year
        dt = pandas.date_range(start, end, freq='D', closed='left')
        # Convert to float year.
        t = (dt - dt[0]) / year
        v = susceptible_recruitment.pdf(t)
        ser = pandas.Series(v, index=dt)
        ser.to_csv(filename)
    return ser


def plot_susceptible_recruitment(ser):
    fig, ax = pyplot.subplots(constrained_layout=True)
    # Make a 2nd copy of the data for the following year.
    ser2 = ser.copy()
    ix = ser.index
    ser2.index = ix + ((ix[-1] - ix[0]) + (ix[1] - ix[0]))
    ser = ser.append(ser2)
    ax.plot(ser)
    ax.set_xlabel('Susceptible recruitment')
    ax.set_ylabel(r'Density ($\mathrm{y}^{-1}$)')
    # Major ticks every 3 months and minor ticks every month.
    interval = 3
    # Using `bymonth=...` instead of `interval=n` starts the month ticks
    # on January.
    ax.xaxis.set_major_locator(
        dates.MonthLocator(bymonth=range(1, 12 + 1, interval)))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(interval))
    # Use month for tick label.
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    seaborn.despine(fig, top=True, right=True)
    # fig.savefig('susceptible_recruitment.png', dpi=300)
    fig.savefig('susceptible_recruitment.pdf')
    # fig.savefig('susceptible_recruitment.pgf')


if __name__ == '__main__':
    ser = get_susceptible_recruitment()
    plot_susceptible_recruitment(ser)
    pyplot.show()
