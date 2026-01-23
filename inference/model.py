'''Model the antibody levels as a 2-state continuous-time Markov chain
with antibody states {low, high}, with antibody-gain and -loss rates
that are constant in time. The parameters are the antibody-gain log
rate and the antibody-loss log rate.'''

import numpy
import pandas

import _ci
import _ml
import _utility
import data


_LOW = 'low'
_HIGH = 'high'
# The calculations in the methods `Model.log_P()`, `Model.log_p()`,
# and `Model._log_p*()` assume this order of `_STATES`.
_STATES = (_LOW, _HIGH)


def load_data(**kws):
    '''Load the raw data and rearrange it to the form used by `Model`.
    The resuling `pandas.DataFrame()` has columns:
    1. SAT,
    2. animal,
    3. number of current capture,
    4. t_0: time of previous capture,
    4. t: time of current capture,
    5. x_0: antibody state at previous capture,
    6. x: antibody state at current capture.'''
    data_ = data.load(**kws)
    # Drop rows with no antibody data.
    data_ = data_.dropna(subset=['high'])
    # Build 'state' column.
    data_['state'] = (
        data_.high
        .map({True: _HIGH, False: _LOW})
        .astype(pandas.CategoricalDtype(_STATES, ordered=True))
    )
    dfr = []
    grouper = data_.groupby(['SAT', 'Numeric Animal ID'],
                            observed=False)
    for (_, sub) in grouper:
        # Columns that will be used for the index.
        tmp = sub[['SAT', 'Numeric Animal ID', 'Capture Number']].copy()
        tmp['t_0'] = sub.date.shift(1)
        tmp['t'] = sub.date
        tmp['x_0'] = sub.state.shift(1)
        tmp['x'] = sub.state
        # Drop the first row, for which there is no previous state.
        tmp = tmp.iloc[1:]
        dfr.append(tmp)
    dfr = pandas.concat(dfr, ignore_index=True)
    # Convert dates to days since the earliest date.
    t_start = data_.date.min()
    for col in ('t_0', 't'):
        dfr[col] = (dfr[col] - t_start) / pandas.offsets.Day()
    return dfr


class Model:
    '''A 2-state continuous-time Markov chain with antibody states
    {low, high}, with antibody-gain and -loss rates that are
    constant in time. The parameters are the antibody-gain log rate
    and the antibody-loss log rate.'''

    # See the comment below for `._transition_rates_to_parameters`
    # for how it depends on the order of `.parameter_names`.
    parameter_names = ('antibody-gain log rate',
                       'antibody-loss log rate')

    parameter_index = pandas.CategoricalIndex(parameter_names,
                                              categories=parameter_names,
                                              ordered=True,
                                              name='parameter')

    def __init__(self, data_):
        self.data = data_
        assert (self.data.t >= self.data.t_0).all()

    def _log_p_null(self, log_c):
        '''Get the logarithm of `p_{low,low}` and `p_{high,low}`
        when `b` == 0.'''
        assert numpy.isneginf(log_c), f'{log_c=}'
        # p_high = 0
        log_p_high = - numpy.inf * numpy.ones_like(self.data.t)
        # p_low = 1
        log_p_low = numpy.zeros_like(self.data.t)
        return (log_p_low, log_p_high)

    def _log_p_constant(self, log_b, log_c):
        '''Get the logarithm of `p_{low,low}` and `p_{high,low}`
        when `b` > 0.'''
        if _utility.exp_is_zero(log_b):
            return self._log_p_null(log_c)
        # mu = numpy.exp(- b * (self.data.t  - self.data.t_0))
        log_mu = - numpy.exp(
            log_b
            + _utility.log_no_0_warning(self.data.t - self.data.t_0)
        )
        # phi = c / b
        log_phi = log_c - log_b
        # p_high = phi - phi * mu
        log_p_high = _utility.log_sub_exp(log_phi, log_phi + log_mu)
        # p_low = p_high + mu
        log_p_low = _utility.log_add_exp(log_p_high, log_mu)
        return (log_p_low, log_p_high)

    def log_p(self, log_h_gain, log_h_loss):
        '''Calculate the logarithm of `p_{low,low}` and `p_{high,low}`,
        the probabilities of being in the low state at time `t`
        given the starting state at time `t_0`.'''
        # b = h_gain + h_loss
        log_b = _utility.log_add_exp(log_h_gain, log_h_loss)
        # c = h_loss
        log_c = log_h_loss
        return self._log_p_constant(log_b, log_c)

    def log_P(self, theta):
        '''Calculate the logarithm of the state probabities at time
        `t` given the starting state `x_0` at time `t_0`.'''
        # Choose between `p_{x_0, low}` using the initial state.
        log_P_low = _utility.choose_by_state(
            self.data.x_0, self.log_p(*theta)
        )
        assert _utility.log_prob_is_valid(log_P_low), f'{log_P_low=}'
        # Compute the complementary probability
        # P_high = 1 - P_low.
        log_P_high = _utility.log_sub_exp(0, log_P_low.clip(max=0))
        assert _utility.log_prob_is_valid(log_P_high), f'{log_P_high=}'
        return (log_P_low, log_P_high)

    def log_likelihood(self, theta):
        '''The log likelihood.'''
        if any(numpy.isnan(theta)):
            return numpy.nan
        # Choose between `P_x` using the state.
        log_P_x = _utility.choose_by_state(
            self.data.x, self.log_P(theta)
        )
        ll = log_P_x.sum()
        if numpy.isnan(ll):
            ll = - numpy.inf
        assert ll <= 0, f'{ll=}'
        return ll

    def minus_log_likelihood(self, theta):
        '''The minus log likelihood.'''
        return -self.log_likelihood(theta)

    def _transition_rates(self):
        '''Estimate the rates of leaving each state from the count
        of these events divided by the total time exposed.'''
        rates = pandas.Series(
            index=pandas.Index(_STATES, name='x_0'),
            name='rate',
        )
        event = (self.data.x != self.data.x_0)
        exposure = self.data.t - self.data.t_0
        for x_0 in rates.index:
            is_x_0 = (self.data.x_0 == x_0)
            rates[x_0] = (
                event[is_x_0].sum()
                / exposure[is_x_0].sum()
            )
        return rates

    _transition_rates_to_parameters = {
        # The rate from the 'low' state sets the gain parameter.
        _LOW: parameter_names[0],
        # The rate from the 'high' state sets the loss parameter.
        _HIGH: parameter_names[1],
    }

    def parameters_initial_guess(self):
        '''Get a rough estimate of the model parameters.'''
        rates = self._transition_rates()
        theta_0 = pandas.Series(
            index=self.parameter_index,
            name='initial_guess',
        )
        assert all('log rate' in parameter
                   for parameter in self.parameter_names)
        for (x_0, rate) in rates.items():
            index = self._transition_rates_to_parameters[x_0]
            theta_0[index] = numpy.log(rate)
        return theta_0

    def estimate_ml(self, theta_0=None, global_=False, **kwds):
        '''Estimate the maximum-likelihood parameter values.'''
        if theta_0 is None:
            theta_0 = self.parameters_initial_guess()
        return _ml.estimate(self, theta_0, global_=global_, **kwds)

    def estimate_ci(self, theta_mle, alpha=0.05):
        '''Estimate the confidence intervals for the parameter values .'''
        return _ci.estimate(self, theta_mle, alpha=alpha)

    def estimate_ml_and_ci(self, theta_0=None, global_=False, alpha=0.05,
                           **kwds):
        '''Estimate the maximum-likelihood parameter values and their
        confidence intervals.'''
        theta_mle = self.estimate_ml(theta_0, global_=global_, **kwds)
        theta_ci = self.estimate_ci(theta_mle, alpha=alpha)
        return pandas.concat([theta_mle, theta_ci], axis='columns')
