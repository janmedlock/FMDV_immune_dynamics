'''Model the antibody levels as a 2-state continuous-time Markov chain
with antibody states {negative, positive}, with antibody-gain and
-loss rates that are constant in time. The parameters are the
antibody-gain log rate and the antibody-loss log rate.'''

import numpy
import pandas

import _ci
import _ml
import _utility
import data


_NEGATIVE = 'negative'
_POSITIVE = 'positive'
# The calculations in the methods `Model.log_P()`, `Model.log_p()`,
# and `Model._log_p*()` assume this order of `_STATES`.
_STATES = (_NEGATIVE, _POSITIVE)
_STATES_DTYPE = pandas.CategoricalDtype(_STATES, ordered=True)


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
    data_ = data_.dropna(subset=['positive'])
    # Build 'state' column.
    data_['state'] = (
        data_.positive
        .map({True: _POSITIVE, False: _NEGATIVE})
        .astype(_STATES_DTYPE)
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
    {negative, positive}, with antibody-gain and -loss rates that are
    constant in time. The parameters are the antibody-gain log rate
    and the antibody-loss log rate.'''

    # See the comment benegative for `._transition_rates_to_parameters`
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

    def _log_p_negative_null(self):
        '''Get the logarithm of
        `p_negative(t | t_0, negative)` and `p_negative(t | t_0, positive)`
        when `lambda_gain` == `lambda_loss` == 0.'''
        # p_negative_negative = 1
        log_p_negative_negative = numpy.zeros_like(self.data.t)
        # p_negative_positive = 0
        log_p_negative_positive = -numpy.inf * numpy.ones_like(self.data.t)
        return (log_p_negative_negative, log_p_negative_positive)

    def log_p_negative(self, log_lambda):
        '''Calculate the logarithm of `p_negative(t | t_0, x_0)`
        the probabilities of being in the negative state at time `t`
        given the starting state `x_0` at time `t_0`.'''
        if _utility.exp_is_zero(log_lambda).all():
            # lambda_gain == lambda_loss == 0
            return self._log_p_negative_null()
        # `lambda_gain` > 0 or `lambda_loss` > 0.
        (log_lambda_gain, log_lambda_loss) = log_lambda
        # lambda_sum = lambda_gain + lambda_loss
        log_lambda_sum = _utility.log_add_exp(log_lambda_gain, log_lambda_loss)
        # lambda_sum > 0.
        assert not _utility.exp_is_zero(log_lambda_sum)
        # mu = numpy.exp(lambda_sum * (self.data.t  - self.data.t_0))
        log_mu = numpy.exp(
            log_lambda_sum
            + _utility.log_no_0_warning(self.data.t - self.data.t_0)
        )
        # phi = lambda_loss / (lambda_gain + lambda_loss)
        log_phi = log_lambda_loss - log_lambda_sum
        # p_negative_positive = phi - phi / mu
        log_p_negative_positive = _utility.log_sub_exp(log_phi,
                                                       log_phi - log_mu)
        # p_negative_negative = p_negative_positive + 1 / mu
        log_p_negative_negative = _utility.log_add_exp(log_p_negative_positive,
                                                       - log_mu)
        return (log_p_negative_negative, log_p_negative_positive)

    def log_p(self, log_lambda):
        '''Calculate the logarithm of the state probabities
        `p_x(t | t, x_0)`.'''
        # Choose between `p_negative(t | t_0, x_0)` using the initial
        # state, `x_0`.
        log_p_negative = _utility.choose_by_state(
            self.data.x_0, self.log_p_negative(log_lambda)
        )
        assert _utility.log_prob_is_valid(log_p_negative), f'{log_p_negative=}'
        # Compute the complementary probability,
        # `p_positive(t | t_0, x_0) = 1 - p_negative(t | t_0, x_0)`.
        log_p_positive = _utility.log_sub_exp(0, log_p_negative.clip(max=0))
        assert _utility.log_prob_is_valid(log_p_positive), f'{log_p_positive=}'
        return (log_p_negative, log_p_positive)

    def log_likelihood(self, log_lambda):
        '''The log likelihood.'''
        if any(numpy.isnan(log_lambda)):
            return numpy.nan
        # Choose between `p_x(t | t_0, x_0)` using the state `x`.
        log_p_x = _utility.choose_by_state(
            self.data.x, self.log_p(log_lambda)
        )
        ll = log_p_x.sum()
        if numpy.isnan(ll):
            ll = - numpy.inf
        assert ll <= 0, f'{ll=}'
        return ll

    def minus_log_likelihood(self, log_lambda):
        '''The minus log likelihood.'''
        return -self.log_likelihood(log_lambda)

    def _transition_rates(self):
        '''Estimate the rates of leaving each state from the count
        of these events divided by the total time exposed.'''
        rates = pandas.Series(
            index=pandas.Index(_STATES, name='x_0'),
            name='rate',
        )
        event = self.data.x != self.data.x_0
        exposure = self.data.t - self.data.t_0
        for x_0 in rates.index:
            is_x_0 = self.data.x_0 == x_0
            rates[x_0] = (
                event[is_x_0].sum()
                / exposure[is_x_0].sum()
            )
        return rates

    _transition_rates_to_parameters = {
        # The rate from the 'negative' state sets the gain parameter.
        _NEGATIVE: parameter_names[0],
        # The rate from the 'positive' state sets the loss parameter.
        _POSITIVE: parameter_names[1],
    }

    def parameters_initial_guess(self):
        '''Get a rough estimate of the model parameters.'''
        rates = self._transition_rates()
        log_lambda_0 = pandas.Series(index=self.parameter_index,
                                     name='initial_guess',
                                     dtype=float)
        for (x_0, rate) in rates.items():
            index = self._transition_rates_to_parameters[x_0]
            log_lambda_0[index] = numpy.log(rate)
        return log_lambda_0

    def estimate_ml(self, log_lambda_0=None, **kwds):
        '''Estimate the maximum-likelihood parameter values.'''
        if log_lambda_0 is None:
            log_lambda_0 = self.parameters_initial_guess()
        return _ml.estimate(self, log_lambda_0, **kwds)

    def estimate_ci(self, log_lambda_mle, alpha=0.05):
        '''Estimate the confidence intervals for the parameter values .'''
        return _ci.estimate(self, log_lambda_mle, alpha=alpha)

    def estimate_ml_and_ci(self, log_lambda_0=None, alpha=0.05, **kwds):
        '''Estimate the maximum-likelihood parameter values and their
        confidence intervals.'''
        log_lambda_mle = self.estimate_ml(log_lambda_0, **kwds)
        log_lambda_ci = self.estimate_ci(log_lambda_mle, alpha=alpha)
        return pandas.concat([log_lambda_mle, log_lambda_ci], axis='columns')
