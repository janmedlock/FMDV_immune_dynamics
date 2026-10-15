'''Statistical functions for confusion matrices. Case (true) values
are the index (rows) and observed (test) values are the columns.'''

import numpy
import pandas


def _groupby_drop(confusion, by):
    '''Make groups using the `by` level of the index without that
    level appearing the index of the groups.'''
    return (
        confusion.reset_index(by, drop=True)
        .groupby(confusion.index.get_level_values(by), sort=False)
    )


def _align_index_and_columns(confusion):
    '''Make `.index` and `.columns` have the same values,
    filling an missing values with 0.'''
    values = confusion.index.union(confusion.columns, sort=False)
    aligned = confusion.reindex(index=values.rename(confusion.index.name),
                                columns=values.rename(confusion.columns.name),
                                fill_value=0, copy=False)
    assert aligned.index.equals(aligned.columns)  # Doesn't compare `.name`
    return aligned


_AXES = ('index', 'columns')


def _axis_standardize(axis):
    '''Get the standardized form of `axis`, which is a string.'''
    try:
        # Convert integer `axis` to string
        axis = _AXES[axis]
    except TypeError:
        # Do not change string `axis`
        pass
    assert axis in _AXES
    return axis


def _axis_other(axis):
    '''The other axis.'''
    axis = _axis_standardize(axis)
    assert axis in _AXES
    assert len(_AXES) == 2
    (other,) = set(_AXES).difference({axis})
    return other


def _marginal_frequencies(confusion, axis, index_name, by=None):
    '''Get the marginal frequencies over `axis`. `axis='index'` is
    predicitive values. `axis='columns'` is true rates.'''
    if by is not None:
        # Recurse over `by` values
        return (
            _groupby_drop(confusion, by)
            .apply(_marginal_frequencies, axis, index_name)
        )
    # `by is None`
    confusion = _align_index_and_columns(confusion)
    totals = confusion.sum(axis=axis)
    freq = confusion.divide(totals, axis=_axis_other(axis))
    return pandas.Series(freq.to_numpy().diagonal(),
                         index=getattr(freq, axis).rename(index_name),
                         copy=False)


def predictive_values(confusion, by=None):
    '''Calculate predictive values from the confusion matrix,
    optionally `by` an index level. Predictive value is the frequency
    of each observation label agreeing with the case label.'''
    return _marginal_frequencies(confusion, 'index', 'predictive_value', by=by)


def true_rates(confusion, by=None):
    '''Calculate true rates from the confusion matrix,
    optionally `by` an index level. True rate is the frequency
    of each case label agreeing with the observation label.'''
    return _marginal_frequencies(confusion, 'columns', 'true_rate', by=by)


def resample_cases(confusion, by=None, alpha_prior=0.5, seed=None):
    '''Resample cases in the confusion matrix, keeping observation
    counts constant, using the Dirichlet-multinomial distribution with
    concentration parameters `alpha_prior + counts`, optionally `by`
    an index level. `alpha_prior=0.5` is Jeffreys's prior, `0` is
    Haldane's prior, and `1` is Bayes's prior.'''

    def resample_cases_one(rng, cases, alpha_prior):
        observations = cases.sum()
        alpha = alpha_prior + cases
        p = rng.dirichlet(alpha)
        return rng.multinomial(observations, p)

    rng = numpy.random.default_rng(seed)
    if by is not None:
        # Recurse over `by` values
        return _groupby_drop(confusion, by).apply(resample_cases,
                                                  alpha_prior=alpha_prior,
                                                  seed=rng)
    # `by is None`
    result = pandas.DataFrame(
        zip(*(
            resample_cases_one(rng, cases, alpha_prior)
            for (_, cases) in confusion.items()
        )),
        index=confusion.index,
        columns=confusion.columns,
        copy=False)
    # Ensure observation counts agree
    assert (confusion.sum(axis='index')
            == result.sum(axis='index')).all()
    return result
