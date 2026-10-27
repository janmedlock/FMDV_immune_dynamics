'''Statistical functions for confusion matrices. Case (true) values
are the index (rows) and observed (predicted) values are the columns.'''

import numpy
import pandas
import sklearn.metrics


def validate(confusion):
    '''Validate the confusion matrix.'''

    def all_integer(dfr):
        return (
            len(dfr.select_dtypes(numpy.integer).columns)
            == len(dfr.columns)
        )

    def is_nonnegative(dfr):
        return (dfr >= 0) | dfr.isnull()

    assert all_integer(confusion)
    assert is_nonnegative(confusion).all().all()


def _groupby_drop(confusion, by):
    '''Make groups using the `by` level of the index without that
    level appearing the index of the groups.'''
    confusion_without_by = confusion.reset_index(by, drop=True)
    groups = confusion.index.get_level_values(by)
    return confusion_without_by.groupby(groups, sort=False)


def _in_unit_interval(dfr):
    return ((0. <= dfr) & (dfr <= 1.)) | dfr.isnull()


def _validate_score_matrix(score_matrix):
    '''Validate the score matrix.'''
    assert _in_unit_interval(score_matrix).all().all()


def _score_matrix(confusion, axis, by):
    '''Calculate the score matrix optionally `by` an index level.'''
    if by is not None:
        grouper = _groupby_drop(confusion, by)
        return grouper.apply(_score, axis, None)
    states = confusion.index.union(confusion.columns, sort=False)
    confusion = confusion.reindex(
        index=states.rename(confusion.index.name),
        columns=states.rename(confusion.columns.name),
    )
    score_matrix = confusion / confusion.to_numpy().sum(axis=axis,
                                                        keepdims=True)
    _validate_score_matrix(score_matrix)
    return score_matrix


def precision_matrix(confusion, by=None):
    '''Calculate the precision matrix, optionally `by` an index
    level. Precision is the frequency of each case label by
    observation labels.'''
    return _score_matrix(confusion, 0, by=by)


def recall_matrix(confusion, by=None):
    '''Calculate the recall matrix, optionally `by` an index
    level. Recall is the frequency of each observation label by case
    label.'''
    return _score_matrix(confusion, 1, by=by)


def _score_index_name(score_name, case_, observed):
    '''Name for the index in `_score()`.'''
    if score_name == 'precision':
        return observed
    if score_name == 'recall':
        return case_
    return 'class'


def _validate_score(score):
    '''Validate the score.'''
    assert _in_unit_interval(score).all()


def _score(confusion, score_name, by):
    '''Calculate the multiclass score from `sklearn.metrics`,
    optionally `by` an index level.'''
    if by is not None:
        grouper = _groupby_drop(confusion, by)
        return grouper.apply(_score, score_name, None)
    scorer = getattr(sklearn.metrics, f'{score_name}_score')
    # Convert from wide to long for `sklearn.metrics`
    case_ = confusion.index.name
    observed = confusion.columns.name
    count = 'count'
    while count in {case_, observed}:  # Ensure the column name is unique
        count += '_'
    dfr = (confusion.melt(value_name=count, ignore_index=False)
           .reset_index())
    index_name = _score_index_name(score_name, case_, observed)
    index = (confusion.index.union(confusion.columns, sort=False)
             .rename(index_name))
    score = pandas.Series(
        scorer(dfr[case_], dfr[observed], sample_weight=dfr[count],
               labels=index, average=None),
        index=index, copy=False,
    )
    _validate_score(score)
    return score


def precision(confusion, by=None):
    '''Calculate precision, optionally `by` an index level. Precision
    is the frequency of each observation label agreeing with the case
    labels.'''
    return _score(confusion, 'precision', by=by)


def recall(confusion, by=None):
    '''Calculate recall, optionally `by` an index level. Recall is the
    frequency of each case label agreeing with the observation labels.'''
    return _score(confusion, 'recall', by=by)


_ALPHA_PRIOR_DEFAULT = 0.5  # Jeffreys's prior


def _resample_counts(counts, alpha_prior, rng):
    '''Resample keeping the total constant, using the
    Dirichlet-multinomial distribution with concentration parameters
    `alpha_prior + counts`.'''
    n = counts.sum()
    alpha = alpha_prior + counts
    p = rng.dirichlet(alpha)
    sample = rng.multinomial(n, p)
    return sample


def _validate_resample(sample, confusion, axis):
    validate(sample)
    assert (sample.sum(axis=axis) == confusion.sum(axis=axis)).all()


def _resample(confusion, axis, alpha_prior, seed, by):
    '''Resample the confusion matrix, keeping total counts along
    `axis` constant, using the Dirichlet-multinomial distribution with
    concentration parameters `alpha_prior + counts`, optionally `by`
    an index level.'''
    rng = numpy.random.default_rng(seed)
    if by is not None:
        grouper = _groupby_drop(confusion, by)
        return grouper.apply(_resample, axis, alpha_prior, rng, None)
    sample = confusion.transform(_resample_counts, axis, alpha_prior, rng)
    _validate_resample(sample, confusion, axis)
    return sample


def resample_cases(confusion, alpha_prior=_ALPHA_PRIOR_DEFAULT,
                   seed=None, by=None):
    '''Resample cases in the confusion matrix, keeping observation
    counts constant, using the Dirichlet-multinomial distribution with
    concentration parameters `alpha_prior + counts`, optionally `by`
    an index level.'''
    return _resample(confusion, 'index', alpha_prior, seed, by)


def resample_observations(confusion, alpha_prior=_ALPHA_PRIOR_DEFAULT,
                          seed=None, by=None):
    '''Resample observations in the confusion matrix, keeping case
    counts constant, using the Dirichlet-multinomial distribution with
    concentration parameters `alpha_prior + counts`, optionally `by`
    an index level.'''
    return _resample(confusion, 'columns', alpha_prior, seed, by)
