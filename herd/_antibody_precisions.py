'''Handle antibody precisions for 'recovered' and 'lost_immunity'.'''

import pandas

from . import _confusion, data

# Map axes values to immune states
_IMMUNE_STATUS_MAP = {
    True: 'recovered',
    False: 'lost_immunity',
}

# Map axes names to standard names
_TRUE = 'true'
_OBSERVED = 'observed'
_TRUE_OBSERVED_MAP = {
    'VNT': _TRUE,
    'LPBE': _OBSERVED,
}


def _axes_names(dfr):
    '''Get the axes names.'''
    return dict(zip(
        ('index', 'columns'),
        (axis.name for axis in dfr.axes),
    ))


_CONFUSION_MATRIX_AXES_NAMES = {
    'index': _TRUE,
    'columns': _OBSERVED,
}


def _validate_confusion_matrix(cnf):
    '''Validate the confusion matrix.'''
    _confusion.validate(cnf)
    assert _axes_names(cnf) == _CONFUSION_MATRIX_AXES_NAMES


def _confusion_matrix(SAT):
    '''Load the confusion matrix for a SAT.'''

    def rename_immune_status(cnf):
        return cnf.rename(index=_IMMUNE_STATUS_MAP,
                          columns=_IMMUNE_STATUS_MAP)

    def axes_true_observed(cnf):
        return {axis: _TRUE_OBSERVED_MAP[name]
                for (axis, name) in _axes_names(cnf).items()}

    def rename_axes_true_observed(cnf):
        return cnf.rename_axis(**axes_true_observed(cnf))

    cnf = (
        data.load_antibodies_lpbe_vnt()
        .loc[SAT]
        .pipe(rename_immune_status)
        .pipe(rename_axes_true_observed)
    )
    _validate_confusion_matrix(cnf)
    return cnf


_PRECISION_INDEX = pandas.Index(
    _IMMUNE_STATUS_MAP.values(),
    name=_OBSERVED,
)


def _validate_precision(prc):
    '''Validate the precision.'''
    # `_confusion.precision()` calls `_confusion._validate_score()`,
    # which ensures the values are in the unit interval.
    pandas.testing.assert_index_equal(prc.index, _PRECISION_INDEX,
                                      check_order=False)


def _precision(cnf):
    '''Get the precisions.'''
    prc = _confusion.precision(cnf)
    _validate_precision(prc)
    return prc


def estimate(SAT):
    '''Get the estimated precisions.'''
    return _precision(
        _confusion_matrix(SAT)
    )


def _resample_cases(SAT, seed=None, **kwds):
    '''Resample cases in the confusion matrix, keeping observation
    counts constant.'''
    sample = _confusion.resample_cases(_confusion_matrix(SAT),
                                       seed=seed, **kwds)
    _validate_confusion_matrix(sample)
    return sample


def sample(SAT, seed=None, **kwds):
    '''Get sample precisions.'''
    return _precision(
        _resample_cases(SAT, seed=seed, **kwds)
    )
