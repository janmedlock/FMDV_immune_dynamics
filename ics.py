#!/usr/bin/python3
'''Use the cached initial conditions to predict missing values.'''

import pathlib
import warnings

import joblib
import numpy
import pandas
import scipy.special
import statsmodels.formula.api

import common
import herd
import herd.initial_conditions.immune_status.solver
import herd.samples
import samples


ICS_NAMES = [
    'hazard_infection',
    'newborn_proportion_immune',
]


def parameters_load():
    '''Load the parameters for each SAT.'''
    parameters = {SAT: herd.Parameters(SAT=SAT)
                  for SAT in common.SATs}
    return parameters


def get_ics_one(parameters, sample, cached_only=True, x_guess=None):
    '''Get the initial conditions.'''
    return herd.initial_conditions.immune_status.solver.get_optimizer(
        parameters.merge(**sample),
        cached_only=cached_only,
        x_guess=x_guess)


def get_ics(cached_only=True):
    '''Get all the initial conditions from the cache.'''
    samples_ = herd.samples.load()
    parameters = parameters_load()
    ics = {(SAT, sample_number): get_ics_one(parameters[SAT],
                                             sample[SAT],
                                             cached_only=cached_only)
           for (sample_number, sample) in samples_.iterrows()
           for SAT in common.SATs}
    ics = pandas.DataFrame.from_dict(ics, orient='index') \
                          .rename_axis(['SAT', 'sample_number'])
    # Move 'SAT' from the column index to the front of the row index,
    # so it is indexed like `ics`.
    samples__ = samples_.rename_axis('sample_number') \
                        .stack('SAT') \
                        .swaplevel()
    dfr = pandas.concat([samples__, ics], axis='columns')
    return dfr.sort_index(level=dfr.index.names)


def load_ics(cached_only=True):
    '''Load the cached initial conditions.'''
    filename = pathlib.Path(__file__).with_suffix('.pkl')
    try:
        ics = pandas.read_pickle(filename)
    except FileNotFoundError:
        assert False
        # This takes tens of minutes.
        ics = get_ics(cached_only=cached_only)
        # Save the values to avoid rebuilding.
        ics.to_pickle(filename)
    return ics



class _Transformer:
    '''Transform the parameters for regression.'''

    # Columns that are not transformed.
    _cols_identity = set()  # Empty.
    # Columns that are logit transformed.
    _cols_logit = {'probability_chronic', 'newborn_proportion_immune'}
    # All other columns are log transformed.

    def __init__(self, dfr):
        # Build data structures with functions that transform and
        # inverse transform each column.
        self._funcs = pandas.Series(self._identity,
                                    index=dfr.columns)
        self._inverse_funcs = self._funcs.copy()
        # Handle logit columns.
        self._funcs[self._cols_logit] = scipy.special.logit
        self._inverse_funcs[self._cols_logit] = scipy.special.expit
        # Handle log columns.
        self._cols_log = dfr.columns.difference(self._cols_identity
                                                | self._cols_logit)
        self._funcs[self._cols_log] = numpy.log
        self._inverse_funcs[self._cols_log] = numpy.exp

    @staticmethod
    def _identity(ser):
        '''Dummy function for identity transform.'''
        return ser

    @staticmethod
    def _vals_transform_base(funcs, data):
        '''Handle both tranform and inverse tranform by passing
        different `funcs`.'''
        if isinstance(data, pandas.DataFrame):
            # In case `data` is missing some columns, this will
            # restrict to only the functions for the columns present
            # in `data`.
            which = data.columns
        elif isinstance(data, pandas.Series):
            which = data.name
        else:
            raise ValueError
        return data.transform(funcs[which])

    def vals_transform(self, data):
        '''Transform the values in `data`.'''
        return self._vals_transform_base(self._funcs, data)

    def vals_inverse_transform(self, data):
        '''Inverse transform the values in `data`.'''
        return self._vals_transform_base(self._inverse_funcs, data)

    def var_transform(self, col):
        '''Transform the variable name `col`, for use in
        `statsmodels.formula.api` statistical formulas. For example,
        if the variable named 'probability_chronic' is logit
        transformed, return
        'scipy.special.logit(probability_chronic)'.'''
        if col in self._cols_identity:
            return col
        elif col in self._cols_log:
            return f'numpy.log({col})'
        elif col in self._cols_logit:
            return f'scipy.special.logit({col})'
        else:
            raise ValueError

    def vars_transform(self, cols):
        '''Transform the variable names `cols`, for use in statistical
        formulas.'''
        return [self.var_transform(col)
                for col in cols]


def predict_missing_sat(dfr):
    '''Predict missing initial conditions for one SAT.'''
    # Untransformed dependent variables.
    cols_dep = ICS_NAMES
    # Untransformed independent variables.
    cols_indep = dfr.columns.difference(cols_dep)
    # The object that transforms the variables.
    transformer = _Transformer(dfr)
    # Transformed independent variables.
    vars_indep = transformer.vars_transform(cols_indep)
    # Find the rows with missing initial conditions.
    missing = dfr[cols_dep].isnull().any(axis='columns')
    # Structure to store the predicted values.
    dfr_missing = dfr.loc[missing].copy()
    # Loop over the dependent variables.
    for col_dep in cols_dep:
        # Transformed dependent variable.
        var_dep = transformer.var_transform(col_dep)
        # The regression formula. This is a string like
        # 'y ~ x_0 + x_1 + ...'.
        formula = var_dep + ' ~ ' + ' + '.join(vars_indep)
        # Build the regression model. The rows with missing initial
        # conditions are dropped.
        model = statsmodels.formula.api.ols(formula,
                                            data=dfr.loc[~missing])
        # Estimate the regression coefficients.
        results = model.fit()
        # Predict the missing values of the transformed dependent
        # variable.
        pred = results.predict(dfr_missing) \
                      .rename(col_dep)
        # Inverse transform the predictions.
        dfr_missing[col_dep] = transformer.vals_inverse_transform(pred)
    return dfr_missing


def predict_missing(dfr):
    '''Predict missing initial conditions.'''
    # Loop over the SATs.
    grouper = dfr.groupby('SAT',
                          group_keys=False)
    return grouper.apply(predict_missing_sat)


def split_sample_ics(sample_ics):
    '''Split the parameter samples and the initial conditions.'''
    ics = sample_ics[ICS_NAMES]
    sample_idx = sample_ics.index.difference(ICS_NAMES)
    sample = sample_ics[sample_idx]
    return (sample, ics)


def run_missing_one(parameters, sample_ics, SAT, sample_number):
    '''Run one missing sample.'''
    (sample, ics) = split_sample_ics(sample_ics)
    try:
        print(f'Finding initial conditions for {SAT=}, {sample_number=}')
        get_ics_one(parameters, sample, x_guess=ics, cached_only=False)
    except AssertionError as err:
        warnings.warn(UserWarning(err))
        print(
            f'Failed finding initial conditions for {SAT=}, {sample_number=}'
        )
    else:
        print(f'Found initial conditions for {SAT=}, {sample_number=}')
        # Run the simulation.
        path = samples.sample_path(SAT, sample_number)
        # If `path` exists, it should have size 0.
        assert ((not path.exists())
                or (path.stat().st_size == 0))
        # `samples.run_one_and_save()` will not run if `path` exists,
        # so delete it.
        path.unlink(missing_ok=True)
        samples.run_one_and_save(parameters, sample,
                                 sample_number, path,
                                 logging_prefix=f'{SAT=}')


def run_missing(missing, n_jobs=-1):
    '''Run the missing samples in parallel.'''
    parameters = parameters_load()
    jobs = (joblib.delayed(run_missing_one)(parameters[SAT], sample_ics,
                                            SAT, sample_number)
            for ((SAT, sample_number), sample_ics) in missing.iterrows())
    joblib.Parallel(n_jobs=n_jobs)(jobs)


if __name__ == '__main__':
    ics = load_ics()
    missing = predict_missing(ics)
    run_missing(missing)
