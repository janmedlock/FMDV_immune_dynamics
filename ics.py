#!/usr/bin/python3
'''Use the cached initial conditions to predict missing values.'''

import numpy
import pandas
import scipy.special
import statsmodels.formula.api

import common
import herd
import herd.initial_conditions.immune_status.solver
import herd.samples


def get_ics_one(parameters, sample, cached_only=False):
    '''Get the initial conditions.'''
    return herd.initial_conditions.immune_status.solver.get_optimizer(
        parameters.merge(**sample),
        cached_only=cached_only)


def get_ics(cached_only=False):
    '''Get all the initial conditions from the cache.'''
    samples = herd.samples.load()
    parameters = {SAT: herd.Parameters(SAT=SAT)
                  for SAT in common.SATs}
    ics = {(SAT, sample_number): get_ics_one(parameters[SAT],
                                             sample[SAT],
                                             cached_only=cached_only)
           for (sample_number, sample) in samples.iterrows()
           for SAT in common.SATs}
    ics = pandas.DataFrame.from_dict(ics, orient='index') \
                          .rename_axis(['SAT', 'sample_number'])
    # Move 'SAT' from the column index to the front of the row index,
    # so it is indexed like `ics`.
    samples_ = samples.rename_axis('sample_number') \
                      .stack('SAT') \
                      .swaplevel()
    dfr = pandas.concat([samples_, ics], axis='columns')
    return dfr.sort_index(level=dfr.index.names)



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

    def transform(self, dfr):
        '''Transform the data in `dfr`.'''
        # In case `dfr` is missing some columns, restrict to only the
        # functions for the columns present in `dfr`.
        funcs = self._funcs[dfr.columns]
        return dfr.transform(funcs)

    def inverse_transform(self, dfr):
        '''Inverse transform the data in `dfr`.'''
        # In case `dfr` is missing some columns, restrict to only the
        # functions for the columns present in `dfr`.
        inverse_funcs = self._inverse_funcs[dfr.columns]
        return dfr.transform(inverse_funcs)

    def transform_var(self, col):
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

    def transform_vars(self, cols):
        '''Transform the variable names `cols`, for use in statistical
        formulas.'''
        return [self.transform_var(col)
                for col in cols]


def predict_missing_sat(dfr):
    '''Predict missing initial conditions for one SAT.'''
    # Untransformed dependent variables.
    cols_dep = ['hazard_infection', 'newborn_proportion_immune']
    # Untransformed independent variables.
    cols_indep = dfr.columns.difference(cols_dep)
    # The object that transforms the variables.
    transformer = _Transformer(dfr)
    # Transformed independent variables.
    vars_indep = transformer.transform_vars(cols_indep)
    # Find the rows with missing initial conditions.
    missing = dfr[cols_dep].isnull().any(axis='columns')
    # Structure to store (transformed) predicted values.
    ics_missing = dfr.loc[missing, cols_dep].copy()
    # Loop over the dependent variables.
    for col_dep in cols_dep:
        # Transformed dependent variable.
        var_dep = transformer.transform_var(col_dep)
        # The regression formula. This is a string like
        # 'y ~ x_0 + x_1 + ...'.
        formula = var_dep + ' ~ ' + ' + '.join(vars_indep)
        # Build the regression model. The rows with missing initial
        # conditions are dropped.
        model = statsmodels.formula.api.ols(formula,
                                            data=dfr[~missing])
        # Estimate the regression coefficients.
        results = model.fit()
        # Predict the missing values of the transformed dependent
        # variable.
        ics_missing[col_dep] = results.predict(dfr[missing])
    # Inverse transform the predictions.
    return transformer.inverse_transform(ics_missing)


def predict_missing(dfr):
    '''Predict missing initial conditions.'''
    # Loop over the SATs.
    return pandas.concat(
        {SAT: predict_missing_sat(dfr.loc[SAT])
         for SAT in common.SATs},
        names=['SAT'])


if __name__ == '__main__':
    # Get the cached initial conditions. This takes tens of minutes.
    ics = get_ics(cached_only=True)
    # Save the values to avoid rebuilding.
    ics.to_pickle('ics.pkl')
    # Load the saved values instead of rebuilding.
    # ics = pandas.read_pickle('ics.pkl')
    # Run the regression and predict the missing initial conditions.
    missing = predict_missing(ics)
