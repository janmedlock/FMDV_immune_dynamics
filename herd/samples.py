'''Load sample parameter values.'''

import pathlib

import numpy
import pandas


_data_path = pathlib.Path(__file__).with_name('data')
_paths = {
    'acute_transmission': 'posterior_SEIR.txt',
    'maternal_immunity_duration': 'posterior_maternal_immunity.txt',
    'chronic_transmission_rate': 'posterior_chronic_transmission_rate.txt',
    'chronic_recovery': 'posterior_chronic_recovery.txt',
    'antibody_hazards': 'posterior_antibody_hazards.csv',
}
# All of these files are in `_data_path`.
_paths = {key: _data_path.joinpath(path)
          for (key, path) in _paths.items()}

_SATs = pandas.RangeIndex(1, 3 + 1, name='SAT')
_PARAMETERS_NAME = 'parameter'
_SAMPLES_NAME = 'sample'


def _load_acute_transmission():
    dfr = pandas.read_csv(_paths['acute_transmission'],
                          delim_whitespace=True,
                          header=None)
    dfr.index.name = _SAMPLES_NAME
    params = pandas.Index(['progression_shape', 'progression_mean',
                           'recovery_shape', 'recovery_mean',
                           'transmission_rate'],
                          name=_PARAMETERS_NAME)
    # The columns have the parameters adjacent across SAT,
    # so build the index so that SAT changes fastest
    # and use `.swaplevel()` to put SAT first.
    dfr.columns = pandas.MultiIndex.from_product([params, _SATs]) \
                                   .swaplevel()
    # Convert from days to years.
    dfr.loc[:, (_SATs, ['progression_mean', 'recovery_mean'])] /= 365
    # Convert from per day to per year.
    dfr.loc[:, (_SATs, 'transmission_rate')] *= 365
    return dfr


def _load_maternal_immunity_duration():
    # The durations of maternal immunity, which are the same for each SAT.
    dfr = pandas.read_csv(_paths['maternal_immunity_duration'],
                          delim_whitespace=True,
                          header=None)
    dfr.index.name = _SAMPLES_NAME
    dfr.columns = pandas.Index(['maternal_immunity_duration_shape',
                                'maternal_immunity_duration_mean'],
                               name=_PARAMETERS_NAME)
    # Expand the maternal-antibody parameters to each SAT.
    dfr = pandas.concat([dfr] * len(_SATs), keys=_SATs, axis='columns')
    return dfr


def _load_chronic_transmission_rate():
    dfr = pandas.read_csv(_paths['chronic_transmission_rate'],
                          delim_whitespace=True,
                          header=None)
    dfr.index.name = _SAMPLES_NAME
    params = pandas.Index(['chronic_transmission_rate'],
                          name=_PARAMETERS_NAME)
    dfr.columns = pandas.MultiIndex.from_product([_SATs, params])
    # Convert from per day to per year.
    dfr.loc[:, (_SATs, 'chronic_transmission_rate')] *= 365
    return dfr


def _load_chronic_recovery():
    dfr = pandas.read_csv(_paths['chronic_recovery'],
                          delim_whitespace=True,
                          header=None)
    dfr.index.name = _SAMPLES_NAME
    # The columns are the all-SAT shape and the mean for each SAT.
    dfr.columns = pandas.Index(
        ['chronic_recovery_shape'] + ['chronic_recovery_mean'] * len(_SATs),
        name=_PARAMETERS_NAME)
    # Label the SATs on the means.
    mean = dfr[['chronic_recovery_mean']]
    mean.columns = pandas.MultiIndex.from_arrays([_SATs, mean.columns])
    # Expand the shape to each SAT.
    shape = dfr[['chronic_recovery_shape']]
    shape = pandas.concat([shape] * len(_SATs), keys=_SATs, axis='columns')
    # Recombine mean and shape.
    dfr = pandas.concat([mean, shape], axis='columns')
    # Convert from days to years.
    dfr.loc[:, (_SATs, 'chronic_recovery_mean')] /= 365
    return dfr


def _load_antibody_hazards():
    dfr = pandas.read_csv(_paths['antibody_hazards'],
                          header=None)
    dfr.index.name = _SAMPLES_NAME
    params = pandas.Index(['antibody_gain_hazard',
                           'antibody_loss_hazard'],
                          name=_PARAMETERS_NAME)
    dfr.columns = pandas.MultiIndex.from_product([_SATs, params])
    # Convert from per day to per year.
    dfr *= 365
    return dfr


# probability_chronic: The posteriors are Beta distributed.
# Beta parameters by SAT.
_probability_chronic_beta_params = pandas.DataFrame(
    numpy.column_stack([[15, 2], [8, 10], [12, 6]]),
    index=['a', 'b'], columns=_SATs, copy=False)


def _load_probability_chronic(size, seed=1):
    # Use a seed for reproducibility.
    rng = numpy.random.default_rng(seed=seed)
    dfr = pandas.DataFrame(numpy.column_stack(
        [rng.beta(*_probability_chronic_beta_params[SAT], size=size)
         for SAT in _SATs]))
    dfr.index.name = _SAMPLES_NAME
    params = pandas.Index(['probability_chronic'],
                          name=_PARAMETERS_NAME)
    dfr.columns = pandas.MultiIndex.from_product([_SATs, params])
    return dfr


def load(SAT=None):
    '''Load the parameter samples.'''
    dfr = pandas.concat([_load_acute_transmission(),
                         _load_maternal_immunity_duration(),
                         _load_chronic_transmission_rate(),
                         _load_chronic_recovery(),
                         _load_antibody_hazards()],
                        axis='columns')
    size = len(dfr)
    dfr = pandas.concat([dfr,
                         _load_probability_chronic(size)],
                        axis='columns')
    dfr.sort_index(axis='columns', inplace=True)
    if SAT is None:
        return dfr
    else:
        return dfr[SAT]
