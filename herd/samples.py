import os.path

import numpy
import pandas


_path = os.path.join(os.path.dirname(__file__), 'data')
_filenames = {
    'acute_transmission': 'posterior_SEIR.txt',
    'maternal_immunity_duration': 'posterior_maternal_immunity.txt',
    'chronic_transmission_rate': 'posterior_chronic_transmission_rate.txt',
    'chronic_recovery': 'posterior_chronic_recovery.txt'
}
# All of these files are in `_path`.
_filenames = {k: os.path.join(_path, v)
              for (k, v) in _filenames.items()}

_SATs = pandas.RangeIndex(1, 3 + 1, name='SAT')
_parameters_name = 'parameter'
_samples_name = 'sample'


def _load_acute_transmission():
    df = pandas.read_csv(_filenames['acute_transmission'],
                         delim_whitespace=True,
                         header=None)
    df.index.name = _samples_name
    params = ['progression_shape', 'progression_mean',
              'recovery_shape', 'recovery_mean',
              'transmission_rate']
    # The columns have the parameters adjacent across SAT,
    # so build the index so that SAT changes fastest
    # and use `.swaplevel()` to put SAT first.
    df.columns = pandas.MultiIndex.from_product(
        [params, _SATs],
        # Stupid pandas doesn't propagate names.
        names=[_parameters_name, _SATs.name]).swaplevel()
    # Convert from days to years.
    df.loc[:, (_SATs, ['progression_mean', 'recovery_mean'])] /= 365
    # Convert from per day to per year.
    df.loc[:, (_SATs, 'transmission_rate')] *= 365
    return df


def _load_maternal_immunity_duration():
    # The durations of maternal immunity, which are the same for each SAT.
    df = pandas.read_csv(_filenames['maternal_immunity_duration'],
                         delim_whitespace=True,
                         header=None)
    df.index.name = _samples_name
    df.columns = pandas.Index(['maternal_immunity_duration_shape',
                               'maternal_immunity_duration_mean'],
                              name=_parameters_name)
    # Expand the maternal-antibody parameters to each SAT.
    df = pandas.concat([df] * len(_SATs), keys=_SATs, axis='columns')
    return df


def _load_chronic_transmission_rate():
    df = pandas.read_csv(_filenames['chronic_transmission_rate'],
                         delim_whitespace=True,
                         header=None)
    df.index.name = _samples_name
    df.columns = pandas.MultiIndex.from_product(
        [_SATs, ['chronic_transmission_rate']],
        names=[_SATs.name, _parameters_name])
    # Convert from per day to per year.
    df.loc[:, (_SATs, 'chronic_transmission_rate')] *= 365
    return df


def _load_chronic_recovery():
    # Chronic recovery.
    df = pandas.read_csv(_filenames['chronic_recovery'],
                         delim_whitespace=True,
                         header=None)
    df.index.name = _samples_name
    # The columns are the all-SAT shape and the mean for each SAT.
    df.columns = pandas.Index(
        ['chronic_recovery_shape'] + ['chronic_recovery_mean'] * len(_SATs),
        name=_parameters_name)
    # Label the SATs on the means.
    mean = df[['chronic_recovery_mean']]
    mean.columns = pandas.MultiIndex.from_arrays([_SATs, mean.columns])
    # Expand the shape to each SAT.
    shape = df[['chronic_recovery_shape']]
    shape = pandas.concat([shape] * len(_SATs), keys=_SATs, axis='columns')
    # Recombine mean and shape.
    df = pandas.concat([mean, shape], axis='columns')
    # Convert from days to years.
    df.loc[:, (_SATs, 'chronic_recovery_mean')] /= 365
    return df


# probability_chronic: The posteriors are Beta distributed.
# Beta parameters by SAT.
_probability_chronic_beta_params = pandas.DataFrame(
    numpy.column_stack([[15, 2], [8, 10], [12, 6]]),
    index=['a', 'b'], columns=_SATs, copy=False)


def _load_probability_chronic(size, seed=1):
    # Use a seed for reproducibility.
    rng = numpy.random.RandomState(seed=seed)
    df = pandas.DataFrame(numpy.column_stack(
        [rng.beta(*_probability_chronic_beta_params[SAT], size=size)
         for SAT in _SATs]))
    df.index.name = _samples_name
    df.columns = pandas.MultiIndex.from_product(
        [_SATs, ['probability_chronic']],
        names=[_SATs.name, _parameters_name])
    return df


def load(chronic=False, SAT=None):
    df = pandas.concat([_load_acute_transmission(),
                        _load_maternal_immunity_duration()],
                       axis='columns')
    if chronic:
        size = len(df)
        df = pandas.concat([df,
                            _load_chronic_transmission_rate(),
                            _load_chronic_recovery(),
                            _load_probability_chronic(size)],
                           axis='columns')
    df.sort_index(axis='columns', inplace=True)
    if SAT is None:
        return df
    else:
        return df[SAT]
