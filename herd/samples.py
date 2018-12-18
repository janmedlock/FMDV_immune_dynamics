import os.path

import pandas


_path = os.path.join(os.path.dirname(__file__), 'data')
_filename = os.path.join(_path, 'posterior_SEIR.txt')
_filename_mi = os.path.join(_path, 'posterior_maternal_immunity.txt')

samples = pandas.read_csv(_filename,
                          delim_whitespace=True,
                          header=None)
samples.index.name = 'sample'
_params = pandas.Index(['progression_shape', 'progression_mean',
                        'recovery_shape', 'recovery_mean',
                        'transmission_rate'],
                       name='parameter')
_SATs = pandas.RangeIndex(1, 3 + 1, name='SAT')
_names = [_params.name, _SATs.name]  # Stupid pandas doesn't propagate names.
# Put 'parameter' first for convenience below.
# The order will be flipped at the end so 'SAT' is first.
samples.columns = pandas.MultiIndex.from_product([_params, _SATs],
                                                 names=_names)

# Convert from days to years.
samples['progression_mean'] /= 365
samples['recovery_mean'] /= 365
# Convert from per day to per year.
samples['transmission_rate'] *= 365

# The durations of maternal immunity, which are the same for each SAT.
_samples_mi = pandas.read_csv(_filename_mi,
                              delim_whitespace=True,
                              header=None)
_samples_mi.index.name = samples.index.name
_samples_mi.columns = pandas.Index(['maternal_immunity_duration_shape',
                                    'maternal_immunity_duration_mean'],
                                   name='parameter')
# Add the maternal-antibody parameters to each SAT.
for _param in _samples_mi.columns:
    for _SAT in _SATs:
        samples[(_param, _SAT)] = _samples_mi[_param]

# Flip order of column MultiIndex levels so 'SAT' is first.
samples.columns = samples.columns.reorder_levels(['SAT', 'parameter'])
samples.sort_index(axis='columns', inplace=True)
