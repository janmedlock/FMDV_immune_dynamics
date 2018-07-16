import os.path

import pandas


_path = os.path.join(os.path.dirname(__file__), 'data')

_samples = {}
for SAT in (1, 2, 3):
    _filename = os.path.join(
        _path, 'SAT{}_SEIRModel_PosteriorSamples.txt'.format(SAT))
    _samples[SAT] = pandas.read_csv(_filename,
                                    delim_whitespace=True,
                                    header=None,
                                    names=('progression_shape',
                                           'progression_mean',
                                           'recovery_shape',
                                           'recovery_mean',
                                           'transmission_rate'))
    # Convert from days to years.
    _samples[SAT]['progression_mean'] /= 365
    _samples[SAT]['recovery_mean'] /= 365
    # Convert from per day to per year.
    _samples[SAT]['transmission_rate'] *= 365

_filename = os.path.join(_path, 'AbDuration_PosteriorSamples.txt')
# `None` causes an error in `pandas.concat()`.
_samples[0] = pandas.read_csv(_filename,
                              delim_whitespace=True,
                              header=None,
                              names=('maternal_immunity_duration_shape',
                                     'maternal_immunity_duration_mean'))

samples = pandas.concat(_samples, axis=1, names=['SAT'])
samples.columns.set_names('parameter', level=1, inplace=True)
# Convert SAT `0` to `None`.
_l = list(samples.columns.levels[0])
_l[_l.index(0)] = None
samples.columns.set_levels(_l, level=0, inplace=True)
samples.index.name = 'sample'
