#!/usr/bin/python3

from context import herd
from context import samples


def needs_running(SAT, sample_number):
    path = samples.sample_path(SAT, sample_number)
    return not path.exists()


def load_parameters_and_samples(SAT, index=None):
    parameters = herd.Parameters(SAT=SAT)
    samples = herd.samples.load(SAT=SAT)
    if index is not None:
        samples = samples.loc[index]
    return (parameters, samples)


def _run_sample(SAT, parameters, sample, sample_number, tmax):
    path = samples.sample_path(SAT, sample_number)
    logging_prefix = f'{SAT=}'
    print(f'Running {logging_prefix} sample {sample_number}.')
    samples.run_one_and_save(parameters, sample, tmax,
                             sample_number, path,
                             logging_prefix=logging_prefix)


def run_sample(SAT, sample_number, tmax):
    '''Run one `sample_number` for testing.'''
    assert needs_running(SAT, sample_number)
    (parameters, sample) = load_parameters_and_samples(SAT,
                                                       index=sample_number)
    _run_sample(SAT, parameters, sample, sample_number, tmax)


def _run_samples_sequential(SAT, parameters, samples, tmax):
    for (sample_number, sample) in samples.iterrows():
        if needs_running(SAT, sample_number):
            _run_sample(SAT, parameters, sample, sample_number, tmax)


def run_samples_sequential(SAT, tmax, index=None):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(SAT, index=index)
    _run_samples_sequential(SAT, parameters, samples, tmax)


def run_subsamples_sequential(SAT, tmax, n_subsamples, seed, index=None):
    '''Run a subsample.'''
    (parameters, samples) = load_parameters_and_samples(SAT, index=index)
    subsamples = samples.sample(n_subsamples, random_state=seed).sort_index()
    _run_samples_sequential(SAT, parameters, subsamples, tmax)


if __name__ == '__main__':
    tmax = 10
    SAT = 2
    sample_number = 309
    run_sample(SAT, sample_number, tmax)
    # index = [309, 326] + list(range(330, 348 + 1))
    # run_samples_sequential(SAT, tmax, index=index)
