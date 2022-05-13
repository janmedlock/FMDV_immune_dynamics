#!/usr/bin/python3

import os.path

import herd
import samples_run


def get_path_and_logging_prefix(SAT):
    path = os.path.join(samples_run._path, str(SAT))
    logging_prefix = f'SAT {SAT}, '
    return (path, logging_prefix)


def get_filename(sample_number, path):
    return os.path.join(path, f'{sample_number}.npy')


def needs_running(sample_number, path):
    return not os.path.exists(get_filename(sample_number, path))


def load_parameters_and_samples(SAT):
    parameters = herd.Parameters(SAT=SAT)
    samples = herd.samples.load(SAT=SAT)
    return (parameters, samples)


def _run_sample(parameters, sample, tmax, path, sample_number, logging_prefix):
    print(f'Running {logging_prefix}sample {sample_number}.')
    samples_run._run_sample(parameters, sample, tmax, path,
                            sample_number, logging_prefix)


def run_sample(SAT, tmax, sample_number):
    '''Run one `sample_number` for testing.'''
    (parameters, samples) = load_parameters_and_samples(SAT)
    (path, logging_prefix) = get_path_and_logging_prefix(SAT)
    sample = samples.loc[sample_number]
    assert needs_running(sample_number, path)
    _run_sample(parameters, sample, tmax, path,
                sample_number, logging_prefix)


def _run_samples_sequential(SAT, tmax, parameters, samples):
    (path, logging_prefix) = get_path_and_logging_prefix(SAT)
    for (sample_number, sample) in samples.iterrows():
        if needs_running(sample_number, path):
            _run_sample(parameters, sample, tmax, path,
                        sample_number, logging_prefix)


def run_samples_sequential(SAT, tmax):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(SAT)
    _run_samples_sequential(SAT, tmax, parameters, samples)


def run_subsamples_sequential(SAT, tmax, n_subsamples, seed):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(SAT)
    subsamples = samples.sample(n_subsamples, random_state=seed).sort_index()
    _run_samples_sequential(SAT, tmax, parameters, subsamples)


if __name__ == '__main__':
    tmax = 10
    SAT = 1
    run_samples_sequential(SAT, tmax)
    # sample_number = 10919
    # run_sample(SAT, tmax, sample_number)
