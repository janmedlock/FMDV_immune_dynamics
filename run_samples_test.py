#!/usr/bin/python3

import os.path

import herd
import run_samples


def get_path_and_logging_prefix(model, SAT):
    path = os.path.join(run_samples._path, model, str(SAT))
    logging_prefix = f'model {model}, SAT {SAT}, '
    return (path, logging_prefix)


def get_filename(sample_number, path):
    return os.path.join(path, f'{sample_number}.npy')


def needs_running(sample_number, path):
    return not os.path.exists(get_filename(sample_number, path))


def load_parameters_and_samples(model, SAT):
    parameters = herd.Parameters(model=model, SAT=SAT)
    samples = herd.samples.load(model=model, SAT=SAT)
    return (parameters, samples)


def _run_sample(parameters, sample, tmax, path, sample_number, logging_prefix):
    print(f'Running {logging_prefix}sample {sample_number}.')
    run_samples._run_sample(parameters, sample, tmax, path,
                            sample_number, logging_prefix)


def run_sample(model, SAT, tmax, sample_number):
    '''Run one `sample_number` for testing.'''
    (parameters, samples) = load_parameters_and_samples(model, SAT)
    (path, logging_prefix) = get_path_and_logging_prefix(model, SAT)
    sample = samples.loc[sample_number]
    assert needs_running(sample_number, path)
    _run_sample(parameters, sample, tmax, path,
                sample_number, logging_prefix)


def _run_samples_sequential(model, SAT, tmax, parameters, samples):
    (path, logging_prefix) = get_path_and_logging_prefix(model, SAT)
    for (sample_number, sample) in samples.iterrows():
        if needs_running(sample_number, path):
            _run_sample(parameters, sample, tmax, path,
                        sample_number, logging_prefix)


def run_samples_sequential(model, SAT, tmax):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(model, SAT)
    _run_samples_sequential(model, SAT, tmax, parameters, samples)


def run_subsamples_sequential(model, SAT, tmax, n_subsamples, seed):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(model, SAT)
    subsamples = samples.sample(n_subsamples, random_state=seed).sort_index()
    _run_samples_sequential(model, SAT, tmax, parameters, subsamples)


if __name__ == '__main__':
    tmax = 10
    model = 'chronic'
    SAT = 1
    run_samples_sequential(model, SAT, tmax)
    # sample_number = 10919
    # run_sample(model, SAT, tmax, sample_number)
