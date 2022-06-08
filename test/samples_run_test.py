#!/usr/bin/python3

from context import herd
from context import samples_run


def get_path_and_logging_prefix(SAT):
    path = samples_run.samples_path / str(SAT)
    logging_prefix = f'{SAT=}'
    return (path, logging_prefix)


def needs_running(sample_number, path_dir):
    path = path_dir / f'{sample_number}.npy'
    return not path.exists()


def load_parameters_and_samples(SAT, index=None):
    parameters = herd.Parameters(SAT=SAT)
    samples = herd.samples.load(SAT=SAT)
    if index is not None:
        samples = samples.loc[index]
    return (parameters, samples)


def _run_sample(parameters, sample, tmax, sample_number,
                path, logging_prefix):
    print(f'Running {logging_prefix} sample {sample_number}.')
    samples_run.run_one_and_save(parameters, sample, tmax,
                                 sample_number, path,
                                 logging_prefix=logging_prefix)


def run_sample(SAT, tmax, sample_number):
    '''Run one `sample_number` for testing.'''
    (parameters, sample) = load_parameters_and_samples(SAT,
                                                       index=sample_number)
    (path, logging_prefix) = get_path_and_logging_prefix(SAT)
    assert needs_running(sample_number, path)
    _run_sample(parameters, sample, tmax, sample_number,
                path, logging_prefix)


def _run_samples_sequential(SAT, tmax, parameters, samples):
    (path, logging_prefix) = get_path_and_logging_prefix(SAT)
    for (sample_number, sample) in samples.iterrows():
        if needs_running(sample_number, path):
            _run_sample(parameters, sample, tmax, sample_number,
                        path, logging_prefix)


def run_samples_sequential(SAT, tmax, index=None):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples) = load_parameters_and_samples(SAT, index=index)
    _run_samples_sequential(SAT, tmax, parameters, samples)


def run_subsamples_sequential(SAT, tmax, n_subsamples, seed, index=None):
    '''Run a subsample.'''
    (parameters, samples) = load_parameters_and_samples(SAT, index=index)
    subsamples = samples.sample(n_subsamples, random_state=seed).sort_index()
    _run_samples_sequential(SAT, tmax, parameters, subsamples)


if __name__ == '__main__':
    tmax = 10
    SAT = 2
    sample_number = 309
    run_sample(SAT, tmax, sample_number)
    # index = [309, 326] + list(range(330, 348 + 1))
    # run_samples_sequential(SAT, tmax, index=index)
