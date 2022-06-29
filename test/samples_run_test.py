#!/usr/bin/python3

from context import herd
from context import samples


def needs_running(SAT, sample_number):
    path = samples.sample_path(SAT, sample_number)
    return not path.exists()


def load_parameters_and_samples(SAT, index=None):
    parameters = herd.Parameters(SAT=SAT)
    samples_ = herd.samples.load(SAT=SAT)
    if index is not None:
        samples_ = samples_.loc[index]
    return (parameters, samples_)


def _run_sample(SAT, parameters, sample, sample_number):
    path = samples.sample_path(SAT, sample_number)
    logging_prefix = f'{SAT=}'
    print(f'Running {logging_prefix} sample {sample_number}.')
    samples.run_one_and_save(parameters, sample,
                             sample_number, path,
                             logging_prefix=logging_prefix,
                             touch=False)


def run_sample(SAT, sample_number):
    '''Run one `sample_number` for testing.'''
    assert needs_running(SAT, sample_number)
    (parameters, sample) = load_parameters_and_samples(SAT,
                                                       index=sample_number)
    _run_sample(SAT, parameters, sample, sample_number)


def _run_samples_sequential(SAT, parameters, samples_):
    for (sample_number, sample) in samples_.iterrows():
        if needs_running(SAT, sample_number):
            _run_sample(SAT, parameters, sample, sample_number)


def run_samples_sequential(SAT, index=None):
    '''Find the `sample_number` that breaks the runs.'''
    (parameters, samples_) = load_parameters_and_samples(SAT, index=index)
    _run_samples_sequential(SAT, parameters, samples_)


def run_subsamples_sequential(SAT, n_subsamples, seed, index=None):
    '''Run a subsample.'''
    (parameters, samples_) = load_parameters_and_samples(SAT, index=index)
    subsamples = samples_.sample(n_subsamples, random_state=seed).sort_index()
    _run_samples_sequential(SAT, parameters, subsamples)


if __name__ == '__main__':
    # SAT = 2
    # sample_number = 309
    # run_sample(SAT, sample_number)
    sample_numbers_by_SAT = {
        1: [407, 412, 425, 428, 429],
        2: [383, 393, 395, 422, 428],
        3: [376, 388, 398, 401, 402, 409, 414, 417, 419, 427, 428],
    }
    for (SAT, sample_numbers) in sample_numbers_by_SAT.items():
        run_samples_sequential(SAT, sample_numbers)
