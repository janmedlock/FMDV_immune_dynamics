'''Common code for running and plotting the parameter samples.'''

import pathlib

from joblib import delayed, Parallel
import numpy
import pandas

import baseline
import common
import h5
import herd
import herd.samples


store_path = pathlib.Path(__file__).with_suffix('.h5')
samples_path = store_path.with_suffix('')


def SAT_path(SAT):
    return samples_path.joinpath(str(SAT))


def sample_path(SAT, sample_number):
    return SAT_path(SAT).joinpath(f'{sample_number}.npy')


def run_one(parameters, sample, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    params = parameters.merge(**sample)
    return baseline.run_one(params, sample_number, *args, **kwargs)


def run_one_and_save(parameters, sample, sample_number, path, *args,
                     touch=True, **kwargs):
    '''Run one simulation and save the output.'''
    if not path.exists():
        if touch:
            path.touch(exist_ok=False)
        dfr = run_one(parameters, sample, sample_number,
                      *args, **kwargs)
        # Save the data for this sample.
        numpy.save(path, dfr.to_records())


def run(*args, n_jobs=-1, **kwargs):
    samples = herd.samples.load()
    parameters = {SAT: herd.Parameters(SAT=SAT)
                  for SAT in common.SATs}
    samples_path.mkdir(exist_ok=True)
    for SAT in common.SATs:
        SAT_path(SAT).mkdir(exist_ok=True)
    # For each sample, iterate over the SATs,
    # i.e. SAT changes fastest.
    jobs = (delayed(run_one_and_save)(parameters[SAT], sample[SAT],
                                      sample_number,
                                      sample_path(SAT, sample_number),
                                      *args, logging_prefix=f'{SAT=}',
                                      **kwargs)
            for (sample_number, sample) in samples.iterrows()
            for SAT in common.SATs)
    Parallel(n_jobs=n_jobs)(jobs)


def _get_SAT(path):
    return int(path.name)


def _get_sample_number(path):
    return int(path.stem)


def combine(unlink=True):
    with h5.HDFStore(store_path, mode='a') as store:
        # (SAT, sample) that are already in `store`.
        store_idx = store.get_index().droplevel(common.t_name).unique()
        paths_SAT = sorted(samples_path.iterdir(), key=_get_SAT)
        for path_SAT in paths_SAT:
            SAT = _get_SAT(path_SAT)
            paths_sample = sorted(path_SAT.iterdir(), key=_get_sample_number)
            for path_sample in paths_sample:
                sample = _get_sample_number(path_sample)
                if (SAT, sample) not in store_idx:
                    assert path_sample.stat().st_size > 0
                    recarray = numpy.load(path_sample)
                    dfr = pandas.DataFrame.from_records(recarray,
                                                        index=common.t_name)
                    common.prepend_index_levels(dfr, SAT=SAT, sample=sample)
                    print('Inserting '
                          + ', '.join((f'SAT={SAT}',
                                       f'sample={sample}'))
                          + '.')
                    store.put(dfr)
                if unlink:
                    path_sample.unlink()
            if unlink:
                path_SAT.rmdir()
        if unlink:
            samples_path.rmdir()
