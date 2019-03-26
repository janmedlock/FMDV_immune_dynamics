#!/usr/bin/python3

import os.path

import h5
import run_common


if __name__ == '__main__':
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for model in ('acute', 'chronic'):
            run_common.run_samples(model, tmax, store,
                                   logging_prefix=f'model {model}, ')
        store.repack(_filename)
