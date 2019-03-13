#!/usr/bin/python3

import os.path

import h5
import herd


if __name__ == '__main__':
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        # for model in ('acute', 'chronic'):
        for model in ('chronic', ):
            run_common.run_samples_SATs(model, tmax, store)
        store.repack(_filename)
