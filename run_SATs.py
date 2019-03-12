#!/usr/bin/python3

import os.path

import h5
import run_common


if __name__ == '__main__':
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    store = h5.HDFStore(_filename)
    for chronic in (False, True):
        run_common.run_SATs(chronic, nruns, tmax, store)
    store.close()
    h5.repack(_filename)
