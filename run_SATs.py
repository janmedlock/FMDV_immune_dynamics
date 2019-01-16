#!/usr/bin/python3

import os.path

import h5
import run_common


if __name__ == '__main__':
    chronic = True
    nruns = 1000
    tmax = 10

    data = run_common.run_SATs(chronic, nruns, tmax)
    _filebase, _ = os.path.splitext(__file__)
    if chronic:
        _filebase += '_chronic'
    _h5file = _filebase + '.h5'
    h5.dump(data, _h5file)
