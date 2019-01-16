#!/usr/bin/python3

import os.path

import h5
import run_common


if __name__ == '__main__':
    SAT = 1
    chronic = True
    nruns = 100
    tmax = 10

    data = run_common.run_start_times(nruns, SAT, chronic, tmax)

    _filebase, _ = os.path.splitext(__file__)
    if chronic:
        _filebase += '_chronic'
    _h5file = _filebase + '.h5'
    h5.dump(data, _h5file)
