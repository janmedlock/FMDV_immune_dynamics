#!/usr/bin/python3

import os.path

import h5
import herd


if __name__ == '__main__':
    chronic = True
    tmax = 10

    data = run_common.run_samples(chronic, tmax)

    _filebase, _ = os.path.splitext(__file__)
    if chronic:
        _filebase += '_chronic'
    _h5file = _filebase + '.h5'
    h5.dump(data, _h5file)
