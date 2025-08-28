#!/usr/bin/python3
'''Run one simulation for each SAT.'''

import pandas

import baseline
import common
import h5
import herd
import sample_simulation


def run_SATs(seed):
    data = {}
    for SAT in common.SATs:
        p = herd.Parameters(SAT=SAT)
        data[SAT] = baseline.run_one(p, seed)
    data = pandas.concat(data, names=['SAT'])
    h5.dump(data, sample_simulation.store_path)
    return data


if __name__ == '__main__':
    data = run_SATs(sample_simulation.SEED)
