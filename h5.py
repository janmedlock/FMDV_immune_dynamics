#!/usr/bin/python3

import os
import subprocess

import numpy
import pandas


def load(filename, key='df'):
    return pandas.read_hdf(filename, key)


def dump(df, filename, key='df', mode='w', format='table'):
    df.to_hdf(filename, key, mode=mode, format=format)
    repack(filename)


def repack(filename):
    tmp = filename + '.repack'
    try:
        subprocess.run(['ptrepack', '--chunkshape=auto',
                        '--propindexes', '--complevel=6',
                        '--complib=blosc:zstd', '--fletcher32=1',
                        filename, tmp],
                       check=True)
    except subprocess.CalledProcessError:
        os.remove(tmp)
    os.rename(tmp, filename)


def convert(filename):
    base, _ = os.path.splitext(filename)
    h5file = base + '.h5'
    dump(pandas.read_pickle(filename), h5file)
