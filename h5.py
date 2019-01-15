#!/usr/bin/python3

import os
import subprocess

import numpy
import pandas


class HDFStore(pandas.HDFStore):
    '''pandas.HDFStore() with fixed key.'''
    def __init__(self, path, key='df', *args, **kwds):
        super().__init__(path, *args, **kwds)
        self.key = key
        self._index = None
        self._columns = None

    def get(self):
        return super().get(self.key)

    def select(self, *args, **kwds):
        return super().select(self.key, *args, **kwds)

    def put(self, *args, **kwds):
        return super().put(self.key, *args, **kwds)

    @property
    def index(self):
        if self._index is None:
            self._index = self.select(columns=[]).index
        return self._index

    @property
    def columns(self):
        if self._columns is None:
            self._columns = self.select(stop=0).columns
        return self._columns


def load(filename, key='df', **kwds):
    return pandas.read_hdf(filename, key, **kwds)


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
