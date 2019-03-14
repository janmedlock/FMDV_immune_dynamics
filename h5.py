#!/usr/bin/python3

import os
import subprocess
import warnings

import numpy
import pandas
import tables


def repack(filename):
    '''
    Use `ptrepack` to compress the HDF file.
    '''
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


class _catch_natural_name_warnings(warnings.catch_warnings):
    '''
    Ignore `tables.NaturalNameWarning`.
    '''
    def __enter__(self):
        super().__enter__()
        warnings.filterwarnings('ignore',
                                category=tables.NaturalNameWarning)


class HDFStore(pandas.HDFStore):
    '''
    pandas.HDFStore() with improved defaults.
    '''
    def __init__(self, path, *args, key='df', **kwds):
        self.key = key
        super().__init__(path, *args, **kwds)

    def get(self, key=None):
        if key is None:
            key = self.key
        return super().get(key)

    def select(self, *args, key=None, **kwds):
        if key is None:
            key = self.key
        return super().select(key, *args, **kwds)

    def put(self, value, format='table', append=True, *args, key=None, **kwds):
        if key is None:
            key = self.key
        with _catch_natural_name_warnings():
            return super().put(key, value,
                               format=format, append=append,
                               *args, **kwds)

    def append(self, value, format='table', append=True, *args, key=None,
               **kwds):
        if key is None:
            key = self.key
        with _catch_natural_name_warnings():
            return super().append(key, value, format=format, append=append,
                                  *args, **kwds)

    def get_index(self, *args, key=None, **kwds):
        # For speed, don't read any columns.
        df = self.select(*args, key=key, columns=[], **kwds)
        return df.index

    def get_index_names(self, *args, key=None, **kwds):
        # For speed, don't read any rows or columns.
        df = self.select(*args, key=key, stop=0, columns=[], **kwds)
        return df.index.names

    def get_columns(self, *args, key=None, **kwds):
        # For speed, don't read any rows.
        df = self.select(*args, key=key, stop=0, **kwds)
        return df.columns

    def repack(self):
        self.close()
        repack(self._path)


def load(filename, key='df', **kwds):
    return pandas.read_hdf(filename, key, **kwds)


def dump(df, filename, key='df', mode='w', format='table', append=True,
         **kwds):
    with _catch_natural_name_warnings():
        df.to_hdf(filename, key, mode=mode, format=format, append=append,
                  **kwds)
    repack(filename)
