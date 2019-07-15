#!/usr/bin/python3

import itertools
import os
import subprocess
import warnings

import numpy
import pandas
import tables


def repack(path):
    '''
    Use `ptrepack` to compress the HDF file.
    '''
    # TODO: Call from Python: something in tables.scripts.ptrepack.
    tmp = path + '.repack'
    try:
        subprocess.run(['ptrepack', '--chunkshape=auto',
                        '--propindexes', '--complevel=6',
                        '--complib=blosc:zstd', '--fletcher32=1',
                        path, tmp],
                       check=True)
    except subprocess.CalledProcessError:
        os.remove(tmp)
    os.rename(tmp, path)


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
    def __init__(self, path, key='df',
                 complevel=6, complib='blosc:zstd', fletcher32=True,
                 **kwargs):
        self.key = key
        super().__init__(path, complevel=complevel,
                         complib=complib, fletcher32=fletcher32, **kwargs)

    def get(self, key=None):
        if key is None:
            key = self.key
        return super().get(key)

    def select(self, *args, key=None, **kwargs):
        if key is None:
            key = self.key
        return super().select(key, *args, **kwargs)

    def put(self, value, key=None, format='table', append=True, **kwargs):
        if len(value) == 0:
            return
        if key is None:
            key = self.key
        with _catch_natural_name_warnings():
            return super().put(key, value,
                               format=format, append=append,
                               **kwargs)

    def append(self, value, key=None, format='table', append=True, **kwargs):
        if key is None:
            key = self.key
        with _catch_natural_name_warnings():
            return super().append(key, value,
                                  format=format, append=append,
                                  **kwargs)

    def create_table_index(self, key=None, **kwargs):
        if key is None:
            key = self.key
        with _catch_natural_name_warnings():
            return super().create_table_index(key, **kwargs)

    def get_index(self, *args, key=None, iterator=False, **kwargs):
        # For speed, don't read any columns.
        df = self.select(*args, key=key, iterator=iterator, columns=[],
                         **kwargs)
        if iterator:
            return (chunk.index for chunk in df)
        else:
            return df.index

    def get_index_names(self, *args, key=None, iterator=False, **kwargs):
        if iterator:
            raise NotImplementedError
        # For speed, don't read any rows or columns.
        df = self.select(*args, key=key, iterator=iterator, stop=0, columns=[],
                         **kwargs)
        return df.index.names

    def get_columns(self, *args, key=None, iterator=False, **kwargs):
        if iterator:
            raise NotImplementedError
        # For speed, don't read any rows.
        df = self.select(*args, key=key, iterator=iterator, stop=0, **kwargs)
        return df.columns

    def groupby(self, by, *args, key=None, debug=False, **kwargs):
        # Iterate through `by` values,
        # then select() each chunk of data.
        kwargs_ = kwargs.copy()
        for k in ('columns', 'iterator'):
            kwargs_.pop(k, None)
        by_iterator = iter(self.select(*args, key=key, columns=by,
                                       iterator=True, **kwargs_))
        # `self.select(iterator=True)` has a chunk size that generally
        # doesn't not align with groups defined by `.groupby(by)`, so
        # we carry the last group from one chunk over to beginning of
        # the next chunk.
        by_carryover = None
        is_last_by_chunk = False
        where_base = kwargs.pop('where', None)
        if debug:
            by_seen = set()
        while True:
            try:
                by_chunk = next(by_iterator)
            except StopIteration:
                # No new data, but we need to handle `carryover`.
                by_chunk = None
                is_last_by_chunk = True
            # Handle carryover.
            try:
                by_values = pandas.concat((by_carryover, by_chunk),
                                          copy=False)
            except ValueError:
                # No data.  `pandas.concat((None, None))`.
                break
            # Iterate through the `by` values.
            by_grouper = by_values.groupby(by)
            by_group_iterator = iter(by_grouper)
            # If this is not the final chunk,
            # carry the last group over to the next chunk.
            n_groups = len(by_grouper)
            stop = n_groups if is_last_by_chunk else (n_groups - 1)
            for (by_value, _) in itertools.islice(by_grouper, stop):
                if debug:
                    assert by_value not in by_seen
                    by_seen.add(by_value)
                where = ' & '.join(f'{k}={v}' for k, v in zip(by, by_value))
                print(where.replace(' & ', ', '))
                if where_base is not None:
                    where = where_base + ' & ' + where
                group = self.select(*args, key=key, where=where, **kwargs)
                yield (by_value, group)
            if not is_last_by_chunk:
                # Carry the last group over to the next chunk.
                (_, by_carryover) = next(by_group_iterator)
            elif debug:
                try: next(by_group_iterator)
                except StopIteration: pass
                else: assert False

    def repack(self):
        self.close()
        repack(self._path)


def load(path, *args, **kwargs):
    with HDFStore(path, mode='r') as store:
        return store.select(*args, **kwargs)


def dump(df, path, mode='a', **kwargs):
    with HDFStore(path, mode=mode) as store:
        store.put(df, **kwargs)
        store.repack()
