'''Common code to work with HDF5 files.'''


import collections.abc
import itertools
import pathlib
import shutil
import subprocess
import warnings

import pandas
import tables


# Defaults
_COMPLIB = 'blosc:zstd'
_COMPLEVEL = 6
_FLETCHER32 = True


def repack(path, complib=_COMPLIB, complevel=_COMPLEVEL,
           fletcher32=_FLETCHER32, chunkshape='auto'):
    '''
    Use `ptrepack` to compress the HDF file.
    '''
    ptrepack = shutil.which('ptrepack')
    if ptrepack is None:
        warnings.warn('ptrepack missing!')
        return
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    path_repack = path.with_stem(path.stem + '_repack')
    cmd = [
        ptrepack,
        f'--complib={complib}',
        f'--complevel={complevel}',
        f'--fletcher32={fletcher32:d}',  # Converted to integer.
        f'--chunkshape={chunkshape}',
        '--propindexes',
        path,
        path_repack
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        try:
            path_repack.unlink()
        except FileNotFoundError:
            pass
        warnings.warn('ptrepack failed!')
    else:
        path_repack.rename(path)


def sort_index(path, level=None, **kwargs):
    '''Very slow code to sort the index of an HDF5 file.'''
    path_sorted = path.with_stem(path.stem + '_sorted')
    with (HDFStore(path, mode='r') as store,
          HDFStore(path_sorted) as store_sorted):
        if level is None:
            level = store.get_index_names()
        elif not isinstance(level, collections.abc.Sequence):
            try:
                level = tuple(level)
            except TypeError:
                level = (level,)
        index = pandas.MultiIndex.from_tuples((), names=level)
        for chunk in store.get_index(iterator=True):
            index = index.union(chunk.drop_duplicates())
        index = index.sort_values()
        assert index.is_monotonic_increasing
        assert all(level.is_monotonic_increasing
                   for level in index.levels)
        for val in index:
            where = ' & '.join(f'{key}={val}'
                               for (key, val) in zip(index.names,
                                                     _as_sequence(val)))
            print(where)
            dfr = store.select(where=where)
            store_sorted.put(dfr, index=False)
        store_sorted.create_table_index()
        store_sorted.repack(**kwargs)
    path_sorted.rename(path)


def _as_sequence(val):
    '''If `val` is not a non-string sequence, wrap it in a 1-element
    list, otherwise just return `val`.'''
    if (isinstance(val, collections.abc.Sequence)
        and not isinstance(val, str)):
        return val
    # Use a list because `pandas` indexing works better with lists.
    return [val]


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
                 complib=_COMPLIB, complevel=_COMPLEVEL,
                 fletcher32=_FLETCHER32, **kwargs):
        self.key = key
        super().__init__(path, complib=complib, complevel=complevel,
                         fletcher32=fletcher32, **kwargs)

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

    def groupby(self, by, *args, key=None, debug=True, **kwargs):
        # `self.select(iterator=True)` has a chunk size that generally
        # does not align with groups defined by `.groupby(by)`, so
        # we carry the last group from one chunk over to beginning of
        # the next chunk.
        by = _as_sequence(by)
        chunk_iterator = iter(self.select(*args, key=key,
                                          iterator=True, **kwargs))
        carryover = None
        is_last_chunk = False
        while True:
            try:
                chunk = next(chunk_iterator)
            except StopIteration:
                # No new data, but we need to handle `carryover`.
                chunk = None
                is_last_chunk = True
            # Append the last group from the previous chunk to this chunk.
            try:
                data = pandas.concat((carryover, chunk), copy=False)
            except ValueError:
                # No data.  `pandas.concat((None, None))`.
                break
            grouper = data.groupby(by)
            group_iterator = iter(grouper)
            n_groups = len(grouper)
            # If this is not the final chunk, carry the last group over to
            # the next chunk.
            stop = n_groups if is_last_chunk else (n_groups - 1)
            for (idx, group) in itertools.islice(group_iterator, stop):
                print(', '.join(f'{k}={v}'
                                for (k, v) in zip(by, _as_sequence(idx))))
                yield (idx, group)
            if is_last_chunk:
                if debug:
                    try:
                        next(group_iterator)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError('group_iterator is not empty!')
                break
            else:
                # Carry the last group over to the next chunk.
                (idx, carryover) = next(group_iterator)

    def repack(self, **kwargs):
        self.close()
        defaults = dict(complib=self._complib,
                        complevel=self._complevel,
                        fletcher32=self._fletcher32)
        kwargs = defaults | kwargs
        repack(self._path, **kwargs)


def load(path, *args, **kwargs):
    with HDFStore(path, mode='r') as store:
        return store.select(*args, **kwargs)


def dump(df, path, mode='a', **kwargs):
    with HDFStore(path, mode=mode) as store:
        store.put(df, **kwargs)
        store.repack()
