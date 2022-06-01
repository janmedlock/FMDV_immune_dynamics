'''Common code to work with HDF5 files.'''


import itertools
import pathlib
import subprocess
import warnings

import pandas
import tables


def repack(path):
    '''
    Use `ptrepack` to compress the HDF file.
    '''
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    path_repack = path.with_suffix(path.suffix + '.repack')
    try:
        subprocess.run(['ptrepack', '--chunkshape=auto',
                        '--propindexes', '--complevel=6',
                        '--complib=blosc:zstd', '--fletcher32=1',
                        path, path_repack],
                       check=True)
    except subprocess.CalledProcessError as exc:
        path_repack.unlink()
        raise RuntimeError('ptrepack failed!') from exc
    else:
        path_repack.rename(path)


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

    def groupby(self, by, *args, key=None, debug=True, **kwargs):
        # `self.select(iterator=True)` has a chunk size that generally
        # does not align with groups defined by `.groupby(by)`, so
        # we carry the last group from one chunk over to beginning of
        # the next chunk.
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
                print(', '.join(f'{k}={v}' for k, v in zip(by, idx)))
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
