'''Common code to work with HDF5 files.'''

import pathlib
import shutil
import stat
import subprocess
import warnings

import pandas


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


class HDFStore(pandas.HDFStore):
    '''
    pandas.HDFStore() with improved defaults.
    '''
    def __init__(self, path,
                 complib=_COMPLIB, complevel=_COMPLEVEL,
                 fletcher32=_FLETCHER32, **kwargs):
        super().__init__(path, complib=complib, complevel=complevel,
                         fletcher32=fletcher32, **kwargs)

    def put(self, key, value, format='table', append=True, **kwargs):
        if len(value) == 0:
            return
        return super().put(key, value,
                           format=format, append=append,
                           **kwargs)

    def append(self, key, value, format='table', append=True, **kwargs):
        return super().append(key, value,
                              format=format, append=append,
                              **kwargs)

    def repack(self, **kwargs):
        self.close()
        defaults = {
            'complib': self._complib,
            'complevel': self._complevel,
            'fletcher32': self._fletcher32,
        }
        kwargs = defaults | kwargs
        repack(self._path, **kwargs)

    _WRITEABLE = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

    def set_read_only(self):
        '''Set as read only.'''
        path = pathlib.Path(self._path)
        perms = stat.S_IMODE(
            path.stat().st_mode
        )
        return path.chmod(
            perms & ~self._WRITEABLE
        )


def load(path, key, *args, mode='r', **kwargs):
    with HDFStore(path, mode=mode) as store:
        return store.select(key, *args, **kwargs)


def dump(path, key, df, mode='a', **kwargs):
    with HDFStore(path, mode=mode) as store:
        store.put(key, df, **kwargs)
        store.repack()
