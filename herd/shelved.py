from functools import update_wrapper
from inspect import getfile
from os.path import splitext
from pickle import HIGHEST_PROTOCOL
import shelve
from multiprocessing import Lock


def _get_cache_file(func):
    '''Get the name of the cache file.
    It is in the same directory as the caller `func`
    and named 'module.func.db'.'''
    root, _ = splitext(getfile(func))
    return '{}.{}'.format(root, func.__name__)


class _ShelvedWrapper:
    '''The function wrapper.'''
    def __init__(self, parameters_to_keep, func):
        self.parameters_to_keep = parameters_to_keep
        self.func = func
        self.cache_file = _get_cache_file(self.func)
        self.write_lock = Lock()
        update_wrapper(self, func)

    @property
    def func_name(self):
        return '{}.{}()'.format(self.func.__module__, self.func.__name__)

    def get_key(self, parameters):
        '''Build the key string from `parameters`.'''
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self.parameters_to_keep)
        return '{{{}}}'.format(', '.join(paramreprs))

    def _open(self):
        return shelve.open(self.cache_file, protocol=HIGHEST_PROTOCOL)

    def open(self, write=False):
        if write:
            with self.write_lock:
                return self._open()
        else:
            return self._open()

    def __call__(self, parameters, *args, **kwargs):
        '''Memoized version of `func`.'''
        key = self.get_key(parameters)
        try:
            with self.open() as shelf:
                val = shelf[key]
        except (KeyError, ValueError, TypeError):
            print('{} not in {} cache.  Computing...'.format(key, self.func_name))
            val = self.func(parameters, *args, **kwargs)
            print('\tFinished computing {}.'.format(self.func_name))
            with self.open(write=True) as shelf:
                shelf[key] = val
        return val


class Shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''
    def __init__(self, *parameters_to_keep):
        self.parameters_to_keep = parameters_to_keep

    def __call__(self, func):
        '''Build the wrapper memoization instance.'''
        return _ShelvedWrapper(self.parameters_to_keep, func)
