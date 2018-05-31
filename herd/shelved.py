from collections.abc import MutableMapping
from contextlib import contextmanager, ExitStack
from functools import update_wrapper
from inspect import getfile
from os.path import splitext
from pickle import HIGHEST_PROTOCOL
import shelve
from multiprocessing import Manager, cpu_count


def _get_cache_file(func):
    '''Get the name of the cache file.
    It is in the same directory as the caller `func`
    and named 'module.func.db'.'''
    root, _ = splitext(getfile(func))
    return '{}.{}'.format(root, func.__name__)


class _ShelvedWrapper(MutableMapping):
    '''The function wrapper.'''
    def __init__(self, parameters_to_keep, func):
        self.parameters_to_keep = parameters_to_keep
        self.func = func
        self.cache_file = _get_cache_file(self.func)
        manager = Manager()
        # To lock the shelf for writing.
        self.write_lock = manager.Lock()
        # Per-key locks to keep multiple processes from computing
        # with the same key.
        self.compute_locks = manager.dict()
        # Processes can't create Lock()'s, so build a pool to reuse.
        self.compute_lock_pool = manager.list([manager.Lock()
                                               for _ in range(cpu_count())])
        update_wrapper(self, func)

    def open(self, write=False):
        # `contextlib.ExitStack()` is a dummy context manager.
        with self.write_lock if write else ExitStack():
            return shelve.open(self.cache_file, protocol=HIGHEST_PROTOCOL)

    def __getitem__(self, parameters):
        with self.open() as shelf:
            return shelf[self.key(parameters)]

    def __setitem__(self, parameters, value):
        with self.open(write=True) as shelf:
            shelf[self.key(parameters)] = value

    def __delitem__(self, parameters):
        with self.open(write=True) as shelf:
            del shelf[self.key(parameters)]

    def __len__(self):
        with self.open() as shelf:
            return len(shelf)

    def __iter__(self):
        with self.open() as shelf:
            return iter(shelf)

    def key(self, parameters):
        '''Build the key string from `parameters`.'''
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self.parameters_to_keep)
        return '{' + ', '.join(paramreprs) + '}'

    def get_func_name(self):
        return '{}.{}()'.format(self.func.__module__, self.func.__name__)

    @contextmanager
    def get_compute_lock(self, parameters):
        '''Get a per-key lock to keep multiple processes
        from computing the value for the same key.'''
        if (parameters in self):
            # Don't need a lock.
            yield
        else:
            key = self.key(parameters)
            # Is this process going to compute the value
            # or is another process already working on it?
            new_lock = (key not in self.compute_locks)
            if new_lock:
                self.compute_locks[key] = self.compute_lock_pool.pop()
            with self.compute_locks[key]:
                yield
            if new_lock:
                self.compute_lock_pool.append(self.compute_locks.pop(key))

    def __call__(self, parameters, *args, **kwargs):
        '''Memoized version of `func`.'''
        with self.get_compute_lock(parameters):
            try:
                val = self[parameters]
            except (KeyError, ValueError, TypeError):
                func_name = self.get_func_name()
                print('{} not in {} cache.  Computing...'.format(
                    self.key(parameters), func_name))
                val = self.func(parameters, *args, **kwargs)
                print('\tFinished computing {}.'.format(func_name))
                self[parameters] = val
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
