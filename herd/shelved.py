from contextlib import contextmanager
from functools import wraps
from inspect import getfile
from os.path import splitext
from pickle import HIGHEST_PROTOCOL
import shelve
from threading import Lock


class Shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''
    def __init__(self, *parameters_to_keep):
        self.parameters_to_keep = parameters_to_keep
        self.lock = {}

    def get_key(self, parameters):
        '''Build the key string from `parameters`.'''
        clsname = '{}.{}'.format(parameters.__module__,
                                 parameters.__class__.__name__)
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self.parameters_to_keep)
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @staticmethod
    def get_cache_file(func):
        '''Get the name of the cache file.
        It is in the same directory as the caller `func`
        and named it 'module.func.db'.'''
        root, _ = splitext(getfile(func))
        return '{}.{}'.format(root, func.__name__)

    @contextmanager
    def get_shelf(self, func):
        '''Acquire the lock and open the shelf file.'''
        f = self.get_cache_file(func)
        print('Acquiring lock.')
        with self.lock[f], shelve.open(f, protocol=HIGHEST_PROTOCOL) as shelf:
            print('Lock acquired.')
            yield shelf
        print('Lock released.')

    def cached_call(self, func, parameters, *args, **kwargs):
        '''Memoized version of `func`.'''
        with self.get_shelf(func) as shelf:
            key = self.get_key(parameters)
            try:
                val = shelf[key]
            except (KeyError, ValueError, TypeError):
                func_name = '{}.{}()'.format(func.__module__, func.__name__)
                print('{} not in {} cache.  Computing...'.format(key, func_name))
                val = shelf[key] = func(parameters, *args, **kwargs)
                print('\tFinished computing {}.'.format(func_name))
        return val

    def __call__(self, func):
        '''Build the wrapper memoization function.'''
        self.lock[self.get_cache_file(func)] = Lock()
        @wraps(func)
        def wrapped(parameters, *args, **kwargs):
            return self.cached_call(func, parameters, *args, **kwargs)
        # Give access to the original function, too.
        wrapped.func = func
        return wrapped
