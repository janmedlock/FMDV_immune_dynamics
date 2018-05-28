from collections import defaultdict
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
    class Locks(defaultdict):
        '''A `collections.defaultdict()` inside another `defaultdict()`.
        The inner `defaultdict()` returns a `threading.Lock()`.'''
        class LocksInner(defaultdict):
            '''A `collections.defaultdict()` that returns a `threading.Lock()`.'''
            def __init__(self, *args, **kwargs):
                super().__init__(Lock, *args, **kwargs)
        def __init__(self, *args, **kwargs):
            super().__init__(self.LocksInner, *args, **kwargs)

    def __init__(self, *parameters_to_keep):
        self.parameters_to_keep = parameters_to_keep
        self.lock = self.Locks()

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

    @staticmethod
    def get_func_name(func):
        '''Get `func.__module__`.`func.__name__`.'''
        return '{}.{}()'.format(func.__module__, func.__name__)

    def cached_call(self, func, parameters, *args, **kwargs):
        '''Memoized version of `func`.'''
        cache_file = self.get_cache_file(func)
        with shelve.open(cache_file, protocol=HIGHEST_PROTOCOL) as shelf:
            key = self.get_key(parameters)
            with self.lock[cache_file][key]:
                try:
                    val = shelf[key]
                except (KeyError, ValueError, TypeError):
                    func_name = self.get_func_name(func)
                    print('{} not in {} cache.  Computing...'.format(key,
                                                                     func_name))
                    val = shelf[key] = func(parameters, *args, **kwargs)
                    print('\tFinished computing {}.'.format(func_name))
        return val

    def __call__(self, func):
        '''Build the wrapper memoization function.'''
        @wraps(func)
        def wrapped(parameters, *args, **kwargs):
            return self.cached_call(func, parameters, *args, **kwargs)
        # Give access to the original function, too.
        wrapped.func = func
        return wrapped
