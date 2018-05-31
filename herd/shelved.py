from functools import wraps
from inspect import getfile
from os.path import splitext
from pickle import HIGHEST_PROTOCOL
import shelve
from multiprocessing import Lock


class Shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''

    def __init__(self, *parameters_to_keep):
        self.parameters_to_keep = parameters_to_keep

    def get_key(self, parameters):
        '''Build the key string from `parameters`.'''
        clsname = '{}.{}'.format(parameters.__module__,
                                 parameters.__class__.__name__)
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self.parameters_to_keep)
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @staticmethod
    def get_func_name(func):
        '''Get `func.__module__`.`func.__name__`.'''
        return '{}.{}()'.format(func.__module__, func.__name__)

    def cached_call(self, cache_file, write_lock,
                    func, parameters, *args, **kwargs):
        '''Memoized version of `func`.'''
        key = self.get_key(parameters)
        try:
            with shelve.open(cache_file, protocol=HIGHEST_PROTOCOL) as shelf:
                val = shelf[key]
        except (KeyError, ValueError, TypeError):
            func_name = self.get_func_name(func)
            print('{} not in {} cache.  Computing...'.format(key, func_name))
            val = func(parameters, *args, **kwargs)
            print('\tFinished computing {}.'.format(func_name))
            with write_lock, \
                 shelve.open(cache_file, protocol=HIGHEST_PROTOCOL) as shelf:
                shelf[key] = val
        return val

    @staticmethod
    def get_cache_file(func):
        '''Get the name of the cache file.
        It is in the same directory as the caller `func`
        and named it 'module.func.db'.'''
        root, _ = splitext(getfile(func))
        return '{}.{}'.format(root, func.__name__)

    def __call__(self, func):
        '''Build the wrapper memoization function.'''
        cache_file = self.get_cache_file(func)
        write_lock = Lock()
        @wraps(func)
        def wrapped(parameters, *args, **kwargs):
            return self.cached_call(cache_file, write_lock, func,
                                    parameters, *args, **kwargs)
        # Give access to the original function, too.
        wrapped.func = func
        return wrapped
