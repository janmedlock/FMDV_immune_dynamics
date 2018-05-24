import functools
import inspect
import os.path
import pickle
import shelve


class Shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''
    def __init__(self, *parameters_to_keep):
        self._parameters_to_keep = parameters_to_keep

    def get_key(self, parameters):
        clsname = '{}.{}'.format(parameters.__module__,
                                 parameters.__class__.__name__)
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self._parameters_to_keep)
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @staticmethod
    def get_cache_file(func):
        # Put the cache file in the same directory as the caller
        # and name it 'module.func.db'.
        root, _ = os.path.splitext(inspect.getfile(func))
        return '{}.{}'.format(root, func.__name__)

    def cached_call(self, func, parameters, *args, **kwargs):
        with shelve.open(self.get_cache_file(func),
                         protocol=pickle.HIGHEST_PROTOCOL) as shelf:
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
        @functools.wraps(func)
        def wrapped(parameters, *args, **kwargs):
            return self.cached_call(func, parameters, *args, **kwargs)
        # Give access to the original function, too.
        wrapped.func = func
        return wrapped
