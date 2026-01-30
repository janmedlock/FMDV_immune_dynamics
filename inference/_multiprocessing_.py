'''Utilities for `multiprocessing.Pool()`.'''

import contextlib
import multiprocessing


class Pool:
    '''Make a new `multiprocessing.Pool()` or proxy an existing one.'''
    def __init__(self, pool=None, **kwds):
        if pool is None:
            self.pool = multiprocessing.Pool(**kwds)
        else:
            assert len(kwds) == 0
            self.pool = contextlib.nullcontext(pool)

    def __enter__(self):
        return self.pool.__enter__()

    def __exit__(self, *args):
        return self.pool.__exit__(*args)
