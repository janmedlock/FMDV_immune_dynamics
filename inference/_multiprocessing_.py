'''Utilities for `multiprocessing.Pool()`.'''

import contextlib
import multiprocessing


def Pool(pool=None, **kwds):
    '''Make a new `multiprocessing.Pool()` or proxy an existing one.'''
    if pool is None:
        return multiprocessing.Pool(**kwds)
    assert len(kwds) == 0
    return contextlib.nullcontext(pool)
