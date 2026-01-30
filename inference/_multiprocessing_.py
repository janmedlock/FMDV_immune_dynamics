'''Utilities for `multiprocessing.Pool()`.'''

import contextlib
import multiprocessing


def Pool(pool=None, **kwds):
    '''Make a new `multiprocessing.Pool()` or proxy an existing one.'''
    if pool is None:
        return multiprocessing.Pool(**kwds)
    # Return a context manager that returns `pool` on enter and does
    # nothing on exit.
    assert len(kwds) == 0
    return contextlib.nullcontext(pool)
