'''Utilities for `multiprocessing.Pool()`.'''

import multiprocessing


class _PoolProxy:
    def __init__(self, pool, **kwds):
        assert len(kwds) == 0
        self.pool = pool

    def __enter__(self):
        return self.pool

    def __exit__(self, *args):
        pass

    def __getattr__(self, attr):
        return getattr(self.pool, attr)


def Pool(pool=None, **kwds):
    '''Make a new `multiprocessing.Pool()` or proxy an existing one.'''
    if pool is None:
        return multiprocessing.Pool(**kwds)
    return _PoolProxy(pool, **kwds)
