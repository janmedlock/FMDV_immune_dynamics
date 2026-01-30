'''`multiprocessing.Pool()` for nested functions.'''

import multiprocessing


class _PoolProxy:
    '''Context manager to use an existing `multiprocessing.Pool()`,
    without calling `multiprocessing.Pool.__exit__()` on exit.'''

    def __init__(self, pool, **kwds):
        assert len(kwds) == 0
        self.pool = pool

    def __enter__(self):
        return self.pool

    def __exit__(self, *args):
        '''Exit without calling `self.pool.close()`.'''

    def __getattr__(self, attr):
        '''Get `attr` from `self.pool`.'''
        return getattr(self.pool, attr)


def Pool(pool=None, **kwds):  # pylint: disable=invalid-name
    '''Factory to a new `multiprocessing.Pool()` or proxy an existing
    one. It is intended for nested functions that use a shared
    `multiprocessing.Pool()` with multiple entry points, so the
    `multiprocessing.Pool()` can be created in the first function
    called and passed to subsequent functions.'''
    if pool is None:
        return multiprocessing.Pool(**kwds)
    return _PoolProxy(pool, **kwds)
