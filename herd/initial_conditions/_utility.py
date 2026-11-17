import numpy


def recarray_fromkwds(**kwds):
    '''Build a `numpy.recarray()` from `kwds`.'''
    return numpy.rec.fromarrays(
        numpy.broadcast_arrays(*kwds.values()),
        dtype=[(k, float) for k in kwds],
    )
