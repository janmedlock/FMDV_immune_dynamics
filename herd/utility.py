'''Utility functions.'''


import numpy


def arange(*args, dtype=None, endpoint=False):
    '''`numpy.arange([start, ] stop, [step, ] dtype=None)`
    that can also optionally include the endpoint `stop`.'''
    if len(args) == 1:
        stop, = args
    elif len(args) == 2:
        start, stop = args
    elif len(args) == 3:
        start, stop, step = args
    elif len(args) == 4:
        if dtype is not None:
            raise TypeError('`dtype` is specified as both positional'
                            ' and keyword arguments.')
        start, stop, step, dtype = args
    else:
        raise TypeError('_arange() requires 1–4 positional arguments.')
    val = numpy.arange(start, stop, step, dtype=dtype)
    if endpoint:
        # `stop` could already be at the end of `val`
        # due to roundoff errors.
        stop = val.dtype.type(stop)
        if (((step > 0) and (val[-1] < stop))
                or ((step < 0) and (val[-1] > stop))):
            val = numpy.hstack((val, stop))
    return val


def is_increasing(arr, strict=False, axis=-1):
    '''Check whether `arr` is increasing along `axis`.'''
    if strict:
        test = numpy.greater
    else:
        test = numpy.greater_equal
    return test(numpy.diff(arr, axis=axis), 0).all(axis=axis)
