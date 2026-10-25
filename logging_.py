def get_prefix(**kwds):
    return ', '.join(f'{key}={val}'
                     for (key, val) in kwds.items())


def append_to_prefix(logging_prefix=None, **kwds):
    if len(kwds) == 0:
        return logging_prefix
    to_append = get_prefix(**kwds)
    if logging_prefix is None:
        return to_append
    return f'{logging_prefix}, {to_append}'
