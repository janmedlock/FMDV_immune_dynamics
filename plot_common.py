import itertools

import matplotlib.collections
import numpy
import pandas

import h5


t_name = 'time (y)'


def _downsample_group(group, t, not_t_names):
    # Only keep time index.
    group = group.reset_index(not_t_names, drop=True)
    # Only interpolate between start and extinction.
    mask = ((t >= group.index.min()) & (t <= group.index.max()))
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def downsample(filename):
    dt = 1 / 365  # Daily timesteps.
    with h5.HDFStore(filename, mode='r') as store_in:
        index_names = store_in.get_index_names()
        not_t_names = [n for n in index_names if n != t_name]
        print('Finding t limits.')
        t_min = numpy.inf
        t_max = - numpy.inf
        for chunk in store_in.get_index(iterator=True):
            not_t_values = (chunk[0][index_names.index(n)]
                            for n in not_t_names)
            print(', '.join(f'{k}={v}'
                            for k, v in zip(not_t_names, not_t_values)))
            t = chunk.get_level_values(t_name)
            t_min = min(t_min, t.min())
            t_max = max(t_max, t.max())
        t = numpy.arange(t_min, t_max, dt)
        remainder = None
        base, ext = os.path.splitext(filename)
        filename_ds = base + '_downsampled' + ext
        with h5.HDFStore(filename, mode='w') as store_out:
            # One more empty chunk at the end.
            for chunk in itertools.chain(store_in.select(iterator=True), [None]):
                data = pandas.concat([remainder, chunk], copy=False)
                grouper = data.groupby(not_t_names)
                data_ds = {}
                for (i, (ix, group)) in enumerate(grouper):
                    if (chunk is None) or (i < (len(grouper) - 1)):
                        print(', '.join(f'{k}={v}'
                                        for k, v in zip(not_t_names, ix)))
                        data_ds[ix] = _downsample_group(group, t, not_t_names)
                    else:
                        # The last group might be continued in the next chunk.
                        remainder = group
                data_ds = pandas.concat(data_ds, copy=False)
                data_ds.rename_axis(not_t_names + [t_name],
                                    inplace=True, copy=False)
                data_ds.dropna(axis=0, inplace=True)
                store_out.put(data_ds)


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)
