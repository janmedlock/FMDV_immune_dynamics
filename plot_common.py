import itertools
import os.path

import matplotlib.collections
import numpy
import pandas

import h5
import run_common


t_name = 'time (y)'


def _downsample_group(group, t, not_t_names):
    # Only keep time index.
    group = group.reset_index(not_t_names, drop=True)
    # Only interpolate between start and extinction.
    mask = ((t >= group.index.min()) & (t <= group.index.max()))
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def downsample(filename, t_min=0, t_max=10, t_step=1/365):
    t = numpy.arange(t_min, t_max, t_step)
    base, ext = os.path.splitext(filename)
    filename_ds = base + '_downsampled' + ext
    with h5.HDFStore(filename, mode='r') as store_in, \
         h5.HDFStore(filename_ds, mode='w') as store_out:
        index_names = store_in.get_index_names()
        not_t_names = [n for n in index_names if n != t_name]
        remainder = None
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
            if len(data_ds) > 0:
                store_out.put(data_ds,
                              min_itemsize=run_common._min_itemsize)


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)
