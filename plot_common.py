import os.path

import matplotlib.collections
import matplotlib.ticker
import numpy
import pandas

import h5


t_name = 'time (y)'


def _build_downsample_group(group, t, by):
    # Only keep time index.
    group = group.reset_index(by, drop=True)
    # Only interpolate between start and extinction.
    mask = ((t >= group.index.min()) & (t <= group.index.max()))
    # Interpolate from the closest point <= t.
    return group.reindex(t[mask], method='ffill')


def build_downsample(filename_in, t_min=0, t_max=10, t_step=1/365):
    t = numpy.arange(t_min, t_max, t_step)
    base, ext = os.path.splitext(filename_in)
    filename_out = base + '_downsampled' + ext
    with h5.HDFStore(filename_in, mode='r') as store_in, \
         h5.HDFStore(filename_out, mode='w') as store_out:
        by = [n for n in store_in.get_index_names() if n != t_name]
        for (ix, group) in store_in.groupby(by):
            downsample = _build_downsample_group(group, t, by)
            # Append `ix` to the index levels.
            downsample = pandas.concat({ix: downsample},
                                       names=by, copy=False)
            store_out.put(downsample, dropna=True, index=False)
        store_out.create_table_index()
        store_out.repack()


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)


class PercentFormatter(matplotlib.ticker.Formatter):
    def __call__(self, x, pos=None):
        return '{:g}%'.format(100 * x)


# Erin's colors.
SAT_colors = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}
