'''Common code for plotting.'''

import itertools

import astropy.units
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import numpy
import statsmodels.nonparametric.api

import common


rc = {
    'font.family': 'serif',
    'font.serif': 'Liberation Serif',  # Substitute for 'Times New Roman'
    'figure.constrained_layout.use': True,
    'figure.dpi': 300,
}

_TICK_LABELSIZE = 8
rc_text_small = {
    'font.size': 7.5,  # Minimium for Proc B.
    'xtick.labelsize': _TICK_LABELSIZE,
    'ytick.labelsize': _TICK_LABELSIZE,
    'axes.labelsize': 9,
    'axes.titlesize': 11,
}

_mm_to_inch = astropy.units.mm.to(astropy.units.imperial.inch)

WIDTH_MAXIMUM = {
    'single_column': 84 * _mm_to_inch,
    'double_column': 175 * _mm_to_inch,
}

HEIGHT_MAXIMUM = 250 * _mm_to_inch


# Erin's colors.
SAT_COLORS = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba'
}

TIME_LABEL = f'Time ({common.TIME_UNIT})'


class SupplementalMaterials:
    '''Supplemental Materials style.'''

    rc = {
        'font.family': 'serif',
        # Substitute for 'Computer Modern Roman'
        'font.serif': 'Latin Modern Math',
    }

    WIDTH_MAXIMUM = 0.7 * 8.5  # inch


def set_violins_linewidth(ax, lw):
    for col in ax.collections:
        if isinstance(col, matplotlib.collections.PolyCollection):
            col.set_linewidth(0)


def get_density(endog, times):
    # Avoid errors if endog is empty.
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=0)
        return kde.evaluate(times)
    return numpy.zeros_like(times)


def kdeplot(endog, ax=None, shade=False, cut=0, **kwds):
    if ax is None:
        ax = matplotlib.pyplot.gca()
    endog = endog.dropna()
    if len(endog) > 0:
        kde = statsmodels.nonparametric.api.KDEUnivariate(endog)
        kde.fit(cut=cut)
        x = numpy.linspace(kde.support.min(), kde.support.max(), 301)
        y = kde.evaluate(x)
    else:
        x, y = [], []
    line, = ax.plot(x, y, **kwds)
    if shade:
        shade_kws = {
            'facecolor': kwds.get('facecolor', line.get_color()),
            'alpha': kwds.get('alpha', 0.25),
            'clip_on': kwds.get('clip_on', True),
            'zorder': kwds.get('zorder', 1),
        }
        ax.fill_between(x, 0, y, **shade_kws)
    return ax


def get_cmap_SAT(SAT):
    '''White to `SAT_COLORS[SAT]`.'''
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'name', ['white', SAT_COLORS[SAT]])


def legend_multicolumn(obj, handles, labels, ncol, **kwds):
    '''Make a multicolumn legend.'''
    def reorder(items, ncol):
        return list(
            itertools.chain.from_iterable(
                items[i::ncol] for i in range(ncol)
            )
        )
    return obj.legend(reorder(handles, ncol),
                      reorder(labels, ncol),
                      ncol=ncol, **kwds)


def get_state_label(state):
    '''Make a plot label for `state`.'''
    return state.replace('_', ' ').capitalize()
