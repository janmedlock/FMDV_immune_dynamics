'''Common code for plotting.'''

import itertools

import astropy.units
import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.scale
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

_MM_TO_INCH = astropy.units.mm.to(astropy.units.imperial.inch)

WIDTH_MAXIMUM = {
    'single_column': 84 * _MM_TO_INCH,
    'double_column': 175 * _MM_TO_INCH,
}

HEIGHT_MAXIMUM = 250 * _MM_TO_INCH


class SupplementalMaterials:
    '''Supplemental Materials style.'''

    rc = {
        'font.family': 'serif',
        # Substitute for 'Computer Modern Roman'
        'font.serif': 'Latin Modern Math',
    }

    WIDTH_MAXIMUM = 0.7 * 8.5  # inch


# Erin's colors.
SAT_COLORS = {
    1: '#2271b5',
    2: '#ef3b2c',
    3: '#807dba',
}

ALPHA = 0.8

TIME_LABEL = f'Time ({common.TIME_UNIT})'

PERSISTENCE_LABEL = (
    f'{common.TIME_MAX}-{common.TIME_UNIT} FMDV persistence'
)


@matplotlib.colors.make_norm_from_scale(matplotlib.scale.LogitScale)
class LogitNorm(matplotlib.colors.Normalize):
    '''Logit norm.'''


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


def add_part_labels(axs, **kws):
    '''Add part labels to the `axs`.'''
    label_start = 'a'
    style = {
        'fontstyle': 'italic',
        'loc': 'left',
        'verticalalignment': 'top',
        'horizontalalignment': 'center',
    }
    for (label_ord, ax) in enumerate(axs.flat, start=ord(label_start)):
        label = chr(label_ord)
        ax.set_title(f'({label})', **style, **kws)
