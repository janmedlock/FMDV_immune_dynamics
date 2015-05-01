#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from matplotlib import pyplot
import mpl_toolkits.mplot3d


def find_quantile_single(d, q):
    s = sorted(d)
    i = int(numpy.floor(q * len(s)))
    return s[i]
  
def find_quantile(D, q):
    return numpy.array([find_quantile_single(d, q) for d in D])

def find_proportion_over_x(D, x):
    return (numpy.asarray(D) >= x).sum(axis = -1) / float(numpy.shape(D)[1])


def plot_slice1D(X, D, parameters1, **kwds):
    pyplot.figure()

    pyplot.boxplot(D['extinctionTimes'].T, positions = X[0])
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel('Extinction Time')
    pyplot.title(', '.join(['{} = {}'.format(k, v)
                                       for (k, v) in kwds.iteritems()]))


def plot_slice2D(X, D, parameters1, **kwds):
    dim = [len(parametersValues[p]) for p in parameters1]
    X = map(lambda x: x.reshape(dim), X)

    pyplot.figure()

    pyplot.subplot(2, 2, 1)
    Y = numpy.median(D['extinctionTimes'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Median')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 2)
    Y = numpy.mean(D['extinctionTimes'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Mean')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 3)
    Y = find_quantile(D['extinctionTimes'], 0.95).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Upper 95% quantile')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 4)
    Y = numpy.max(D['extinctionTimes'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Maximum')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])


def plot_slice(**kwds):
    parameters1 = [k for k in parametersOrdered if k not in kwds]

    filters = [extinctionTimes[k] == v for (k, v) in kwds.iteritems()]
    f = reduce(lambda x, y: x & y, filters)
    D = numpy.compress(f, extinctionTimes)

    X = [D[p] for p in parameters1]

    if len(kwds) == 2:
        plot_slice1D(X, D, parameters1, **kwds)
    elif len(kwds) == 1:
        plot_slice2D(X, D, parameters1, **kwds)
    else:
        raise ValueError(
            'I don\'t know how to handle kwds = "{}"!'.format(kwds))


# def setTicks(axis, base = 10, minors = 10):
#     (loga, logb) = axis.get_view_interval()
#     (a, b) = (base ** loga, base ** logb)

#     majorLocs = numpy.arange(numpy.ceil(loga),
#                              numpy.floor(logb) + 1)

#     minorExps = numpy.arange(numpy.floor(loga), numpy.ceil(logb) + 1)
#     minorSubs = numpy.arange(1, base, base / float(minors))
#     minorLocs = numpy.log10(numpy.outer(base ** minorExps, minorSubs).flatten())
#     minorLocs = numpy.compress((minorLocs >= loga)
#                                & (minorLocs <= logb),
#                                minorLocs)
    
#     labels = (10 ** majorLocs).astype(int)

#     # axis.set_ticks(majorLocs)
#     # axis.set_ticks(minorLocs, minor = True)
#     # axis.set_ticklabels(labels)
#     # Minor ticks aren't working for some reason
#     axis.set_ticks(minorLocs)
#     labels = [(10 ** l).astype(int) if l in majorLocs else ''
#               for l in minorLocs]
#     labels[-1] = '{:g}'.format(10 ** minorLocs[-1])
#     axis.set_ticklabels(labels)

#     axis.set_pane_color((1, 1, 1, 0))


def setAxis(axes, coord, key):
    # axes.set_xlim, etc.
    getattr(axes, 'set_{}lim'.format(coord))(numpy.log10(limits[key]))

    # axes.set_xlabel, etc.
    getattr(axes, 'set_{}label'.format(coord))(parametersLabels[key])

    tickLocsMajor = ticks[key]
    tickLocsMinor = parametersValues[key]
    tickLocs = numpy.unique(numpy.sort(numpy.hstack((tickLocsMajor,
                                                     tickLocsMinor))))
    tickLabels = ['{:g}'.format(l) if l in tickLocsMajor else ''
                  for l in tickLocs]

    # axes.set_xticks, etc.
    getattr(axes, 'set_{}ticks'.format(coord))(numpy.log10(tickLocs))
    # axes.set_xticklabels, etc.
    getattr(axes, 'set_{}ticklabels'.format(coord))(tickLabels)

    # axes.xaxis.set_pane_color, etc.
    getattr(axes, '{}axis'.format(coord)).set_pane_color((1, 1, 1, 0))


def plot_3D(X, Y, Z, C, title):
    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1, projection = '3d', axisbg = (0, 0, 0, 0))
    
    points = axes.scatter(numpy.log10(X), numpy.log10(Y), numpy.log10(Z),
                          c = C,
                          cmap = pyplot.cm.spectral_r,
                          linewidth = 0,
                          vmin = 0.)

    W = limits['populationSize'][0] * numpy.ones_like(X)
    axes.scatter(numpy.log10(W), numpy.log10(Y), numpy.log10(Z),
                 c = (0.5, 0.5, 0.5),
                 linewidth = 0)
    # W = limits['infectionDuration'][1] * numpy.ones_like(Y)
    # axes.scatter(numpy.log10(X), numpy.log10(W), numpy.log10(Z),
    #              c = (0.5, 0.5, 0.5),
    #              linewidth = 0)
    W = limits['R0'][0] * numpy.ones_like(Z)
    axes.scatter(numpy.log10(X), numpy.log10(Y), numpy.log10(W),
                 c = (0.5, 0.5, 0.5),
                 linewidth = 0)

    setAxis(axes, 'x', 'populationSize')
    setAxis(axes, 'y', 'infectionDuration')
    setAxis(axes, 'z', 'R0')
    
    if len(C) > 0:
        cbar = fig.colorbar(points)
        cbar.set_label(title)
    else:
        cbar = None

    return (fig, cbar)


def plot_average_extinction_time(average = 'mean'):
    title = '{} time to extinction (days)'.format(average.capitalize())

    X = extinctionTimes['populationSize']
    Y = extinctionTimes['infectionDuration']
    Z = extinctionTimes['R0']
    C = getattr(numpy, average)(extinctionTimes['extinctionTimes'], axis = 1)

    # Only keep those with average greater than 30 days
    f = (C > 30.)
    X = X.compress(f)
    Y = Y.compress(f)
    Z = Z.compress(f)
    C = C.compress(f)

    (fig, cbar) = plot_3D(X, Y, Z, C, title)

    return fig


def plot_persistance(years):
    title = u'Proportion persisting ≥{} year{}'.format(
        years,
        '' if years == 1 else 's')
    
    X = extinctionTimes['populationSize']
    Y = extinctionTimes['infectionDuration']
    Z = extinctionTimes['R0']
    C = find_proportion_over_x(extinctionTimes['extinctionTimes'],
                               365. * years)

    # Keep only those with persistence > 0
    f = (C > 0.)
    X = X.compress(f)
    Y = Y.compress(f)
    Z = Z.compress(f)
    C = C.compress(f)

    (fig, cbar) = plot_3D(X, Y, Z, C, title)

    if cbar is not None:
       cbar.set_ticks([float(l.get_text())
                       for l in cbar.ax.yaxis.get_ticklabels()])
       cbar.set_ticklabels(['{}%'.format(100. * float(l.get_text()))
                            for l in cbar.ax.yaxis.get_ticklabels()])

    return fig


# Read in data.
extinctionTimes = numpy.genfromtxt('searchParameters.csv',
                                   delimiter = ',',
                                   skip_header = 1,
                                   dtype = [('populationSize', int),
                                            ('infectionDuration', float),
                                            ('R0', float),
                                            ('extinctionTimes', float,
                                             (100, ))],
                                   invalid_raise = False)

# Convert to days
extinctionTimes['infectionDuration'] *= 365.
extinctionTimes['extinctionTimes'] *= 365.

# Truncate to parameter ranges.
ticks = {'populationSize':    (100, 500, 1000, 5000, 10000),
         'infectionDuration': (1.6, 5., 10., 15., 21.),
         'R0':                (1.2, 5., 10., 20., 30.)}

limits = {k: (v[0], v[-1]) for (k, v) in ticks.iteritems()}

# f = reduce(lambda x, y: x & y,
#            [(extinctionTimes[k] >= v[0]) & (extinctionTimes[k] <= v[1])
#             for (k, v) in limits.iteritems()])
# extinctionTimes = numpy.compress(f, extinctionTimes)


parametersOrdered = ('populationSize',
                     'infectionDuration',
                     'R0')

parametersLabels = {'populationSize':    'Population size',
                    'infectionDuration': 'Infection duration (days)',
                    'R0':                '$R_0$'}

parametersValues = {k: numpy.unique(extinctionTimes[k])
                    for k in parametersOrdered}

default = {'populationSize': 100,
           'infectionDuration': 1.6 / 365. * 365.,
           'R0': 5.}
           

# plot_slice(infectionDuration = default['infectionDuration'],
#            R0 = default['R0'])
# plot_slice(populationSize = default['populationSize'],
#            R0 = default['R0'])
# plot_slice(populationSize = default['populationSize'],
#            infectionDuration = default['infectionDuration'])


# plot_slice(R0 = default['R0'])
# plot_slice(infectionDuration = default['infectionDuration'])
# plot_slice(populationSize = default['populationSize'])


fig = plot_average_extinction_time('mean')
fig.savefig('extinction_time_mean.pdf')
fig = plot_average_extinction_time('median')
fig.savefig('extinction_time_median.pdf')

for years in (1, 2, 3, 5):
    fig = plot_persistance(years)
    fig.savefig('persistance_{}year{}.pdf'.format(years,
                                                  '' if years == 1 else 's'))


# pyplot.show()
