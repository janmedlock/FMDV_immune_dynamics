#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy
import functools
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

    pyplot.boxplot(D['extinction_times'].T, positions = X[0])
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel('Extinction Time')
    pyplot.title(', '.join(['{} = {}'.format(k, v)
                                       for (k, v) in kwds.items()]))


def plot_slice2D(X, D, parameters1, **kwds):
    dim = [len(parametersValues[p]) for p in parameters1]
    X = [x.reshape(dim) for x in X]

    pyplot.figure()

    pyplot.subplot(2, 2, 1)
    Y = numpy.median(D['extinction_times'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Median')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 2)
    Y = numpy.mean(D['extinction_times'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Mean')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 3)
    Y = find_quantile(D['extinction_times'], 0.95).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Upper 95% quantile')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])

    pyplot.subplot(2, 2, 4)
    Y = numpy.max(D['extinction_times'], axis = 1).reshape(dim)
    pyplot.pcolor(X[0], X[1], Y) 
    pyplot.colorbar()
    pyplot.title('Maximum')
    pyplot.xlabel(parameters1[0])
    pyplot.ylabel(parameters1[1])


def plot_slice(**kwds):
    parameters1 = [k for k in parametersOrdered if k not in kwds]

    filters = [extinction_times[k] == v for (k, v) in kwds.items()]
    f = functools.reduce(lambda x, y: x & y, filters)
    D = numpy.compress(f, extinction_times)

    X = [D[p] for p in parameters1]

    if len(kwds) == 2:
        plot_slice1D(X, D, parameters1, **kwds)
    elif len(kwds) == 1:
        plot_slice2D(X, D, parameters1, **kwds)
    else:
        raise ValueError(
            'I don\'t know how to handle kwds = "{}"!'.format(kwds))


# def set_ticks(axis, base = 10, minors = 10):
#     (loga, logb) = axis.get_view_interval()
#     (a, b) = (base ** loga, base ** logb)

#     major_locs = numpy.arange(numpy.ceil(loga),
#                               numpy.floor(logb) + 1)

#     minor_exps = numpy.arange(numpy.floor(loga), numpy.ceil(logb) + 1)
#     minor_subs = numpy.arange(1, base, base / float(minors))
#     minor_locs = numpy.log10(numpy.outer(base ** minor_exps, minor_subs).flatten())
#     minor_locs = numpy.compress((minor_locs >= loga)
#                                & (minor_locs <= logb),
#                                minor_locs)
    
#     labels = (10 ** major_locs).astype(int)

#     # axis.set_ticks(major_locs)
#     # axis.set_ticks(minor_locs, minor = True)
#     # axis.set_ticklabels(labels)
#     # Minor ticks aren't working for some reason
#     axis.set_ticks(minor_locs)
#     labels = [(10 ** l).astype(int) if l in major_locs else ''
#               for l in minor_locs]
#     labels[-1] = '{:g}'.format(10 ** minor_locs[-1])
#     axis.set_ticklabels(labels)

#     axis.set_pane_color((1, 1, 1, 0))


def set_axis(axes, coord, key):
    # axes.set_xlim, etc.
    getattr(axes, 'set_{}lim'.format(coord))(numpy.log10(limits[key]))

    # axes.set_xlabel, etc.
    getattr(axes, 'set_{}label'.format(coord))(parametersLabels[key])

    ticklocs_major = ticks[key]
    ticklocs_minor = parametersValues[key]
    ticklocs = numpy.unique(numpy.sort(numpy.hstack((ticklocs_major,
                                                     ticklocs_minor))))
    ticklabels = ['{:g}'.format(l) if l in ticklocs_major else ''
                  for l in ticklocs]

    # axes.set_xticks, etc.
    getattr(axes, 'set_{}ticks'.format(coord))(numpy.log10(ticklocs))
    # axes.set_xticklabels, etc.
    getattr(axes, 'set_{}ticklabels'.format(coord))(ticklabels)

    # axes.xaxis.set_pane_color, etc.
    getattr(axes, '{}axis'.format(coord)).set_pane_color((1, 1, 1, 0))


def plot_3D(X, Y, Z, C, title):
    fig = pyplot.figure()
    axes = fig.add_subplot(1, 1, 1, projection = '3d', axisbg = (0, 0, 0, 0))
    
    points = axes.scatter(numpy.log10(X), numpy.log10(Y), numpy.log10(Z),
                          c = C,
                          cmap = pyplot.cm.spectral_r,
                          linewidth = 0)

    W = limits['populationSize'][0] * numpy.ones_like(X)
    axes.scatter(numpy.log10(W), numpy.log10(Y), numpy.log10(Z),
                 c = (0.5, 0.5, 0.5),
                 linewidth = 0)
    # W = limits['infectionDuration'][1] * numpy.ones_like(Y)
    # axes.scatter(numpy.log10(X), numpy.log10(W), numpy.log10(Z),
    #              c = (0.5, 0.5, 0.5),
    #              linewidth = 0)
    W = limits['maternal_immunity_duration'][0] * numpy.ones_like(Z)
    axes.scatter(numpy.log10(X), numpy.log10(Y), numpy.log10(W),
                 c = (0.5, 0.5, 0.5),
                 linewidth = 0)

    set_axis(axes, 'x', 'population_size')
    set_axis(axes, 'y', 'maternal_immunity_duration')
    set_axis(axes, 'z', 'recovery_infection_duration')
    
    if len(C) > 0:
        cbar = fig.colorbar(points)
        cbar.set_label(title)
    else:
        cbar = None

    return (fig, cbar)


def plot_average_extinction_time(average = 'mean'):
    title = '{} time to extinction (days)'.format(average.capitalize())

    X = extinction_times['population_size']
    Y = extinction_times['maternal_immunity_duration']
    Z = extinction_times['recovery_infection_duration']
    C = getattr(numpy, average)(extinction_times['extinction_times'], axis = 1)

    # Only keep those with average greater than 30 days
    f = (C > 30.)
    X = X.compress(f)
    Y = Y.compress(f)
    Z = Z.compress(f)
    C = C.compress(f)

    (fig, cbar) = plot_3D(X, Y, Z, C, title)

    return fig


def plot_persistance(years):
    title = 'Proportion persisting ≥{} year{}'.format(
        years,
        '' if years == 1 else 's')
    
    X = extinction_times['population_size']
    Y = extinction_times['maternal_immunity_duration']
    Z = extinction_times['infection_duration']
    C = find_proportion_over_x(extinction_times['extinction_times'],
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
extinction_times = numpy.genfromtxt('search_parameters.csv',
                                    delimiter = ',',
                                    skip_header = 1,
                                    dtype = [
                                        ('population_size', int),
                                        ('maternal_immunity_duration', float),
                                        ('recovery_infection_duration', float),
                                        ('extinction_times', float,
                                         (100, ))],
                                    invalid_raise = False)

# Convert to days
extinction_times['infection_duration'] *= 365
extinction_times['extinction_times'] *= 365

# Truncate to parameter ranges.
ticks = {'population_size':            (100, 500, 1000, 5000, 10000),
         'maternal_immunity_duration': (0.25, 0.5, 0.75),
         'infection_duration':         (1.6, 5, 10, 15, 21)}

limits = {k: (v[0], v[-1]) for (k, v) in ticks.items()}

# f = functools.reduce(lambda x, y: x & y,
#                      [((extinction_times[k] >= v[0])
#                        & (extinction_times[k] <= v[1]))
#                       for (k, v) in limits.items()])
# extinction_times = numpy.compress(f, extinction_times)


parameters_ordered = ('population_size',
                      'maternal_immunity_duration',
                      'infection_duration')

parameters_labels = {
    'population_size':            'Population size',
    'maternal_immunity_duration': 'Maternal immunity duration (days)',
    'infection_duration':         'Infection duration (days)'}

parameters_values = {k: numpy.unique(extinction_times[k])
                     for k in parameters_ordered}

default = {'population_size': 100,
           'transmission_rate': 7.1,
           'infection_duration': 1.6 / 365 * 365}
           

# plot_slice(infection_duration = default['infection_duration'],
#            R0 = default['R0'])
# plot_slice(population_size = default['population_size'],
#            R0 = default['R0'])
# plot_slice(population_size = default['population_size'],
#            infection_duration = default['infection_duration'])


# plot_slice(R0 = default['R0'])
# plot_slice(infection_duration = default['infection_duration'])
# plot_slice(population_size = default['population_size'])


fig = plot_average_extinction_time('mean')
fig.savefig('extinction_time_mean.pdf')
fig = plot_average_extinction_time('median')
fig.savefig('extinction_time_median.pdf')

for years in (1, 2, 3, 5):
    fig = plot_persistance(years)
    fig.savefig('persistance_{}year{}.pdf'.format(years,
                                                  '' if years == 1 else 's'))


# pyplot.show()
