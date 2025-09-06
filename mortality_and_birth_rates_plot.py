#!/usr/bin/python3
'''Plot the mortality and birth rates.'''

import abc
import pathlib

import matplotlib.pyplot
import matplotlib.ticker
import numpy
import seaborn

import common
import herd
import supplemental_materials


rc = common.rc | supplemental_materials.rc | {
    'figure.figsize': (supplemental_materials.WIDTH_MAXIMUM, 4),
    'axes.spines.right': False,
    'axes.spines.top': False,
}


class Base(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def x_label(self): pass

    @property
    @abc.abstractmethod
    def y_label(self): pass

    @property
    @abc.abstractmethod
    def x_lim(self): pass

    @property
    @abc.abstractmethod
    def x_major_multiple(self): pass

    @property
    @abc.abstractmethod
    def x_minor_ndivs(self): pass

    @property
    @abc.abstractmethod
    def y_major_multiple(self): pass

    @classmethod
    @abc.abstractmethod
    def plot_rate(cls, axes, x): pass

    rvs = herd.RandomVariables()
    n_points = 1001
    y_minor_ndivs = 2

    @classmethod
    def plot(cls, axes, **kwds):
        x = numpy.linspace(*cls.x_lim, cls.n_points)
        cls.plot_rate(axes, x, **kwds)
        axes.set_xlabel(cls.x_label)
        axes.set_ylabel(cls.y_label)
        axes.set_xlim(*cls.x_lim)
        axes.set_ylim(bottom=0)
        for which in ('x', 'y'):
            axis = getattr(axes, f'{which}axis')
            axis.set_major_locator(
                matplotlib.ticker.MultipleLocator(
                    getattr(cls, f'{which}_major_multiple')
                )
            )
            axis.set_minor_locator(
                matplotlib.ticker.AutoMinorLocator(
                    getattr(cls, f'{which}_minor_ndivs')
                )
            )


class Mortality(Base):
    x_label = 'Age (year)'
    y_label = r'Mortality rate (year$^{-1}$)'
    x_lim = [0, 20]
    x_major_multiple = 4
    x_minor_ndivs = 4
    y_major_multiple = 0.2

    @classmethod
    def plot_rate(cls, axes, age, **kwds):
        rate = cls.rvs.mortality.hazard(age)
        axes.step(age, rate, where='post', **kwds)


class Birth(Base):
    x_label = 'Time (year)'
    y_label = r'Birth rate (year$^{-1}$)'
    x_lim = [0, 2]
    x_major_multiple = 1
    x_minor_ndivs = 2
    y_major_multiple = 1

    @classmethod
    def plot_rate(cls, axes, time, **kwds):
        rate = cls.rvs.birth.hazard(time, 4)
        axes.plot(time, rate, **kwds)


def plot_rates(save=True):
    '''Plot the rates.'''
    rates = Base.__subclasses__()
    with seaborn.axes_style('ticks'), matplotlib.pyplot.rc_context(rc=rc):
        (fig, axes) = matplotlib.pyplot.subplots(len(rates))
        colors = (f'C{i}' for i in range(len(rates)))
        for (rate, ax, color) in zip(rates, axes, colors):
            rate.plot(ax, color=color, zorder=3)
        fig.align_labels()
        if save:
            source_path = pathlib.Path(__file__)
            output_path_stem = source_path.with_name(
                source_path.name.replace('_plot.py', '')
            )
            fig.savefig(output_path_stem.with_suffix('.pgf'))
        return fig


if __name__ == '__main__':
    plot_rates()
    matplotlib.pyplot.show()
