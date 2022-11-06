#!/usr/bin/python3

import matplotlib.pyplot
import numpy

import common
import population_size_and_susceptibility as psas
import population_size_plot
import susceptibility_plot
import sensitivity


def load_extinction_time():
    return common.load_extinction_time(psas.store_path)



def get_persistence(dfr):
    grouper = dfr.groupby(['SAT',
                           'lost_immunity_susceptibility',
                           'population_size'])
    return grouper.apply(common.get_persistence)



def plot_persistence(dfr):
    rc = common.rc.copy()
    width = 183 / 25.4  # convert mm to in
    height = 4  # in
    rc['figure.figsize'] = (width, height)
    # Between 5pt and 7pt.
    rc['font.size'] = 6
    rc['axes.titlesize'] = 9
    rc['axes.labelsize'] = 8
    rc['xtick.labelsize'] = rc['ytick.labelsize'] = 7
    with matplotlib.pyplot.rc_context(rc=rc):
        persistence = get_persistence(dfr)
        grouper = persistence.groupby('SAT')
        ncols = len(grouper)
        (fig, axes) = matplotlib.pyplot.subplots(1, ncols,
                                                 sharey='row',
                                                 layout='constrained')
        axes = numpy.atleast_1d(axes)
        for ((SAT, group), ax) in zip(grouper, axes):
            # Move population_size to columns.
            arr = group.unstack()
            # Fill missing values with values to their left.
            # This will fill the missing values with 100%'s.
            arr.fillna(method='ffill', axis='columns', inplace=True)
            x = arr.columns
            y = arr.index.droplevel('SAT')
            cmap = sensitivity.get_cmap(SAT)
            ax.pcolormesh(x, y, arr,
                          cmap=cmap, vmin=0, vmax=1,
                          shading='gouraud')
            ax.set_title(f'SAT{SAT}')
            ax.set_xscale('log')
            ax.set_xlim(min(psas.population_sizes),
                        max(psas.population_sizes))
            ax.set_xlabel(population_size_plot.sens.label)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylim(min(psas.susceptibilities),
                            max(psas.susceptibilities))
                ax.set_ylabel(susceptibility_plot.sens.label)
        fig.align_xlabels(axes)


if __name__ == '__main__':
    dfr = load_extinction_time()
    plot_persistence(dfr)
    matplotlib.pyplot.show()
