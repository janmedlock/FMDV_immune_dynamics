#!/usr/bin/python3

import matplotlib.pyplot
import numpy

import common
import population_size
import population_size_and_susceptibility
import sensitivity
import susceptibility


def load_extinction_time():
    return common.load_extinction_time(
        population_size_and_susceptibility.store_path)



def get_persistence(dfr):
    grouper = dfr.groupby(['SAT',
                           susceptibility.var,
                           population_size.var])
    return grouper.apply(common.get_persistence)



def fill_missing_persistence(dfr):
    assert population_size_and_susceptibility.is_monotone_increasing(
        dfr.columns)
    # Starting from the left, where there is a missing value, if the
    # value in the previous column is 1, set the current value to 1.
    # Skip the first column since it has no previous column.
    for (col_prev, col_curr) in zip(dfr.columns[:-1], dfr.columns[1:]):
        to_update = (dfr[col_curr].isnull()
                     & (dfr[col_prev] == 1))
        dfr.loc[to_update, col_curr] = 1


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
        for ((SAT, group), ax) in zip(grouper, axes):
            # Move population_size from an index level to columns.
            arr = group.unstack()
            fill_missing_persistence(arr)
            x = arr.columns
            y = arr.index.droplevel('SAT')
            cmap = sensitivity.get_cmap(SAT)
            ax.pcolormesh(x, y, arr,
                          cmap=cmap, vmin=0, vmax=1,
                          shading='gouraud')
            ax.set_title(f'SAT{SAT}')
            ax.set_xscale('log')
            ax.set_xlim(min(population_size.values),
                        max(population_size.values))
            ax.set_xlabel(population_size.label)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylim(min(susceptibility.values),
                            max(susceptibility.values))
                ax.set_ylabel(susceptibility.label)
        fig.align_xlabels(axes)


if __name__ == '__main__':
    dfr = load_extinction_time()
    plot_persistence(dfr)
    matplotlib.pyplot.show()
