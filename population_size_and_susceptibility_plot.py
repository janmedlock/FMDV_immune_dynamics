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
    isnull = dfr.isnull()
    # Fill missing values with values to their left in the same row.
    # This should fill the missing values with 1's.
    dfr.fillna(method='ffill', axis='columns', inplace=True)
    # Check that the filled values are all 1.
    assert all(numpy.allclose(dfr.loc[row, row_mask], 1)
               for (row, row_mask) in isnull.iterrows())


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
