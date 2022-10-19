#!/usr/bin/python3
'''Analyze and plot the results of the simulations with 0
susceptibility of the lost-immunity class. This requires the file
`susceptibility.h5`, which is built by `susceptibility_run.py`.'''

from matplotlib import pyplot

import baseline_plot
import common
import h5
import susceptibility
import susceptibility_null


def _build():
    susceptibility_level = 'lost_immunity_susceptibility'
    where = f'{susceptibility_level}=0'
    with (h5.HDFStore(susceptibility.store_path, mode='r') as store_in,
          h5.HDFStore(susceptibility_null.store_path, mode='w') as store_out):
        for chunk in store_in.select(where=where, iterator=True):
            chunk.reset_index(susceptibility_level, drop=True, inplace=True)
            store_out.put(chunk, index=False)
        store_out.create_table_index()
        store_out.repack()


def load():
    if not susceptibility_null.store_path.exists():
        _build()
    path = susceptibility_null.store_path
    infected = common.load_infected(path)
    extinction_time = common.load_extinction_time(path)
    return (infected, extinction_time)


def plot(infected, extinction_time, draft=False, save=True):
    fig = baseline_plot.plot(infected, extinction_time,
                             draft=draft, save=False)
    if save:
        fig.savefig(susceptibility_null.store_path.with_suffix('.pdf'))
        fig.savefig(susceptibility_null.store_path.with_suffix('.png'),
                    dpi=300)
    return fig


if __name__ == '__main__':
    DRAFT = False
    (infected, extinction_time) = load()
    plot(infected, extinction_time, draft=DRAFT)
    pyplot.show()
