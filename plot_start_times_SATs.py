#!/usr/bin/python3

from matplotlib import pyplot
import pandas
import seaborn


def get_persistence_time():
    data = pandas.read_pickle('run_start_times_SATs.pkl')
    persistence_time = {}
    for i, d in data.groupby(level=data.index.names[:-1]):
        t = d.index.get_level_values(-1)
        persistence_time[i] = t.max() - t.min()
    persistence_time = pandas.Series(persistence_time)
    persistence_time.index.names = data.index.names[:-1]
    persistence_time = persistence_time.reset_index(name='persistence time')
    persistence_time['start_time'] \
        = (12 * persistence_time['start_time']).astype(int)
    return persistence_time


if __name__ == '__main__':
    persistence_time = get_persistence_time()
    seaborn.factorplot(data=persistence_time,
                       x='start_time', y='persistence time', col='SAT',
                       kind='box')
    pyplot.show()
