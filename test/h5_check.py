#!/usr/bin/python3
'''Check simuation output for consistency.'''

from context import baseline
from context import common
from context import h5


def check_completed(path, by, **kwds):
    with h5.HDFStore(path, mode='r') as store:
        grouper = store.groupby(by, columns=common.cols_infected, **kwds)
        for (_, group) in grouper:
            infected = group.sum(axis='columns')
            extinction = infected.iloc[-1] == 0
            t = group.index.get_level_values(common.t_name)
            time = t.max() - t.min()
            completed = extinction or (time == common.TMAX)
            assert completed


if __name__ == '__main__':
    SAT = 1
    check_completed(baseline.store_path, ['SAT', 'run'], where=f'{SAT=}')
