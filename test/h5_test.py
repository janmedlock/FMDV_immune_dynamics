#!/usr/bin/python3

import sys

sys.path.append('..')
import h5
sys.path.pop()


filename = '../run.h5'
by = ['SAT', 'run']
columns = ['exposed', 'infectious', 'chronic']
SAT = 1
where = f'{SAT=}'

with h5.HDFStore(filename, mode='r') as store:
    for (_, group) in store.groupby(by, columns=columns,
                                    debug=True, where=where):
        infected = group.sum(axis='columns')
        observed = (infected.iloc[-1] == 0)
        t = group.index.get_level_values('time (y)')
        time = t.max() - t.min()
        assert observed or (time == 10)
