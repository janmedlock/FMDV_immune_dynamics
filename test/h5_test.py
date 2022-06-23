#!/usr/bin/python3

from context import baseline
from context import common
from context import h5


by = ['SAT', 'run']
SAT = 1
where = f'{SAT=}'

with h5.HDFStore(baseline.store_path, mode='r') as store:
    for (_, group) in store.groupby(by, columns=common.cols_infected,
                                    debug=True, where=where):
        infected = group.sum(axis='columns')
        observed = (infected.iloc[-1] == 0)
        t = group.index.get_level_values(common.t_name)
        time = t.max() - t.min()
        assert observed or (time == 10)
