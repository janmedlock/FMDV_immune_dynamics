#!/usr/bin/python3

import h5


filename = 'run_population_sizes.h5'
by = ['model', 'SAT', 'population_size', 'run']
columns = ['exposed', 'infectious', 'chronic']
where = dict(model='acute', SAT=1)
where = ' & '.join(f'{k}={v}' for (k, v) in where.items())

with h5.HDFStore(filename, mode='r') as store:
    for (_, group) in store.groupby(by, columns=columns,
                                    debug=True, where=where):
        infected = group.sum(axis='columns')
        observed = (infected.iloc[-1] == 0)
        t = group.index.get_level_values('time (y)')
        time = t.max() - t.min()
        assert observed or (time == 10)
