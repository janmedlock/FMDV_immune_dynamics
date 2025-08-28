'''Common code to run and plot one simulation for each SAT.'''

import pathlib


SEED = 0xa70a_8239

store_path = pathlib.Path(__file__).with_suffix('.h5')
