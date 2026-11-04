'''Load the antibody data.'''

import functools
import pathlib

import pandas

_PATH = pathlib.Path(__file__).parents[2] / 'data' / 'antibodies_lpbe_vnt.csv'


@functools.lru_cache(maxsize=1)
def load():
    '''Load the antibody data.'''
    return (
        pandas.read_csv(_PATH, index_col=['SAT', 'LPBE', 'VNT'])
        .loc[:, 'count']
        .unstack('LPBE')
    )
