'''Load the antibody data.'''

import functools
import pathlib

import pandas

_DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'


@functools.lru_cache(maxsize=1)
def load_antibodies_lpbe_vnt():
    '''Load the predictive values of the antibody titer.'''
    return (
        pandas.read_csv(_DATA_PATH / 'antibodies_lpbe_vnt.csv')
        .set_index(['SAT', 'LPBE', 'VNT'])
        .loc[:, 'count']
        .unstack('LPBE')
    )
