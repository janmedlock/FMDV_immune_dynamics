'''Caching with `joblib.Memory()`.'''

import pathlib

import joblib

_path = pathlib.Path(__file__).with_name('_cache')

memory = joblib.Memory(_path, verbose=1)
