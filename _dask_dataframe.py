'''Import dask.dataframe while silencing a warning.'''

import warnings

with warnings.catch_warnings(action='ignore', category=FutureWarning):
    from dask.dataframe import *
