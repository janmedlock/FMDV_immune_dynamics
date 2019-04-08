'''Monkeypatch `joblib.memory.MemorizedFunc()._persist_input()`
to silence the warning about slow persisting input arguments.'''

import functools
import warnings

from joblib import Memory
from joblib.memory import MemorizedFunc


_persist_input_orig = MemorizedFunc._persist_input

@functools.wraps(_persist_input_orig)
def _persist_input_quiet(self, *args, **kwds):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='joblib', category=UserWarning,
                                message='Persisting input arguments took ')
        _persist_input_orig(self, *args, **kwds)

MemorizedFunc._persist_input = _persist_input_quiet
