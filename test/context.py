'''Context for the current directory.'''

import pathlib
import sys

_path = pathlib.Path(__file__).parents[1]
sys.path.append(str(_path))
import h5
import herd
import run
import samples_run
sys.path.pop()
