'''Context for the current directory.'''

import pathlib
import sys

_path = pathlib.Path(__file__).parents[1]
sys.path.append(str(_path))
import baseline
import common
import herd
import plotting
sys.path.pop()
