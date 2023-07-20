'''Common code for the running and plotting with no chronic infections.'''

import pathlib

import baseline


store_path = pathlib.Path(__file__).with_suffix('.h5')


_parameters_nochronic = dict(probability_chronic=0)


def run(SAT, nruns, hdfstore, _parameters=None, *args, **kwargs):
    if _parameters is None:
        _parameters = {}
    _parameters =  _parameters | _parameters_nochronic
    return baseline.run(SAT, nruns, hdfstore, _parameters=_parameters,
                        *args, **kwargs)
