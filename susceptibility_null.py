'''Common code for simulations with 0 susceptibility of the
lost-immunity class.'''

import pathlib

import baseline


store_path = pathlib.Path(__file__).with_suffix('.h5')


_parameters_susceptibility_null = {
    'lost_immunity_susceptibility': 0,
}


def run(SAT, nruns, store, _parameters=None, *args, **kwargs):
    if _parameters is None:
        _parameters = {}
    _parameters =  _parameters | _parameters_susceptibility_null
    return baseline.run(SAT, nruns, store, _parameters=_parameters,
                        *args, **kwargs)
