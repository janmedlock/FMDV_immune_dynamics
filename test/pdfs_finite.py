#!/usr/bin/python3

import copy
import sys

import pandas

sys.path.append('..')
import herd
import herd.chronic_recovery
import herd.initial_conditions.immune_status
import herd.progression
import herd.recovery
import herd.samples
sys.path.pop()


RVS = {
    'progression',
    'recovery',
    'chronic_recovery'
}


def test_sample(parameters, sample, sat, idx):
    params = copy.copy(parameters)
    for (key, val) in sample.items():
        setattr(params, key, val)
    shapes = pandas.Series(dtype=float)
    for rv in RVS:
        name = f'{rv}_shape'
        shapes[name] = getattr(params, name)
    is_small = (shapes < 1)
    solver = herd.initial_conditions.immune_status.Solver(params)
    try:
        solver.get_A()
    except AssertionError:
        shapes_small = ', '.join(f'{key}={val}'
                                 for (key, val) in shapes[is_small].items())
        print(f'{sat=}, {idx=}, {shapes_small}')
        assert is_small.any()
    else:
        assert ~(is_small.any())


def test_sat(sat):
    parameters = herd.Parameters(SAT=sat)
    samples = herd.samples.load(SAT=sat)
    for (n, sample) in samples.iterrows():
        test_sample(parameters, sample, sat, n)


if __name__ == '__main__':
    for sat in (1, 2, 3):
        test_sat(sat)
