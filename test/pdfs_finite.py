#!/usr/bin/python3

from context import common, herd
from herd.initial_conditions.immune_status import solver
import herd.samples


def test_sample(parameters, sample, sat, idx, debug=False):
    print(f'{sat=}, {idx=}')
    params = parameters.merge(**sample)
    slvr = solver.Solver(params, debug=debug)
    # This was failing because some pdf(0) = infinity, i.e. gamma with
    # shape < 1, but it should work for all now.
    slvr.get_A()


def test_sat(sat, debug=False):
    parameters = herd.Parameters(SAT=sat)
    samples = herd.samples.load(SAT=sat)
    for (idx, sample) in samples.iterrows():
        test_sample(parameters, sample, sat, idx, debug=debug)


if __name__ == '__main__':
    for sat in common.SATs:
        test_sat(sat)
