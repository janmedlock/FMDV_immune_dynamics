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
    'chronic_recovery',
    'progression',
    'recovery',
}


def test_sample(parameters, sample, sat, idx):
    params = copy.copy(parameters)
    for (key, val) in sample.items():
        setattr(params, key, val)
    solver = herd.initial_conditions.immune_status.Solver(params)
    # This was failing because some pdf(0) = infinity, i.e. gamma with
    # shape < 1, but it should work for all now.
    solver.get_A()


def test_sat(sat):
    parameters = herd.Parameters(SAT=sat)
    samples = herd.samples.load(SAT=sat)
    for (idx, sample) in samples.iterrows():
        test_sample(parameters, sample, sat, idx)


if __name__ == '__main__':
    for sat in (1, 2, 3):
        test_sat(sat)
