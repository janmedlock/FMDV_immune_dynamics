#!/usr/bin/python3

import copy
import os.path
import sys

import numpy

sys.path.append('..')
import herd
import herd.chronic_recovery
import herd.progression
import herd.recovery
import herd.samples
sys.path.pop()


RVS = {
    'progression',
    'recovery',
    'chronic_recovery'
}


def test_sample(parameters, vals, sat, sample):
    p = copy.copy(parameters)
    for (k, v) in vals.items():
        setattr(p, k, v)
    for rv in RVS:
        val = getattr(herd, rv).gen(p).pdf(0)
        if numpy.isinf(val):
            shape = getattr(p, f'{rv}_shape')
            print(f'{sat=}, {sample=}, {rv=}, {shape=}')
            assert shape < 1


def test_sat(sat):
    parameters = herd.Parameters(SAT=sat)
    samples = herd.samples.load(SAT=sat)
    for (n, sample) in samples.iterrows():
        test_sample(parameters, sample, sat, n)


if __name__ == '__main__':
    for sat in (1, 2, 3):
        test_sat(sat)
