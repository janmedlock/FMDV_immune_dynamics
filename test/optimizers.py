#!/usr/bin/python3

from context import common, herd
from herd.initial_conditions.immune_status import _solver


if __name__ == '__main__':
    for SAT in common.SATs:
        print(f'SAT{SAT}')
        parameters = herd.Parameters(SAT=SAT)
        optimizer = _solver.get_optimizer(parameters)
        for (key, val) in optimizer.items():
            print(f'{key}={val}')
