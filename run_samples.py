#!/usr/bin/python3

import os

import run_common


if __name__ == '__main__':
    tmax = 10

    os.makedirs('run_samples', exist_ok=True)
    for model in ('acute', 'chronic'):
        run_common.run_samples(model, tmax,
                               logging_prefix=f'model {model}, ')
