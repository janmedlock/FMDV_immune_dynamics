#!/usr/bin/python3
'''Sort the data from the susceptibility simulations in case they
somehow got out of order.'''


import h5
import susceptibility_run


if __name__ == '__main__':
    h5.sort_index(susceptibility_run.store_path,
                  ('SAT', 'lost_immunity_susceptibility', 'run'))
