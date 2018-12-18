#!/usr/bin/python3
import sys

from matplotlib import pyplot

sys.path.append('..')
import herd
from herd._initial_conditions import find_hazard_infection, plot
sys.path.pop(-1)


for chronic in (False, True):
    params = herd.Parameters(chronic=chronic)
    hazard_infection = find_hazard_infection(params)
    plot(hazard_infection, params, show=False)
pyplot.legend(['Acute only', 'Chronic'])
pyplot.show()
