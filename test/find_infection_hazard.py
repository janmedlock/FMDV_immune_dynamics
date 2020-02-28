#!/usr/bin/python3
import sys

from matplotlib import pyplot

sys.path.append('..')
import herd
from herd.initial_conditions import infection
sys.path.pop()


models = ('acute', 'chronic')

for model in models:
    params = herd.Parameters(model=model)
    hazard_infection = infection.find_hazard(params)
    infection.plot(hazard_infection, params, show=False,
                   label=model.capitalize())
pyplot.legend()
pyplot.show()
