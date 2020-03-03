#!/usr/bin/python3
import sys

sys.path.append('..')
import herd
from herd.initial_conditions import infection
sys.path.pop()


params = herd.Parameters()
hazard_infection = infection.find_hazard(params)
infection.plot(hazard_infection, params)
