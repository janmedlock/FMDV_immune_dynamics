#!/usr/bin/python3
import sys
sys.path.append('..')

from herd._initial_conditions import find_hazard_infection, find_AIC, plot


hazard_infection = find_hazard_infection()
if len(hazard_infection) > 1:
    AIC = find_AIC(hazard_infection)
    print('Separate AIC = {:g}'.format(AIC.sum() - AIC['Pooled']))
    print('Pooled   AIC = {:g}'.format(AIC['Pooled']))
plot(hazard_infection)
