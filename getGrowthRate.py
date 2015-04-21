#!/usr/bin/python
#

import numpy
import time
import parameters


ageSteps = [0.5, 0.2, 0.1,
            0.05, 0.02, 0.01,
            0.005, 0.002, 0.001]

ageMax = 30.

N = []
scalings = []
times = []

for i in range(len(ageSteps)):
    print 'Running ageStep = {}'.format(ageSteps[i])
    
    t0 = time.time()
    
    parameters.birth.findBirthScaling(parameters.mortality,
                                      parameters.male,
                                      ageMax = ageMax,
                                      ageStep = ageSteps[i])

    N.append(ageMax / ageSteps[i] + 1.)

    times.append(time.time() - t0)

    print '\t{} seconds'.format(times[-1])

    scalings.append(parameters.birth.scaling)

    print '\tscaling = {}'.format(scalings[-1])
