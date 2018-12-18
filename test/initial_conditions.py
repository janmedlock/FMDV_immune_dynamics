#!/usr/bin/python3
import sys
sys.path.append('..')

from matplotlib import pyplot
import numpy

from herd import Parameters, RandomVariables


age_max = 10
age_bin_width = 0.1
ages = numpy.linspace(0, age_max, 301)

chronic = True
parameters = Parameters(chronic=chronic)
RVs = RandomVariables(parameters)
ICs = RVs.initial_conditions

fig, axes = pyplot.subplots(3, 1, sharex=True)

status_prob_cond = ICs._proportion(ages)
total = numpy.zeros_like(ages)
for (i, (k, v)) in enumerate(status_prob_cond.items()):
    total += v
    z = len(status_prob_cond) - i
    axes[0].fill_between(ages, total, label=k, zorder=z)
axes[0].set_ylabel('probability\ngiven age')

status_prob = ICs.pdf(ages)
total = numpy.zeros_like(ages)
for (i, (k, v)) in enumerate(status_prob.items()):
    total += v
    z = len(status_prob) - i
    axes[1].fill_between(ages, total, label=None, zorder=z)
axes[1].set_ylabel('joint\ndensity')

numpy.random.seed(1) # Make this cache friendly.
status_ages = ICs.rvs(parameters.population_size)
left = numpy.arange(0, age_max, age_bin_width)
bins = numpy.hstack((left, age_max))
width = age_bin_width
bottom = numpy.zeros_like(left)
for (status, ages) in status_ages.items():
    height, _ = numpy.histogram(ages, bins=bins)
    axes[2].bar(left, height, width, bottom, label=None, align='edge')
    bottom += height
axes[2].set_ylabel('number in\nsample')

axes[-1].set_xlabel('age')

fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(loc='lower center',
           ncol=int(numpy.ceil(status_prob.shape[1] / 2)))

pyplot.show()
