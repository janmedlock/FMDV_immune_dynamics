#!/usr/bin/python

import numpy
from matplotlib import pyplot, ticker

import parameters


t = numpy.linspace(0, 20, 1001)

def axclean(ax):
    for l in ('top', 'right'):
        ax.spines[l].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')            
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


(fig, ax) = pyplot.subplots(1, 2, sharey = True)

ax[0].step(t, parameters.mortality.hazard(t), where = 'post', color = 'black')
ax[0].set_title('Death')
ax[0].set_xlabel('Age (years)')

ax[1].plot(t, parameters.birth.hazard(t, 0, 4 - t), color = 'black')
ax[1].set_title('Birth')
ax[1].set_xlabel('Time (years)')
ax[1].set_xlim(xmax = 3)

ax[0].set_ylim(0, 1)

for axes in ax:
    axclean(axes)

fig.savefig('hazards.pdf')


(fig, ax) = pyplot.subplots(1, 4, sharey = True)

ax[0].plot(t, parameters.mortality.sf(t), color = 'black')
ax[0].set_title('Death')
ax[0].set_xlabel('Age (years)')
ax[0].xaxis.set_major_locator(ticker.MaxNLocator(2))

ax[1].plot(t, parameters.birth.sf(t, 0, 4), color = 'black')
ax[1].set_title('Birth')
ax[1].set_xlabel('Time (years)')
ax[1].set_xlim(xmax = 10)
ax[1].xaxis.set_major_locator(ticker.MaxNLocator(2))

t1 = numpy.linspace(0, 1, 1001)
ax[2].step(t1, parameters.maternalImmunityWaning.sf(t1), where = 'post',
           color = 'black')
ax[2].set_title('Waning')
ax[2].set_xlabel('Age (years)')
ax[2].xaxis.set_major_locator(ticker.MaxNLocator(2))

t2 = numpy.linspace(0, 5, 1001)
ax[3].step(t2, parameters.recovery.sf(t2 / 365.), where = 'post',
           color = 'black')
ax[3].set_title('Recovery')
ax[3].set_xlabel('Time (days)')
ax[3].xaxis.set_major_locator(ticker.MaxNLocator(5))

ax[0].set_ylim(0, 1)

for axes in ax:
    axclean(axes)

fig.savefig('survivals.pdf')
