import numpy
import pandas


def probability(age, hazard_infection, parameters):
    '''The probability of being in each immune status at age `a`,
    given being alive at age `a`.'''
    index = pandas.Index(numpy.atleast_1d(age),
                         name='age')
    # TODO: Filler for now.
    columns = ('maternal immunity', 'susceptible', 'exposed',
               'infectious', 'chronic', 'recovered', 'lost immunity')
    status = pandas.DataFrame(1 / len(columns),
                              index=index,
                              columns=columns)
    assert (status >= 0).all()
    assert numpy.isclose(status.sum(axis='columns'), 1).all()
    return status
