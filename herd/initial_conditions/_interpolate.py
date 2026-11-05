import numpy

from .. import immune_status


class Interpolant:
    '''Interpolate the solution from `solver.solve()` to different
    ages.'''

    def __init__(self, immune_status_pdf):
        self._immune_status_pdf = immune_status_pdf

    def __call__(self, age):
        # Interpolate `self._solution` to `age`.
        # For some reason, duplicates may be present after `.union()`.
        index = (self._immune_status_pdf.index
                 .union(numpy.atleast_1d(age))
                 .drop_duplicates())
        result = (self._immune_status_pdf.reindex(index)
                  .interpolate()
                  .loc[age])
        shape = numpy.shape(age) + (len(immune_status.immune_statuses),)
        assert result.shape == shape
        return result
