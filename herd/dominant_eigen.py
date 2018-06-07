import numpy
from scipy import sparse


def find(A, which='LR', maxiter=100000, *args, **kwargs):
    '''Find the dominant eigenvalue & eigenvector of `A` using
    `scipy.sparse.linalg.eigs()`, which works for both sparse
    and dense matrices.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    # I think this check handles dense & sparse matrices, etc.
    assert numpy.isfinite(A[numpy.nonzero(A)]).all(), 'A has inf/NaN entries.'
    L, V = sparse.linalg.eigs(A, k=1, which=which, maxiter=maxiter,
                              *args, **kwargs)
    l0 = numpy.real_if_close(L[0])
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    v0 = V[:, 0]
    v0 = numpy.real_if_close(v0 / v0.sum())
    assert all(numpy.isreal(v0)), 'Complex dominant eigenvector: {}'.format(v0)
    assert all((numpy.real(v0) >= 0) | numpy.isclose(v0, 0)), \
        'Negative component in the dominant eigenvector: {}'.format(v0)
    return (l0, v0)
