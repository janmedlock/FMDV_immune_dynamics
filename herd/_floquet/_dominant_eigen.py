import numpy
import scipy


def find(M, which='LM', return_eigenvector=True, maxiter=100000,
         **kwargs):
    '''Find the dominant eigenvalue & eigenvector of `M` using
    `scipy.sparse.linalg.eigs()`, which works for both sparse
    and dense matrices.
    `which='LR'` gets the eigenvalue with largest real part.
    `which='LM'` gets the eigenvalue with largest magnitude.'''
    # The solver just spins with inf/NaN entries.
    # This check handles both dense & sparse matrices.
    assert numpy.isfinite(M[numpy.nonzero(M)]).all(), \
        'M has inf or NaN entries.'
    result = scipy.sparse.linalg.eigs(M, k=1, which=which, maxiter=maxiter,
                                      return_eigenvectors=return_eigenvector,
                                      **kwargs)
    if return_eigenvector:
        (evals, evecs) = result
    else:
        evals = result
    eval_0 = numpy.real_if_close(evals[0])
    assert numpy.isreal(eval_0), f'Complex dominant eigenvalue: {eval_0}'
    if not return_eigenvector:
        return eval_0
    evecs /= evecs.sum(axis=0)
    evec_0 = numpy.real_if_close(evecs[:, 0])
    assert all(numpy.isreal(evec_0)), \
        f'Complex dominant eigenvector: {evec_0}'
    assert all((numpy.real(evec_0) >= 0) | numpy.isclose(evec_0, 0)), \
        f'Negative component in the dominant eigenvector: {evec_0}'
    # In case a component is negative but close to 0
    evec_0 = evec_0.clip(min=0)
    return (eval_0, evec_0)
