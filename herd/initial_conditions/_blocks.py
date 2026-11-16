'''Blocks used by solver.Solver().'''

import abc
import itertools
import re

import numpy
import numpy.lib.recfunctions
import pandas
import scipy

from .. import immune_status

# The short names of the variables.
# The order of the output is determined by the order of these, too.
SHORT_NAMES = {
    'maternal_immunity'  : 'M',
    'susceptible'        : 'S',
    'exposed'            : 'E',
    'infectious'         : 'I',
    'carrier'            : 'C',
    'recovered_true'     : 'R',
    'recovered_false'    : 'Q',
    'lost_immunity_true' : 'L',
    'lost_immunity_false': 'K',
}
assert set(SHORT_NAMES) == set(immune_status.immune_statuses)
assert len(set(SHORT_NAMES.values())) == len(SHORT_NAMES), 'Duplicated values'

SHORT_NAMES_TO_LONG_NAMES = {s: l for (l, s) in SHORT_NAMES.items()}

LONG_NAMES_DTYPE = pandas.CategoricalDtype(SHORT_NAMES, # `.keys()`
                                           ordered=True)


class _Block(metaclass=abc.ABCMeta):
    '''Get a block row M_X of M and block b_X of b.'''

    def __init__(self, params, I, A):
        self.params = params
        self.I = I
        self.A = A
        self._set_M_XY_regex = re.compile(f'set_M_{self.X}([A-Z])$')
        self.set_M_X()
        self.b_X = self.build_b_X()

    @property
    @abc.abstractmethod
    def X(self):
        '''The variable name.'''
        # This gets set automatically by `_set_X()` at the bottom of
        # this file.

    def build_b_X(self, empty=True):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        if empty:
            return scipy.sparse.csr_array((len(self), 1), dtype=float)
        # Make the pattern, with birth in the 0th entry, then zeros
        # everywhere else. The value in the 0th entry will be updated
        # in `update()`.
        return scipy.sparse.csr_array(([1], ([0], [0])),
                                      shape=(len(self), 1), dtype=float)

    def _is_set_M_XY(self, attr):
        '''Check if `attr` is 'set_M_XY' for some Y.'''
        return (self._set_M_XY_regex.match(attr) is not None)

    def set_M_X(self):
        '''Assemble the block row M_X from its columns M_XY.'''
        self.M_X = {}
        for attr in dir(self):
            if self._is_set_M_XY(attr):
                getattr(self, attr)()

    def update(self):
        '''Do nothing unless overridden.'''


class _BlockODE(_Block, metaclass=abc.ABCMeta):
    '''A `_Block()` for a variable governed by an ODE.'''
    def __len__(self):
        return self.params.length_ode

    def _get_M_XX(self, hazard_out):
        '''Get the diagonal block `M_XX` that maps state X to itself.'''
        f_XX = -((hazard_out + self.params.hazard.mortality)
                 * self.params.step / 2)
        F_XX = scipy.sparse.diags_array(f_XX, format='csr')
        return (
            self.I - self.A - (self.I + self.A) @ F_XX
        )

    def _get_M_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `M_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        f_XY = hazard_in * self.params.step / 2
        F_XY = scipy.sparse.diags_array(f_XY, format='csr')
        return -(self.I + self.A) @ F_XY

    def _get_M_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `M_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        F_XY = scipy.sparse.coo_array((len(self), self.params.length_pde))
        for i in range(len(self)):
            j = numpy.arange(i + 1)
            F_XY[i, j] = (pdf_in[i - j]
                          * self.params.survival.mortality[i]
                          / self.params.survival.mortality[j]
                          * self.params.step ** 2)
        return -(self.I + self.A) @ F_XY


class _BlockPDE(_Block, metaclass=abc.ABCMeta):
    '''A `_Block()` for a variable governed by a PDE.'''
    def __len__(self):
        return self.params.length_pde

    def _get_M_XX(self):
        '''Get the diagonal block `M_XX` that maps state X to itself.'''
        return self.I

    def _get_M_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `M_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        f_XY = hazard_in / 2
        F_XY = scipy.sparse.diags_array(f_XY)
        return -F_XY

    def _get_M_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `M_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        F_XY = scipy.sparse.coo_array((len(self), self.params.length_pde))
        for i in range(len(self)):
            j = numpy.arange(i + 1)
            F_XY[i, j] = (pdf_in[i - j]
                          * self.params.survival.mortality[i]
                          / self.params.survival.mortality[j]
                          * self.params.step)
        return -F_XY

    def integrate(self, p_X, survival_out):
        P_X = numpy.empty(self.params.length_ode)
        for i in range(self.params.length_ode):
            j = numpy.arange(i + 1)
            P_X[i] = (numpy.dot(survival_out[j],
                                (self.params.survival.mortality[i]
                                 / self.params.survival.mortality[i - j]
                                 * p_X[i - j]))
                      * self.params.step)
        return P_X


class BlockM(_BlockODE):
    def build_b_X(self):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        return super().build_b_X(empty=False)

    def set_M_MM(self):
        self.M_X['M'] = self._get_M_XX(
            self.params.hazard.maternal_immunity_waning)

    def update(self):
        self.b_X[0] = self.params.birth_maternal_immunity_proportion


class BlockS(_BlockODE):
    def build_b_X(self):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        return super().build_b_X(empty=False)

    def set_M_SM(self):
        self.M_X['M'] = self._get_M_XY_ODE(
            self.params.hazard.maternal_immunity_waning)

    def set_M_SS(self):
        self.M_X['S'] = self._get_M_XX(self.params.hazard.infection)

    def update(self):
        self.b_X[0] = 1 - self.params.birth_maternal_immunity_proportion
        self.set_M_SS()


class BlockE(_BlockPDE):
    def set_M_ES(self):
        self.M_X['S'] = self._get_M_XY_ODE(self.params.hazard.infection)

    def set_M_EE(self):
        self.M_X['E'] = self._get_M_XX()

    def set_M_EQ(self):
        self.M_X['Q'] = self.M_X['L'] = self._get_M_XY_ODE(
            self.params.lost_immunity_susceptibility
            * self.params.hazard.infection)

    def set_M_EL(self):
        # `self.M_X['L']` is set in `.set_M_EQ()`
        pass

    def update(self):
        self.set_M_ES()
        self.set_M_EQ()
        self.set_M_EL()

    def integrate(self, p_E):
        return super().integrate(p_E, self.params.survival.progression)


class BlockI(_BlockPDE):
    def set_M_IE(self):
        self.M_X['E'] = self._get_M_XY_PDE(self.params.pdf.progression)

    def set_M_II(self):
        self.M_X['I'] = self._get_M_XX()

    def integrate(self, p_I):
        return super().integrate(p_I, self.params.survival.recovery)


class BlockC(_BlockPDE):
    def set_M_CI(self):
        self.M_X['I'] = self._get_M_XY_PDE(
            self.params.recovery_to_carrier_probability
            * self.params.pdf.recovery)

    def set_M_CC(self):
        self.M_X['C'] = self._get_M_XX()

    def integrate(self, p_C):
        return super().integrate(p_C, self.params.survival.carrier_recovery)


class BlockR(_BlockODE):
    def set_M_RI(self):
        self.M_X['I'] = self._get_M_XY_PDE(
            self.params.recovered_true_probability
            * (1 - self.params.recovery_to_carrier_probability)
            * self.params.pdf.recovery)

    def set_M_RC(self):
        self.M_X['C'] = self._get_M_XY_PDE(
            self.params.recovered_true_probability
            * self.params.pdf.carrier_recovery)

    def set_M_RR(self):
        self.M_X['R'] = self._get_M_XX(self.params.hazard.antibody_loss)

    def set_M_RL(self):
        self.M_X['L'] = self.M_X['K'] = self._get_M_XY_ODE(
            self.params.recovered_true_probability
            * self.params.hazard.antibody_gain)

    def set_M_RK(self):
        # `self.M_X['K']` is set in `.set_M_RL()`
        pass


class BlockQ(_BlockODE):
    def set_M_QI(self):
        self.M_X['I'] = self._get_M_XY_PDE(
            (1 - self.params.recovered_true_probability)
            * (1 - self.params.recovery_to_carrier_probability)
            * self.params.pdf.recovery)

    def set_M_QC(self):
        self.M_X['C'] = self._get_M_XY_PDE(
            (1 - self.params.recovered_true_probability)
            * self.params.pdf.carrier_recovery)

    def set_M_QQ(self):
        self.M_X['Q'] = self._get_M_XX(
            (self.params.lost_immunity_susceptibility
             * self.params.hazard.infection)
            + self.params.hazard.antibody_loss)

    def set_M_QL(self):
        self.M_X['L'] = self.M_X['K'] = self._get_M_XY_ODE(
            (1 - self.params.recovered_true_probability)
            * self.params.hazard.antibody_gain)

    def set_M_QK(self):
        # `self.M_X['K']` is set in `.set_M_QL()`
        pass

    def update(self):
        self.set_M_QQ()


class BlockL(_BlockODE):
    def set_M_LR(self):
        self.M_X['R'] = self.M_X['Q'] = self._get_M_XY_ODE(
            self.params.lost_immunity_true_probability
            * self.params.hazard.antibody_loss)

    def set_M_LQ(self):
        # `self.M_X['Q']` is set in `.set_M_LR()`
        pass

    def set_M_LL(self):
        self.M_X['L'] = self._get_M_XX(
            (self.params.lost_immunity_susceptibility
             * self.params.hazard.infection)
            + self.params.hazard.antibody_gain)

    def update(self):
        self.set_M_LL()


class BlockK(_BlockODE):
    def set_M_KR(self):
        self.M_X['R'] = self.M_X['Q'] = self._get_M_XY_ODE(
            (1 - self.params.lost_immunity_true_probability)
            * self.params.hazard.antibody_loss)

    def set_M_KQ(self):
        # `self.M_X['Q']` is set in `.set_M_KR()`
        pass

    def set_M_KK(self):
        self.M_X['K'] = self._get_M_XX(self.params.hazard.antibody_gain)


def _set_block_names():
    '''Set the names BlockX.X'''

    def set_X(Block):
        '''Set the variable name from the last letter of the class name.'''
        assert Block.__name__.startswith('Block')
        Block.X = Block.__name__.removeprefix('Block')
        abc.update_abstractmethods(Block)

    for _BlockXDE in _Block.__subclasses__():
        for _Block_ in _BlockXDE.__subclasses__():
            set_X(_Block_)


_set_block_names()


def _get_vars_ode_and_vars_pde():
    '''Get the variables governed by ODEs and by PDEs.'''
    vars_ode = tuple(_Block_.X
                     for _Block_ in _BlockODE.__subclasses__())
    vars_pde = tuple(_Block_.X
                     for _Block_ in _BlockPDE.__subclasses__())
    assert set(vars_ode + vars_pde) == set(SHORT_NAMES.values())
    return (vars_ode, vars_pde)


(vars_ode, vars_pde) = _get_vars_ode_and_vars_pde()


def _get_blocks():
    '''Get all `_Block()`s.'''
    return tuple(itertools.chain.from_iterable(
        _BlockXDE.__subclasses__()
        for _BlockXDE in _Block.__subclasses__()))


Blocks = _get_blocks()
