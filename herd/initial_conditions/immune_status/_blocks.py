'''Blocks used by solver.Solver().'''

import abc
import re

import numpy
import numpy.lib.recfunctions
from scipy import sparse


# The short & long names of the variables.
# The order of the output is determined by the order of these, too.
NAMES = {'M': 'maternal immunity',
         'S': 'susceptible',
         'E': 'exposed',
         'I': 'infectious',
         'C': 'chronic',
         'R': 'recovered',
         'L': 'lost immunity'}


class _Block(metaclass=abc.ABCMeta):
    '''Get a block row A_X of A and block b_X of b.'''

    def __init__(self, params):
        self.params = params
        self._set_A_XY_regex = re.compile(f'set_A_{self.X}([A-Z])$')
        self.set_A_X()
        self.set_b_X()

    @property
    @abc.abstractmethod
    def X(self):
        '''The variable name.'''
        # This gets set automatically by `_set_X()` at the bottom of
        # this file.

    def _set_b_X(self, initial_value):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # The initial value is in the 0th entry,
        # then zeros everywhere else.
        self.b_X = sparse.csr_matrix(([initial_value], ([0], [0])),
                                     shape=(len(self), 1))

    @abc.abstractmethod
    def set_b_X(self):
        '''Make a sparse n × 1 matrix of the right-hand sides.'''
        # This should call `self._set_b_X(initial_value)`.
        raise NotImplementedError

    def _is_set_A_XY(self, attr):
        '''Check if `attr` is 'set_A_XY' for some Y.'''
        return (self._set_A_XY_regex.match(attr) is not None)

    def set_A_X(self):
        '''Assemble the block row A_X from its columns A_XY.'''
        self.A_X = {}
        for attr in dir(self):
            if self._is_set_A_XY(attr):
                getattr(self, attr)()

    def update(self):
        '''Do nothing unless overridden.'''


class _BlockODE(_Block, metaclass=abc.ABCMeta):
    '''A `_Block()` for a variable governed by an ODE.'''
    def __len__(self):
        return self.params.length_ode

    def _get_A_XX(self, hazard_out):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        d_X = ((hazard_out + self.params.hazard.mortality)
               * self.params.step / 2)
        diags = ((numpy.hstack([1, 1 + d_X]), 0),  # The diagonal
                 (- 1 + d_X, -1))  # The subdiagonal
        # Ensure that the off-diagonal entries are non-positive.
        for (v, k) in diags:
            if k != 0:
                assert (v <= 0).all(), type(self)
        return sparse.diags(*zip(*diags),
                            shape=(len(self), len(self)))

    def _get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        f_XY = - hazard_in * self.params.step / 2
        diags = ((numpy.hstack([0, f_XY]), 0),  # The diagonal
                 (f_XY, -1))  # The subdiagonal
        return sparse.diags(*zip(*diags),
                            shape=(len(self), self.params.length_ode))

    def _get_A_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_pde))
        for i in range(1, len(self)):
            j = numpy.arange(i + 1)
            A_XY[i, j] = - (pdf_in[i - j]
                            * self.params.survival.mortality[i]
                            / self.params.survival.mortality[j]
                            * self.params.step ** 2)
        return A_XY


class _BlockPDE(_Block, metaclass=abc.ABCMeta):
    '''A `_Block()` for a variable governed by a PDE.'''
    def __len__(self):
        return self.params.length_pde

    def _get_A_XX(self):
        '''Get the diagonal block `A_XX` that maps state X to itself.'''
        return sparse.eye(len(self))

    def _get_A_XY_ODE(self, hazard_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by an ODE.'''
        # The values on the diagonal and subdiagonal.
        f_XY = - hazard_in / 2
        diags = ((numpy.hstack([0, f_XY]), 0),  # The diagonal
                 (f_XY, -1))  # The subdiagonal
        return sparse.diags(*zip(*diags),
                            shape=(len(self), self.params.length_ode))

    def _get_A_XY_PDE(self, pdf_in):
        '''Get the off-diagonal block `A_XY` that maps state Y to X,
        where Y is a variable governed by a PDE.'''
        A_XY = sparse.lil_matrix((len(self), self.params.length_pde))
        for i in range(1, len(self)):
            j = numpy.arange(i + 1)
            A_XY[i, j] = - (pdf_in[i - j]
                            * self.params.survival.mortality[i]
                            / self.params.survival.mortality[j]
                            * self.params.step)
        return A_XY

    def set_b_X(self):
        self._set_b_X(0)

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
    def set_A_MM(self):
        self.A_X['M'] = self._get_A_XX(
            self.params.hazard.maternal_immunity_waning)

    def set_b_X(self):
        self._set_b_X(self.params.newborn_proportion_immune)

    def update(self):
        self.set_b_X()


class BlockS(_BlockODE):
    def set_A_SM(self):
        self.A_X['M'] = self._get_A_XY_ODE(
            self.params.hazard.maternal_immunity_waning)

    def set_A_SS(self):
        self.A_X['S'] = self._get_A_XX(self.params.hazard.infection)

    def set_b_X(self):
        self._set_b_X(1 - self.params.newborn_proportion_immune)

    def update(self):
        self.set_A_SS()
        self.set_b_X()


class BlockE(_BlockPDE):
    def set_A_ES(self):
        self.A_X['S'] = self._get_A_XY_ODE(self.params.hazard.infection)

    def set_A_EL(self):
        self.A_X['L'] = self._get_A_XY_ODE(
            self.params.lost_immunity_susceptibility
            * self.params.hazard.infection)

    def set_A_EE(self):
        self.A_X['E'] = self._get_A_XX()

    def update(self):
        self.set_A_ES()
        self.set_A_EL()

    def integrate(self, p_E):
        return super().integrate(p_E, self.params.survival.progression)


class BlockI(_BlockPDE):
    def set_A_IE(self):
        self.A_X['E'] = self._get_A_XY_PDE(self.params.pdf.progression)

    def set_A_II(self):
        self.A_X['I'] = self._get_A_XX()

    def integrate(self, p_I):
        return super().integrate(p_I, self.params.survival.recovery)


class BlockC(_BlockPDE):
    def set_A_CI(self):
        self.A_X['I'] = self._get_A_XY_PDE(self.params.probability_chronic
                                           * self.params.pdf.recovery)

    def set_A_CC(self):
        self.A_X['C'] = self._get_A_XX()

    def integrate(self, p_C):
        return super().integrate(p_C, self.params.survival.chronic_recovery)


class BlockR(_BlockODE):
    def set_A_RI(self):
        self.A_X['I'] = self._get_A_XY_PDE(
            (1 - self.params.probability_chronic)
            * self.params.pdf.recovery)

    def set_A_RC(self):
        self.A_X['C'] = self._get_A_XY_PDE(self.params.pdf.chronic_recovery)

    def set_A_RR(self):
        self.A_X['R'] = self._get_A_XX(self.params.hazard.antibody_loss)

    def set_A_RL(self):
        self.A_X['L'] = self._get_A_XY_ODE(self.params.hazard.antibody_gain)

    def set_b_X(self):
        self._set_b_X(0)


class BlockL(_BlockODE):
    def set_A_LR(self):
        self.A_X['R'] = self._get_A_XY_ODE(self.params.hazard.antibody_loss)

    def set_A_LL(self):
        self.A_X['L'] = self._get_A_XX(
            self.params.hazard.antibody_gain
            + (self.params.lost_immunity_susceptibility
               * self.params.hazard.infection))

    def set_b_X(self):
        self._set_b_X(0)

    def update(self):
        self.set_A_LL()


def _set_X(Block):
    '''Set the variable name from the last letter of the class name.'''
    assert len(Block.__name__) == (5 + 1)
    assert Block.__name__[:5] == 'Block'
    Block.X = Block.__name__[-1]
    abc.update_abstractmethods(Block)


# Set the names BlockX.X
for _BlockXDE in _Block.__subclasses__():
    for _Block_ in _BlockXDE.__subclasses__():
        _set_X(_Block_)


# Get the variables governed by ODEs
vars_ode = [_Block_.X
            for _Block_ in _BlockODE.__subclasses__()]

# Get the variables governed by PDEs
vars_pde = [_Block_.X
            for _Block_ in _BlockPDE.__subclasses__()]

# Make sure `NAMES` agrees with `vars_ode` and `vars_pde`
assert set(vars_ode + vars_pde) == NAMES.keys()


# Get all `_Block()`s.
Blocks = []
for _BlockXDE in _Block.__subclasses__():
    Blocks.extend(_BlockXDE.__subclasses__())
