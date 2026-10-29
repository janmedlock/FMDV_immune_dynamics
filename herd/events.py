'''Generate all the `.event.Event()` subclasses.'''

__all__ = (
    'antibody_gain', 'antibody_loss', 'birth', 'carrier_recovery',
    'infection', 'maternal_immunity_waning', 'mortality',
    'progression', 'recovery',
)

from . import (antibody_gain, antibody_loss, birth, carrier_recovery,
               infection, maternal_immunity_waning, mortality,
               progression, recovery)
