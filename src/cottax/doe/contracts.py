'''
Driver contracts: the level `Sampler`, its final `Sampling` (the interior evaluated at every
point of the plans), and `CONTRACTS`, the default.
'''

from ..core import Place
from ..core.driven import DriverContract
from .doe_condition import is_doe


class Sampler[V: Place](DriverContract[V]):
    '''Answers a DOE by evaluating its cycle at each point of its plans.'''
    accepts = staticmethod(is_doe)


class Sampling[V: Place](Sampler[V]):
    '''Every point, `i = 1..n`. It takes nothing, `n` being the plans' length, and reports
    nothing, the records being the condition's own until the history rework (`TODO.md`).'''
    takes   = ()
    reports = ()


CONTRACTS = (Sampling(),)
'''The default contracts: an implementation answers a problem by the first one it implements.'''
