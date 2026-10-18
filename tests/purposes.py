'''The purposes the tests derive names for: the shipped ones, by the modules declaring them,
and a few of the tests' own.'''

from cottax.core import Closure, Copy, Mint
from cottax.consumers.surface.interface_types import Given, Start, Steps
from cottax.consumers.declaration.interface_types import Solve
from cottax.relational.mints import Cond, Lower, Require, Target, Upper
from cottax.ode.mints import Initial, Rate, Span, Times, Trajectory


class Guess(Mint):
    '''A start of the tests' own, beside the shipped `^start`.'''


class Twin(Mint):
    '''A cut's copy of the tests' own, beside the shipped `^copy`.'''


class Budget(Mint):
    '''Where a test driver's budget is read.'''


__all__ = ['Given', 'Copy', 'Closure', 'Solve', 'Start', 'Steps', 'Cond', 'Lower', 'Require',
           'Target', 'Upper', 'Initial', 'Rate', 'Span', 'Times', 'Trajectory', 'Guess', 'Twin',
           'Budget']
