'''The purposes relational's declarations derive names for.'''

from ..core import Mint


class Cond(Mint):
    '''What an implicit model's function writes per unknown, the condition on it: `^cond.u`.'''


class Require(Mint):
    '''Where the requirement beside a constraint binds: `^require.at`.'''


class Lower(Mint):
    '''A lower bound written as a value: `^lower.u`.'''


class Upper(Mint):
    '''An upper bound written as a value: `^upper.u`.'''


class Target(Mint):
    '''An equality's target written as a value: `^target.h`.'''


class Scale(Mint):
    '''The magnitude something is read in: an objective's, attached to it, `^scale.f`; a
    relation's residual, attached to it, off the place it is named by, `^scale.^cond.^copy.x`.'''
