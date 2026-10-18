'''The purposes the ODE's declarations derive names for, and its roles' places.'''

from ..core import Mint


class Rate(Mint):
    '''What a rate's function writes per state, the state's rate: `^rate.x`.'''


class Initial(Mint):
    '''Where a state's initial value is read unless `initial=` names it: `^initial.x`.'''


class Span(Mint):
    '''Where the span is read unless `span=` names it: `^span.<the first state>`.'''


class Times(Mint):
    '''Where the save times are read, written as a value: `^times.<the first state>`, or
    `^times.<the problem>` for a driver's.'''


class Trajectory(Mint):
    '''Where a state's history is written unless `record=` names it: `^trajectory.x`.'''
