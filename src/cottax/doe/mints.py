'''The purposes a DOE's declarations derive names for.'''

from ..core import Mint


class Points(Mint):
    '''Where a factor's plan is read unless `over=` names it: `^points.x`.'''


class Record(Mint):
    '''Where a response's record is written unless `record=` names it: `^record.h`.

    Workaround, pending the history rework: named for the DOE alone, and renamed or unified
    with other drivers' histories by that rework.'''
