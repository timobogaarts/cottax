'''
The roles the shipped contracts give their places. A role is a class, never a value; its
`arity` is its kind's.
'''

from .driver import PerProblem, PerUnknown


class Start(PerUnknown):
    '''Where each unknown starts.'''


class Steps(PerProblem):
    '''Reported: how many iterations the solve took.'''
