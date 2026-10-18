r'''
How a problem is answered, before anything runs it: drivers, driven nodes, their ops.

Dependency:

driver                           (the contract: DriverContract, Role / PerUnknown / PerProblem, arity;
                                  entry_for, a table's entry for a contract's most specific class)
 ||---------
 \/        \/
nodes     roles                  (nodes: DrivenNode, the level; Driven, its final; condition_of;
 ||                               role_slots, where each problem takes or reports a role, by unknown;
 ||                               roles: Start / Steps, the places the shipped contracts take)
 ||---------
 \/        \/
ops       proof                  (ops: Assign / Undrive; AssignByRule, an Assign at the places
                                  a rule gives; proof: DrivenGraph, every problem driven --
                                  what an export reads)

'''

from .driver import DriverContract, PerProblem, PerUnknown, Role, entry_for
from .nodes import Driven, DrivenNode, condition_of, is_driven, role_slots
from .ops import Assign, AssignByRule, Undrive
from .roles import Start, Steps
from .proof import DrivenGraph

__all__ = [
    'DriverContract', 'PerProblem', 'PerUnknown', 'Role', 'entry_for',
    'DrivenNode', 'Driven', 'is_driven', 'condition_of', 'role_slots', 'Assign', 'AssignByRule', 'Undrive',
    'DrivenGraph', 'Start', 'Steps',
]
