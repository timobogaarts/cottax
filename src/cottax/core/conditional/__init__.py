r'''
Conditional node with a Condition (algebraic, ode, etc.) and a Driver (which checks if it fits the Condition).

Dependency:

condition                        (Condition, the value a node states)
 ||
 \/
driver                           (Driver & DriverContract)
 ||
 \/
node                             (ConditionalNode, its driver fitting it one of its rules:
 ||                               check_driver_fits; kind_of, is_driven)
 ||-------------------------------------
 \/        \/          \/              \/
ops       assign      roles           proof
(SetOwnsInside: what  (Assign /       (Start / Steps,  (Driven, every problem
 a node owns inside)   Undrive;        the shipped      driven -- what an export
                       AssignByRule;   roles;           reads)
                       `driven`, a     role_slots)
                       node built
                       driven)

The front door `cottax.core` gives the node, its condition and `SetOwnsInside`; the driver's
vocabulary is imported from here.
'''

from .condition import Condition
from .driver import Driver, DriverContract, PerCondition, PerUnknown, Role, entry_for
from .node import ConditionalNode, check_driver_fits, is_driven, kind_of
from .ops import SetOwnsInside
from .assign import Assign, AssignByRule, Undrive, driven, driving_violations
from .roles import Start, Steps, role_slots
from .proof import Driven, DrivenNodes

__all__ = [
    'Condition', 'ConditionalNode', 'is_driven', 'kind_of', 'SetOwnsInside',
    'Driver', 'DriverContract', 'PerCondition', 'PerUnknown', 'Role', 'entry_for',
    'Assign', 'AssignByRule', 'Undrive', 'driven', 'driving_violations',
    'Start', 'Steps', 'role_slots',
    'Driven', 'DrivenNodes', 'check_driver_fits',
]
