r'''
How a problem is answered, before anything runs it: what a driver is, the edits putting one
on a condition node and taking it off, and the proof that every problem has one. A condition
node carries its driver as a field; the maths knows it only as an `Attachment`.

Dependency:

driver                           (Driver, the attachment: a contract and where its roles sit;
 ||                               DriverContract, Role / PerUnknown / PerProblem, arity;
 ||                               entry_for, a table's entry for a contract's most specific class)
 \/
roles                            (Start / Steps, the places the shipped contracts take;
 ||                               role_slots, where each problem takes or reports a role)
 ||---------
 \/        \/
ops       proof                  (ops: Assign / Undrive; AssignByRule, an Assign at the places
                                  a rule gives; `driven`, a node built driven; proof:
                                  DrivenGraph, every problem driven -- what an export reads)

'''

from .driver import Driver, DriverContract, PerProblem, PerUnknown, Role, entry_for
from .roles import Start, Steps, role_slots
from .ops import Assign, AssignByRule, Undrive, driven, driving_violations
from .proof import DrivenGraph, DrivenNodes, check_driver_fits

__all__ = [
    'Driver', 'DriverContract', 'PerProblem', 'PerUnknown', 'Role', 'entry_for', 'role_slots', 'driven', 'driving_violations', 'Assign', 'AssignByRule', 'Undrive',
    'DrivenGraph', 'DrivenNodes', 'check_driver_fits', 'Start', 'Steps',
]
