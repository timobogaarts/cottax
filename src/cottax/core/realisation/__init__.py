r'''
What runs a node, at any level, a driven one included. Takes `core` and `driven/`: a
driven node's wrapper joins its level. Generic in the place, and no backend's.

Dependency:

driven/                          (DrivenNode, the level ImplementedDriven joins)
 ||
 \/
implemented                      (Implemented, the level; ImplementedFunction /
 ||                               ImplementedDriven, picked by the node's level in
 ||                               implemented_as; Implement / Unimplement,
 \/                               ImplementDrivers)
proof                            (RealisedGraph over driven.DrivenGraph: every node implemented)

'''

from .implemented import (
    Implement, Implemented, ImplementedDriven, ImplementedFunction, Unimplement,
    ImplementDrivers, NotImplementable, implemented_as, is_implemented,
)
from .proof import RealisedGraph

__all__ = [
    'Implemented', 'ImplementedFunction', 'ImplementedDriven', 'implemented_as',
    'is_implemented', 'Implement', 'Unimplement', 'ImplementDrivers', 'NotImplementable',
    'RealisedGraph',
]
