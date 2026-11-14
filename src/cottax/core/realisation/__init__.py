r'''
What runs a node, at any level, a driven one included. Takes `core` and `driven/`. Generic
in the place, and no backend's.

Dependency:

driven/                          (DrivenNode: a driven node is implemented by its contract)
 ||
 \/
implemented                      (Implementation, the attribute a node carries; implemented /
 ||                               implementation_of / is_implemented; Implement /
 \/                               Unimplement, ImplementDrivers)
proof                            (RealisedGraph over driven.DrivenGraph: every node implemented)

'''

from .implemented import (
    Implement, Implementation, Unimplement, ImplementDrivers,
    implementation_of, implemented, is_implemented,
)
from .proof import RealisedGraph

__all__ = [
    'Implementation', 'implemented', 'implementation_of',
    'is_implemented', 'Implement', 'Unimplement', 'ImplementDrivers',
    'RealisedGraph',
]
