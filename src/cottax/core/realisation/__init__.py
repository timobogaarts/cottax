r'''
What runs a node, at any level, a driven one included. Takes `core` and `driven/`. Generic
in the place, and no backend's.

Dependency:

driven/                          (is_driven: a driven node is implemented by what runs its driver)
 ||
 \/
implemented                      (Implementation, the attribute a node carries; implemented /
 ||                               implementation_of / is_implemented; Implement /
 ||                               Unimplement, ImplementDrivers; the rule an implementation
 \/                               fits its node by, check_implementation_fits)
proof                            (Realised over driven.Driven: every node implemented,
                                  each where it fits)

'''

from .implemented import (
    Implement, Implementation, Unimplement, ImplementDrivers,
    check_implementation_fits, implementation_of, implemented, is_implemented,
)
from .proof import Realised, RealisedNodes

__all__ = [
    'Implementation', 'implemented', 'implementation_of',
    'is_implemented', 'check_implementation_fits', 'Implement', 'Unimplement', 'ImplementDrivers',
    'Realised', 'RealisedNodes',
]
