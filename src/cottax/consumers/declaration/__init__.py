'''
Declaring a model: functions wrapped where they are written or used, and the nodes they
declare. A value written where a place is due becomes a given boundary input, and a tag on
a node or a variable is carried into the `Declared`.
'''

from .interface_types import Description, Explicit, Unimplemented
from .conventions import Conventions, ParameterReads, Shipped, per_place_of, term_of
from .declared import Declared, declare, loose_at, tags_of

__all__ = [
    'Conventions', 'Shipped', 'ParameterReads', 'term_of', 'per_place_of',
    'Description', 'Unimplemented', 'Explicit', 'Declared', 'declare', 'loose_at', 'tags_of',
]
