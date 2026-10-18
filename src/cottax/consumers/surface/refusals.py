'''Why what is written is no name: neither a place nor a node, several places where one is
due, a purpose no word or several words name, or a role with no place.'''

from ...core import Violation


class NotAPlace(Violation):
    role    : str
    written : str
    error    = TypeError
    template = ('{role}: a place is a string, or a path written in full (`V.x`, a `VarPath`), '
                'and {written} is neither')


class StandsForSeveral(Violation):
    role    : str
    written : str
    places  : tuple
    error    = ValueError
    template = ('{role}: {written} stands for {places!r}, where one place is due -- name one')


class NotANode(Violation):
    role    : str
    written : str
    error    = TypeError
    template = ('{role}: a node is a string, a path written in full (`N.x`, a `NodePath`) or '
                'a declaration, and {written} is none of these')


class NoPlaceFor(Violation):
    role : str
    error    = KeyError
    template = ('{role} has no place: give it a rule in `role_places`, or pass its places to '
                '`Assign`')


__all__ = ['NotAPlace', 'StandsForSeveral', 'NotANode', 'NoPlaceFor']
