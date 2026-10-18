'''
Why an architecture is refused: a top the model does not hold or that is no problem, two problems sharing a body
with neither inside the other, what it would fold combining as another kind, a model
already nested; and two closures bound at one name. Why a problem is no implicit model
is its language's.
'''

from ...core import Violation


class TopUnbound(Violation):
    top : str
    error    = KeyError
    template = ('{top} is no node of the model: the top is a problem the model declares -- '
                'declare it with the rest, or name the slot it is in')


class TopNotAProblem(Violation):
    top : str
    error    = ValueError
    template = '{top} is not a problem: an architecture resolves what lies under a problem'


class NotCombinable(Violation):
    problem      : str
    kind         : str
    architecture : str
    into         : str
    error    = ValueError
    template = ('{problem} combines as {kind}, and {architecture} combines it into {into}: '
                'only what combines as one kind is combined')


class BodyShared(Violation):
    one    : str
    two    : str
    body   : str
    remedy : str
    error    = ValueError
    template = '{one} and {two} share the body {body}: one solve written as two -- {remedy}'


class NestedAlready(Violation):
    pairs : tuple
    error    = ValueError
    template = ('{pairs!r} already nested: an architecture places every problem itself, '
                'over a model with no nesting -- `Unnest` first, or start from the model')


class ClosuresCollide(Violation):
    names : tuple
    error    = ValueError
    template = ('two closures bound at {names!r}: the `At` names two groups alike -- '
                'one closure per cycle under `AtFirstMember`: cut it as one group, or write an `At` naming them apart')


__all__ = ['TopUnbound', 'TopNotAProblem', 'NotCombinable', 'BodyShared', 'NestedAlready', 'ClosuresCollide']
