'''
Why an architecture is refused: a top the model does not hold or that is no problem, two problems sharing a body
with neither inside the other, what it would fold combining as another kind, a model
already nested; and two closures bound at one name. Why a `Combine` is refused, and why a
cut rule is. Why a problem is no implicit model is its language's.
'''

from ...core import Label, Violation


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




class NothingToCombine(Violation):
    error    = ValueError
    template = 'combine(): fewer than two nodes, so nothing to combine'

class NodeTwice(Violation):
    nodes : tuple
    error    = ValueError
    template = 'combine(): {nodes!r} names a node twice'

class NotAddable(Violation):
    kind : str
    error    = TypeError
    template = ('combine(): nothing answers a join of {kind}, so no two of them can be folded '
                '-- a kind folds where its `ModelReading` answers `join`')

class NotOfKind(Violation):
    nodes : tuple
    kind  : str
    found : str
    error    = TypeError
    template = ('combine({kind}): {nodes!r} is {found}, and a fold is well-defined over '
                'one kind only')

class CarriesAttributes(Violation):
    node       : object
    attributes : tuple[str, ...]
    error    = TypeError
    template = ('combine(): {node!r} carries {attributes}, and what two nodes carry has no join '
                '-- take it off first')

class FoldsDriven(Violation):
    nodes : tuple
    error    = TypeError
    template = ('{nodes} carry a driver: which algorithm answers a join is not read off the two, '
                'so a fold is of conditions -- `Undrive` them, combine, then drive the fold')


class OrderMismatch(Violation):
    order : tuple
    nodes : tuple
    error    = ValueError
    template = ('gauss_seidel: the order {order!r} does not name the nodes {nodes!r} exactly '
                'once each -- an order sweeps one cycle, and one per cycle is `ByName`')


class CycleTooLarge(Violation):
    size  : int
    limit : int
    error    = ValueError
    template = ('{size} nodes on one cycle is beyond the exact order search (limit {limit}): '
                'raise `GaussSeidelMinimal(limit=)`, or hand `GaussSeidelOrdered` an order')


class KeyRepeats(Violation):
    keys : tuple
    error    = ValueError
    template = 'by_name: {keys!r} each key more than one rule'


class NoRule(Violation):
    nodes : tuple
    error    = ValueError
    template = ('no rule keyed by a node of {nodes!r}: key one, `NoCuts()` to leave it '
                'alone, or use the one rule alone for every cycle')


class SeveralRules(Violation):
    keyed : tuple
    error    = ValueError
    template = ('several rules keyed on one cycle, by {keyed!r}: cycles are disjoint, so '
                'one key lies on one')


__all__ = ['TopUnbound', 'TopNotAProblem', 'NotCombinable', 'BodyShared', 'NestedAlready', 'ClosuresCollide',
           'NothingToCombine', 'NodeTwice', 'NotAddable', 'NotOfKind', 'CarriesAttributes', 'FoldsDriven',
           'OrderMismatch', 'CycleTooLarge', 'KeyRepeats', 'NoRule', 'SeveralRules']
