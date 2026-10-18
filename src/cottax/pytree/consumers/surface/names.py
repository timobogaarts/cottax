'''
The pytree place's names: a node name and a place from each other's keys, and a key one
deeper (`PytreeDerivedNames`), what a path written in full means (`WrittenKeys`),
and what a bare string means, a jax key under a root (`Under`) or nothing, every place written
in full (`InFull`).
'''

from collections.abc import Iterator

from ....core import Violation
from ....core.frozen import answers, choice
from ....pytree.path import DictKey, GetAttrKey, KeyEntry, NodePath, SequenceKey, VarPath
from ....consumers.surface.interface import DerivedNames, Locator, Written
from ....consumers.surface.interface_types import Field, Index, Key, Step
from .builders import Keys, V, Var, keys_of, node_of, place_of


@answers(VarPath, NodePath)
class PytreeDerivedNames(DerivedNames[NodePath, VarPath]):
    '''A node name and a place give each other their keys; a name one deeper is a string key;
    a model tree is read by key path.'''

    def beside(self, place: VarPath, name: str) -> VarPath:
        return VarPath((*place.segments, DictKey(name)))

    def node_at(self, place: VarPath) -> NodePath:
        return NodePath(place.segments)

    def place_at(self, node: NodePath) -> VarPath:
        return VarPath(node.segments)

    def node_in_tree(self, steps: tuple[Step, ...]) -> NodePath:
        return NodePath(tuple(map(key_of, steps)))


@answers(Keys)
class WrittenKeys(Written[NodePath, VarPath]):
    '''A path written in full: a `V` path the place it builds, an `N` path the node name.'''

    def places(self, written: Keys) -> tuple[VarPath, ...] | None:
        return (place_of(written),) if written._into is VarPath else None

    def node(self, written: Keys, names: DerivedNames) -> NodePath | None:
        return node_of(written) if written._into is NodePath else None


class PytreeLocator(Locator[VarPath, NodePath]):
    '''What every pytree locator decides alike: a string where a node is due is a key at the
    root, as a slot in a mapping names one. What a string means where a place is due is each
    final's.'''

    places = VarPath
    nodes  = NodePath

    def node(self, s: str) -> NodePath:
        return NodePath((DictKey(s),))


def key_of(step: Step) -> KeyEntry:
    '''The jax key a step down a model tree is: a field a `GetAttrKey`, a key a `DictKey`, an
    index a `SequenceKey`.'''
    match step:
        case Field(name):
            return GetAttrKey(name)
        case Key(key):
            return DictKey(key)
        case Index(index):
            return SequenceKey(index)
    raise TypeError(f'{step!r} is no step down a model tree')


def check_name_key(locator: 'Under') -> Iterator[Violation]:
    if locator.key not in (GetAttrKey, DictKey):
        yield NotANameKey(repr(locator.key))


@choice(Locator)
class Under(PytreeLocator):
    '''A place string one key under a root, the key a `GetAttrKey` or a `DictKey`.'''

    root : Var
    key  : type[GetAttrKey] | type[DictKey]

    checks = (check_name_key,)

    def place(self, s: str) -> VarPath:
        return VarPath((*keys_of(self.root), self.key(s)))

    @property
    def V(self) -> Keys:
        '''A builder at the root: any path relative to it, in `V`'s syntax, where a string can
        say only one name: `tfcoil.locator.V.coils[0].j`.'''
        return Keys(VarPath, keys_of(self.root))


@choice(Locator)
class InFull(PytreeLocator):
    '''No place strings: every place written in full, a `V` path or a `VarPath`.'''

    def place(self, s: str) -> VarPath:
        raise TypeError(f'{s!r}: a string names a place only in a scope (`cx.flat`, '
                        f'`cx.Scope`) -- write it with `V`')

    @property
    def V(self) -> Keys:
        '''`V` itself: the root is the root.'''
        return V


class NotANameKey(Violation):
    key : str
    error    = TypeError
    template = ('a scope writes a name as a `GetAttrKey` or a `DictKey`, not {key}: any '
                'other key is written in a `V` path')
