'''A tree of models declared: walked in written order, planned with the values and tags written beside it (`declare`, `Declared`).'''

from collections.abc import Callable, Iterator, Mapping
import dataclasses
from typing import Any

from ...core import (
    Label, NodeDefinition, Place, Violation, AbstractGraph, NestingGraph, GraphOp, Plan,
)
from ...core.frozen import Frozen, OrderedMap, answer, answers, checked
from ..surface.interface import DerivedNames, Written
from ..surface.interface_types import Field, Index, Key, Step
from .interface_types import Declaration, Tagged, Tags, Wrapper, merged_tags, untagged
from .refusals import HeldNotInModel, Nameless, NotAnInput, Unnamed


type Steps = tuple[Step, ...]

# ---------------------------------------------------------------- what the tree declares

class Declared[N: Label, V: Place](Frozen):
    '''
    A plan, the values given at its boundary inputs, and the tags written on its places and
    on its node names. An op edits the plan alone: a value or a tag at a place the plan
    renames away is left where it was written, and a run takes no value nothing reads.
    '''

    plan       : Plan
    given      : OrderedMap[V, Any] = dataclasses.field(compare=False)
    place_tags : Tags[V] = dataclasses.field(default=OrderedMap(), compare=False)
    node_tags  : Tags[N] = dataclasses.field(default=OrderedMap(), compare=False)

    @property
    def graph(self) -> NestingGraph[N, V]:
        return self.plan.graph

    def __add__(self, op: GraphOp) -> 'Declared[N, V]':
        return dataclasses.replace(self, plan=self.plan + op)

    def using(self, values: Mapping[V, Any]) -> 'Declared[N, V]':
        '''The same, with a value given at each of `values`, replaced or added: each a
        boundary input of the graph.'''
        inputs = set(self.graph.graph.boundary_inputs)
        if stray := tuple(p for p in values if p not in inputs):
            raise NotAnInput(stray).raised
        return dataclasses.replace(self, given=OrderedMap({**self.given, **values}))

# ---------------------------------------------------------------- a tree of models walked

def _check_held_in_model(*models: Any) -> Iterator[Violation]:
    '''A declaration another holds is a leaf too: what stays when held is bound at its name.'''
    found = [leaf for _, leaf, _ in _leaves(*models)]
    in_model = {id(leaf) for leaf in found}
    if missing := tuple(repr(c) for leaf in found if isinstance(leaf, Declaration)
                        for c in leaf.holds
                        if isinstance(c, Declaration) and id(c) not in in_model):
        yield HeldNotInModel(missing)


@checked(_check_held_in_model)
def declare(*models: Any) -> Declared:
    '''Everything the models declare, each at the node name its position in the tree gives,
    as a plan, with the values and tags written beside it: `declare(d1, d2, {'opt': opt})`.
    A declaration held by another keeps only `nodes_when_held`, a held node nothing.'''
    found = _leaves(*models)
    held = {id(c) for _, leaf, _ in found if isinstance(leaf, Declaration) for c in leaf.holds}
    places = (p for _, leaf, _ in found for p in (
        leaf.outputs if isinstance(leaf, Declaration) else (*leaf.owns, *leaf.reads)))
    if (first := next(places, None)) is None:
        if found:
            raise Nameless(tuple(leaf for _, leaf, _ in found)).raised
        return Declared(Plan(NestingGraph(AbstractGraph(OrderedMap()), OrderedMap())),
                        OrderedMap())
    names = answer(DerivedNames, first)
    nodes, node_tags = [], []
    for steps, leaf, tags in found:
        at = names.node_in_tree(steps) if steps is not None else loose_at(leaf, names)
        if isinstance(leaf, Declaration):
            nodes.extend(leaf.nodes_when_held(at) if id(leaf) in held else leaf.nodes(at))
        elif id(leaf) not in held:
            nodes.append((at, leaf))
        node_tags.append((at, tags))
    graph = NestingGraph(AbstractGraph(OrderedMap(nodes)), OrderedMap())
    declarations = [leaf for _, leaf, _ in found if isinstance(leaf, Declaration)]
    return Declared(Plan(graph),
                    OrderedMap(p_v for leaf in declarations for p_v in leaf.given.items()),
                    merged_tags(p_t for leaf in declarations for p_t in leaf.tags.items()),
                    merged_tags(node_tags))
# --------------------------------------------- internal helpers

def _leaves(*models: Any
           ) -> list[tuple[Steps | None, Declaration | NodeDefinition, tuple[Tagged, ...]]]:
    '''Every declaration and node the models hold, with the steps to it (`None` for one
    handed over loose) and the tags written around it: each of their top-level entries
    (`_join`) walked under its step. Anything else in a slot is refused.'''
    found = [(None if step is None else (step, *steps), leaf) for step, child in _join(*models)
             for steps, leaf in _walk(child, _is_model)]
    if strays := [leaf for _, leaf in found if not _is_model(leaf)]:
        raise TypeError(f'{strays[0]!r} is not a model: a slot holds a wrapper, a node '
                        f'definition, None or a tree of those -- a setting is a place, read '
                        f'like any other')
    return [(steps, *untagged(leaf)) for steps, leaf in found]


def _walk(tree: Any, is_leaf: Callable[[Any], bool]) -> list[tuple[Steps, Any]]:
    '''Every leaf with its steps, a leaf being what `is_leaf` says or anything that is no
    container, and `None` holding nothing: a mapping's keys in the order written, a dataclass
    by its public fields.'''
    out = []
    def down(x: Any, steps: Steps) -> None:
        children = None if is_leaf(x) else _children(x, is_leaf)
        if children is None:
            out.append((steps, x))
        for step, child in children or ():
            down(child, (*steps, step))
    down(tree, ())
    return out


def _join(*models: Any) -> list[tuple[Step | None, Any]]:
    '''The top-level entries of several models, side by side: a model handed over loose
    under `None`, named by itself (`Declaration.loose_at`), and every other tree's own
    entries. `_join(disc1, {'opt': opt})` is `[(None, disc1), (Key('opt'), opt)]`.'''
    entries = []
    for model in models:
        if _is_model(model):
            entries.append((None, model))
        elif (children := _children(model, _is_model)) is None:
            raise TypeError(f'{model!r} is not a model: a wrapper, or a tree of models')
        else:
            entries.extend(children)
    return entries


def _children(x: Any, is_leaf: Callable[[Any], bool]) -> list[tuple[Step, Any]] | None:
    '''What a container holds, one step deeper, in written order; `None` for no container.'''
    if x is None:
        return []
    if isinstance(x, Mapping):
        return [(Key(k), v) for k, v in x.items()]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        children = []
        for f in dataclasses.fields(x):
            value = getattr(x, f.name)
            if f.init and not f.name.startswith('_'):
                children.append((Field(f.name), value))
            elif any(is_leaf(leaf) for _, leaf in _walk(value, is_leaf)):
                raise TypeError(f'{type(x).__name__}.{f.name} holds a model but is no slot '
                                f'(private, or not in `__init__`): make it a public field')
        return children
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        return [(Field(f), getattr(x, f)) for f in x._fields]
    if isinstance(x, (tuple, list)):
        return [(Index(i), v) for i, v in enumerate(x)]
    return None


def _is_model(x: Any) -> bool:
    '''A declaration or a node, tagged or not.'''
    return isinstance(untagged(x)[0], (Declaration, NodeDefinition))




def loose_at[N](model: Declaration | NodeDefinition, names: DerivedNames[N, Any]) -> N:
    '''The node name a model handed over loose takes, refused where it says none.'''
    at = model.loose_at(names) if isinstance(model, Declaration) else None
    if at is None:
        raise Unnamed(repr(model)).raised
    return at


class WrittenDeclaration(Written):
    '''A declaration written where a name is due: as a node name, the one it takes handed over
    loose (`loose_at`, refused where it says none).'''

    def node(self, written: Declaration, names: DerivedNames) -> Any:
        return loose_at(written, names)


@answers(Wrapper)
class WrittenWrapper(WrittenDeclaration):
    '''A wrapper stands for its outputs.'''

    def places(self, written: Wrapper) -> tuple[Any, ...]:
        return written.outputs


@answers(Declaration)
class WrittenProblem(WrittenDeclaration):
    '''Any other declaration, a problem, stands for no place.'''

    def places(self, written: Declaration) -> None:
        return None


def tags_of[T: Tagged](declared: Declared, name: Any, kind: type[T]) -> tuple[T, ...]:
    '''The tags of `kind` written on `name`, a place or a node name.'''
    written = (*declared.place_tags.get(name, ()), *declared.node_tags.get(name, ()))
    return tuple(t for t in written if isinstance(t, kind))


__all__ = ['Declared', 'declare', 'loose_at', 'tags_of']
