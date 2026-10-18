'''
What a declaration is written with and built on: a term as written (`Side`), a tag or a value
beside a place (`Tagged`, `PlaceWithValue`), a function described (`Description`), what a
vocabulary's kinds extend (`Declaration`, `Wrapper`, `Paired`) and cottax's own (`Explicit`),
and a kind as a declaring part's method (`declares`, reading its `Keyword`s).
'''

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
import dataclasses
import inspect
import sys
from typing import TYPE_CHECKING, Any, overload

from ...core import ConditionalNode, Function, Label, Mint, NodeDefinition, Place, Violation, minted
from ...core.frozen import AbstractClassVar, AbstractVar, Frozen, OrderedMap, answer, name_of
from ...core.realisation import ImplementedFunction
from ..surface.interface_types import Given, Key, Named, Surface
from .refusals import HeldByValue, NoOutputs, NotPositional, ReadsUnlike, Unnamed

if TYPE_CHECKING:                       # the interface is built on this
    from ..surface.interface import DerivedNames
    from .interface import Declaring


# ============================================================================ what is written

type Side[P] = Named[P] | Any
'''A term written where a place is due: a name, a path, or a value (a number, an array, a
tagged one).'''


class Tagged(Frozen):
    '''
    What is written (a name, a path, a value or a model in a slot) with a tag the
    declarations carry but never read: the feature that gives the tag its meaning reads it
    (`Static`: part of the compile key). Around a name or a path the place carries it,
    around a value the place minted for it, around a model the node name its slot gives.
    '''
    value : Any


def untagged(x: Any) -> tuple[Any, tuple[Tagged, ...]]:
    '''What is written under its tags, and the tags, outermost first, each without what it
    wraps.'''
    tags = []
    while isinstance(x, Tagged):
        tags.append(dataclasses.replace(x, value=None))
        x = x.value
    return x, tuple(tags)


class PlaceWithValue[V: Place](Frozen):
    '''A place, the value written for it (`None` where none was), and the tags written on
    the place.'''

    place : V
    value : Any | None = dataclasses.field(default=None, compare=False)
    tags  : tuple[Tagged, ...] = dataclasses.field(default=(), compare=False)


def given_in[V: Place](*groups: Iterable[PlaceWithValue[V] | None]) -> OrderedMap[V, Any]:
    '''Every value written among `groups`, by its place: what a declaration gives.'''
    return OrderedMap((x.place, x.value) for g in groups for x in g
                      if x is not None and x.value is not None)


type Tags[K] = OrderedMap[K, tuple[Tagged, ...]]
'''The tags written on each place, or each node name.'''


def merged_tags[K](pairs: Iterable[tuple[K, tuple[Tagged, ...]]]) -> Tags[K]:
    '''Each name's tags, once each, in the order written; a name with none left out.'''
    out: dict[K, tuple[Tagged, ...]] = {}
    for name, tags in pairs:
        if tags:
            out[name] = tuple(dict.fromkeys((*out.get(name, ()), *tags)))
    return OrderedMap(out)


class Description(Frozen):
    '''
    A function described rather than given: what a wrapper needs of it, its parameters in
    order. Its node is implemented by the description (a `RepoRef` to code elsewhere, say),
    which no backend runs until one resolves it.
    '''
    parameters : AbstractVar[tuple[str, ...]]


class Unimplemented(Description):
    '''A function yet to be written: its parameters, and nothing else. Its node has no
    implementation.'''
    parameters : tuple[str, ...]


type Body = Callable[..., Any] | Description
'''What a wrapper wraps: a function, or a description of one.'''


# ============================================================================ a function's signature

def parameters_of(fn: Body) -> tuple[str, ...]:
    '''The parameters of a function, in order: a description's, or a callable's signature.'''
    if isinstance(fn, Description):
        return tuple(fn.parameters)
    try:
        return tuple(inspect.signature(fn).parameters)
    except ValueError:
        raise TypeError(f'{name_of(fn)} has no signature to read: wrap it in a `def`') from None


def check_signature(wrapper: 'Wrapper') -> Iterator[Violation]:
    '''Each parameter a read, bound by position, and one place in `function_reads` for each.'''
    fn = name_of(wrapper.fn)
    if not isinstance(wrapper.fn, Description):
        for p in inspect.signature(wrapper.fn).parameters.values():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                yield NotPositional(fn, p.name, p.kind.description)
            elif p.default is not p.empty:
                yield HeldByValue(fn, p.name)
    if len(wrapper.function_reads) != len(names := parameters_of(wrapper.fn)):
        yield ReadsUnlike(fn, tuple(str(r.place) for r in wrapper.function_reads), names)


def check_outputs(wrapper: 'Wrapper') -> Iterator[Violation]:
    '''At least one place written: a node owning nothing is unreachable.'''
    if not wrapper.outputs:
        yield NoOutputs(name_of(wrapper.fn), type(wrapper).__name__)


# ============================================================================ what a vocabulary's kinds extend

class Declaration[N: Label, V: Place](Frozen):
    '''What a model declares: the nodes it binds at a name, and the values and tags written
    for its places.'''

    outputs : AbstractVar[tuple[V, ...]]
    '''The places it writes, as written: a residual's or a problem's unknowns.'''
    given   : AbstractVar[OrderedMap[V, Any]]
    '''Each value written, by the place minted for it.'''
    tags    : AbstractVar[Tags[V]]
    '''The tags written on each place it reads or writes.'''
    holds   : AbstractVar[tuple[Any, ...]]
    '''What it takes over: each keeps only `nodes_when_held` at its own name.'''

    @abstractmethod
    def nodes(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        '''What this declares, bound at `at` and at names derived from it.'''

    @abstractmethod
    def nodes_when_held(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        '''What stays at `at` when another declaration holds this one.'''

    @abstractmethod
    def loose_at(self, names: 'DerivedNames[N, V]') -> N | None:
        '''The node name it takes handed over loose, outside a slot, `names` its place's;
        `None` for none: then it goes in a slot.'''


class Wrapper[N: Label, V: Place](Declaration[N, V]):
    '''
    A function, the place each parameter reads, by position, and the places it writes. It
    carries no name: where it is put names it (`nodes(at)`), and a name it derives it asks
    the place (`DerivedNames`). A final is declared by a vocabulary method `declares` makes.
    '''

    fn               : Body
    '''The function, or a description of one (`Unimplemented`, a reference to code elsewhere).'''
    function_reads   : tuple[PlaceWithValue[V], ...]
    '''The place each parameter of `fn` reads, in signature order, with the value where one
    was written.'''
    outputs          : tuple[V, ...]
    output_tags      : Tags[V] = dataclasses.field(default=OrderedMap(), kw_only=True,
                                                   compare=False)
    '''The tags written on its outputs.'''
    function_outputs : AbstractVar[tuple[V, ...]]
    '''The places its return values are written to, in order: a residual's are places of
    their own, not the unknowns'.'''
    terms            : AbstractVar[tuple[PlaceWithValue[V] | None, ...]]
    '''What else it writes where a place is due, each a place (with the value where one was
    written) or `None` for zero: a constraint's bounds or targets; none for most.'''

    checks = (check_outputs, check_signature)

    holds = ()

    @property
    def inputs(self) -> tuple[V, ...]:
        '''The place each parameter of `fn` reads, in signature order: `outputs`' mirror.'''
        return tuple(r.place for r in self.function_reads)

    @property
    def given(self) -> OrderedMap[V, Any]:
        return given_in(self.function_reads, self.terms)

    @property
    def tags(self) -> Tags[V]:
        return merged_tags((*((r.place, r.tags) for r in (*self.function_reads, *self.terms)
                              if r is not None),
                            *self.output_tags.items()))

    def nodes_when_held(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        '''Its function: what holds it takes its condition.'''
        return ((at, self.function_node()),)

    def loose_at(self, names: 'DerivedNames[N, V]') -> N | None:
        '''Its function's name, as a slot of that name gives it; none for a lambda or a
        callable value.'''
        name = getattr(self.fn, '__name__', None)
        return None if name in (None, '<lambda>') else names.node_in_tree((Key(name),))

    def function_node(self) -> Function[V] | ImplementedFunction[V]:
        '''The node running `fn`: reading `function_reads`, owning `function_outputs`; with no
        implementation where `fn` is `Unimplemented`.'''
        node = Function(self.inputs, self.function_outputs)
        return node if isinstance(self.fn, Unimplemented) else ImplementedFunction(node, self.fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        '''What `inspect.signature` and `help` follow.'''
        return self.fn


class Explicit[N: Label, V: Place](Wrapper[N, V]):
    '''A function computing its outputs, returned in that order.'''

    terms = ()

    @property
    def function_outputs(self) -> tuple[V, ...]:
        return self.outputs

    def nodes(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        return ((at, self.function_node()),)


class Solve(Mint):
    '''Where the problem beside a model binds, or a problem named after its objective:
    `^solve.at`.'''


class Paired[N: Label, V: Place](Wrapper[N, V]):
    '''A function, and beside it a condition node over what it outputs, at the function's name
    under `condition_mint` (`^solve.at`, `^require.at`): an implicit model, a constraint,
    an ODE.'''

    condition_mint : AbstractClassVar[type[Mint]]
    '''The purpose the condition node's name is derived for: `Solve`, `Require`.'''
    condition      : AbstractVar[ConditionalNode[V]]
    '''The condition node over what the function outputs.'''

    @abstractmethod
    def condition_node(self, at: N) -> ConditionalNode[V]:
        '''The condition node as bound at `at`, its own name: `condition`, or one whose
        places derive from its name.'''

    def condition_at(self, at: N) -> N:
        '''The wrapper's name -> the condition node's.'''
        return minted(self.condition_mint(), at)

    def nodes(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        beside = self.condition_at(at)
        return ((at, self.function_node()), (beside, self.condition_node(beside)))

    def loose_condition(self) -> N:
        '''Where the condition node binds when the wrapper is handed to `declare` loose: beside
        its function's name. In a slot the slot names it, and this name is bound nowhere.'''
        from ..surface.interface import DerivedNames               # the interface is built on this
        at = self.loose_at(answer(DerivedNames, self.outputs[0]))
        if at is None:
            raise Unnamed(repr(self)).raised
        return self.condition_at(at)


# ============================================================================ a kind as a vocabulary method

class Keyword(Frozen):
    '''A wrapper field read from a keyword argument of its own: `read(surface, fn, outputs,
    written)`, `outputs` the places written and `written` the argument, `None` if not given.'''

    keyword : str
    read    : Callable[..., Any]


class Declares[W: Wrapper]:
    '''
    A declaring part's method declaring a `kind`, `method(fn, *outputs)` or `@method(*outputs)`,
    built from `built_from(surface, fn, outputs, reads, read_in)` and each field named in
    `keywords` read from its `Keyword`. Named where it is bound; `doc` is its docstring.
    '''

    def __init__(self, kind: type[W], doc: str, keywords: Mapping[str, Keyword]) -> None:
        if stray := [f for f, k in keywords.items() if not isinstance(k, Keyword)]:
            raise TypeError(f'declares({kind.__name__}, ...): {stray!r} -- a field is read '
                            f'from a `Keyword`')
        self.__declares__, self.__doc__ = kind, doc
        self.__implementation__ = self                  # a name a level implements, for `Frozen`
        self.__name__ = self.__qualname__ = '<declared>'
        self.by_keyword = {k.keyword: (field, k.read) for field, k in keywords.items()}

    def __set_name__(self, owner: type, name: str) -> None:
        self.__name__, self.__qualname__ = name, f'{owner.__qualname__}.{name}'
        self.__module__ = owner.__module__

    @overload
    def __get__(self, surface: None, owner: type) -> 'Declares[W]': ...
    @overload
    def __get__(self, surface: 'Declaring', owner: type) -> 'BoundDeclares[W]': ...
    def __get__(self, surface: 'Declaring | None', owner: type) -> 'Declares[W] | BoundDeclares[W]':
        return self if surface is None else BoundDeclares(self, surface)


class BoundDeclares[W: Wrapper]:
    '''A `Declares` on its surface: the call form returns the kind, the decorator form a
    decorator returning it.'''

    def __init__(self, declares: Declares[W], surface: 'Declaring') -> None:
        self.declares, self.surface = declares, surface
        self.__doc__, self.__name__ = declares.__doc__, declares.__name__
        self.__qualname__, self.__module__ = declares.__qualname__, declares.__module__

    @overload
    def __call__(self, fn: Body, /, *outputs: 'Named[Place] | Tagged | Any', reads: Any = ...,
                 read_in: Any = ..., **keywords: Any) -> W: ...
    @overload
    def __call__(self, *outputs: 'Named[Place] | Tagged | Any', reads: Any = ..., read_in: Any = ...,
                 **keywords: Any) -> Callable[[Callable[..., Any]], W]: ...
    def __call__(self, *args: Any, reads: Any = OrderedMap(), read_in: Any = OrderedMap(),
                 **written: Any) -> Any:
        by_keyword, surface = self.declares.by_keyword, self.surface
        if stray := [k for k in written if k not in by_keyword]:
            raise TypeError(f'{self.__qualname__}() got an unexpected keyword argument '
                            f'{stray[0]!r}')
        def build(fn: Body, outputs: tuple) -> W:
            fn, function_reads, places, tags = built_from(surface, fn, outputs, reads, read_in)
            own = {field: read(surface, fn, places, written.get(k))
                   for k, (field, read) in by_keyword.items()}
            return self.declares.__declares__(fn, function_reads, places, output_tags=tags, **own)
        return now_or_as_decorator(args, build)


def built_from[V: Place](surface: 'Declaring', fn: Body, outputs: tuple, reads: Any = OrderedMap(),
                         read_in: Mapping[Surface, str | tuple[str, ...]] = OrderedMap()
                         ) -> tuple[Body, tuple[PlaceWithValue[V], ...], tuple[V, ...], Tags[V]]:
    '''What a wrapper is built from: `fn`, a place per parameter (the surface's
    `parameter_reads`, each `read_in` surface's `place` its rule, a value minted beside the
    first output), the outputs as places, and the tags written on them.'''
    bare, tags = zip(*map(untagged, outputs)) if outputs else ((), ())
    places = surface.places(bare, role='an output')
    rules = {}
    for other, names in read_in.items():
        if not isinstance(other, Surface):
            raise TypeError(f'read_in={{{other!r}: ...}}: a surface, and the names read '
                            f'by its rule')
        rules[other.place] = names
    beside = surface.derived.beside
    function_reads = surface.parameter_reads(fn, reads, rules,
                                             given=lambda p: minted(Given(), beside(places[0], p)))
    return fn, function_reads, places, OrderedMap((p, t) for p, t in zip(places, tags) if t)


def declares[W: Wrapper](kind: type[W], doc: str, **keywords: Keyword) -> Declares[W]:
    '''A surface's method declaring `kind`: `residual = declares(Residual, doc)`.'''
    return Declares(kind, doc, keywords)


def now_or_as_decorator(args: tuple, build: Callable[[Body, tuple], Any]) -> Any:
    '''`build(fn, outputs)` now, where a function (or its description) comes first:
    `residual(fn, 'u')`. Else `args` are the outputs, and what is returned is a decorator,
    at a `def` only, that builds from it: `@residual('u')`. The wrapper takes the function's
    name, so the function is re-pointed to be found through it (`disc2.fn`): a graph holding
    the function still pickles.'''
    if args and (callable(args[0]) or isinstance(args[0], Description)):
        return build(args[0], args[1:])
    def decorate(fn: Callable[..., Any]) -> Any:
        held = sys.modules.get(getattr(fn, '__module__', None) or '')
        for part in getattr(fn, '__qualname__', '').split('.'):
            held = getattr(held, part, None)
        if isinstance(fn, Wrapper) or held is fn:
            raise TypeError(f'{name_of(fn)} is wrapped or defined elsewhere, so decorating it '
                            f'would rename it: wrap it where it is used, the function first '
                            f'(`function(fn, ...)`)')
        wrapper = build(fn, args)
        fn.__qualname__ = f'{fn.__qualname__}.fn'
        return wrapper
    return decorate


__all__ = [
    'Side', 'Tagged', 'untagged', 'PlaceWithValue', 'given_in', 'Tags', 'merged_tags',
    'Description', 'Unimplemented', 'Body', 'parameters_of', 'check_outputs', 'check_signature',
    'Declaration', 'Wrapper', 'Explicit', 'Solve', 'Paired', 'Keyword', 'built_from', 'declares', 'Declares',
    'BoundDeclares', 'now_or_as_decorator',
]
