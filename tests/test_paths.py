'''
`pytree.path` alone: `Path` and the two namespaces under it, and the node.

Everything here is written the long way -- key entries by hand, no converters -- because
that is the whole claim of the module: it defines, and nothing in it guesses what a
caller meant.
'''


import jax
import pytest
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

from cottax.core import Function
from cottax.core.realisation import ImplementedFunction
from cottax.pytree.path import NodePath, Path, VarPath
import cottax.assemblies.jax as cx


def V(*keys) -> VarPath:
    return VarPath(tuple(keys))


def N(*keys) -> NodePath:
    return NodePath(tuple(keys))


# ---------------------------------------------------------------- the type

def test_a_path_is_built_from_jaxs_key_vocabulary():
    # The full vocabulary, spelled out. Entering a data structure is part of naming a
    # place inside it, so which *kind* of step it is belongs in the name.
    p = V(DictKey('eq'), SequenceKey(0), GetAttrKey('psi'), FlattenedIndexKey(2))
    assert p.segments == (DictKey('eq'), SequenceKey(0), GetAttrKey('psi'),
                      FlattenedIndexKey(2))


def test_the_base_is_abstract():
    # A place is always a place of *something*; `Path` itself names no namespace.
    with pytest.raises(TypeError, match='abstract'):
        Path((GetAttrKey('u'),))


def test_the_two_namespaces_do_not_meet():
    # Same keys, different worlds. This is the point of two finals rather than one
    # class: a variable named `d1` and a node named `d1` are not the same name.
    assert V(GetAttrKey('d1')) != N(GetAttrKey('d1'))
    assert V(GetAttrKey('d1')) == V(GetAttrKey('d1'))


def test_a_path_is_its_keys():
    assert V(GetAttrKey('a'), GetAttrKey('b')) == V(GetAttrKey('a'), GetAttrKey('b'))
    assert V(GetAttrKey('a')) != V(DictKey('a'))       # attribute is not item access
    assert {V(GetAttrKey('u')): 1}[V(GetAttrKey('u'))] == 1
    # ... and not its spelling: a name is a place, `'a.b'` is text.
    assert V(GetAttrKey('a'), GetAttrKey('b')) != 'a.b'
    assert V(GetAttrKey('u')) != 'u'


def test_a_builder_takes_any_field_name():
    '''A `Path` answers only `segments` and `spelling`, so a field called `parent` or
    `leaf` is written as any other.'''
    assert cx.absolute.place(cx.V.tree.parent.leaf) == V(GetAttrKey('tree'), GetAttrKey('parent'),
                                                 GetAttrKey('leaf'))


# ---------------------------------------------------------------- showing

def test_a_path_is_written_the_way_it_is_read():
    # Asked for by name: a name is written by `spelling`, never by printing the path,
    # which says which namespace it is in.
    assert V(GetAttrKey('mesh'), GetAttrKey('nodes')).spelling == '.mesh.nodes'
    assert V(DictKey('eq'), SequenceKey(0), GetAttrKey('psi')).spelling == "['eq'][0].psi"

    assert V(GetAttrKey('u')).spelling == '.u'
    assert V(DictKey('u')).spelling == "['u']"


def test_mints_stack_and_unminted_drops_one():
    from cottax.core import mint_of, unminted
    import purposes

    start, hat, y = purposes.Start(), purposes.Copy(), V(DictKey('y'))
    assert start(hat(y)) != start(y)
    assert unminted(start(hat(y))) == hat(y) and mint_of(unminted(start(hat(y)))) == purposes.Copy()
    assert start(hat(y)).spelling == "^start^copy['y']"


def test_a_repr_says_which_namespace():
    assert repr(V(GetAttrKey('mesh'), GetAttrKey('nodes'))) == 'VarPath(.mesh.nodes)'
    assert repr(N(GetAttrKey('d1'))) == 'NodePath(.d1)'


# ---------------------------------------------------------------- as data

def test_a_path_is_an_opaque_pytree_leaf():
    # The whole reason it is a frozen dataclass and not an `eqx.Module`: a name must
    # reach `eqn.params` as a jit cache key, not be structure jax walks into.
    p = V(GetAttrKey('a'), GetAttrKey('b'))
    assert jax.tree_util.tree_leaves(p) == [p]


def test_a_path_carries_no_instance_dict():
    with pytest.raises(AttributeError):
        V(GetAttrKey('u')).__dict__


def test_paths_are_unordered():
    # Key entries are unordered and there is no order over kinds to invent -- and an
    # invented one would rest on how a key entry happens to `repr`. Anything wanting one
    # sorts by the spelling and says so; anything wanting a keyed container uses an
    # `OrderedDict`, which jax flattens without sorting.
    with pytest.raises(TypeError, match='not supported between'):
        sorted([V(GetAttrKey('b')), V(GetAttrKey('a'))])


def test_a_spelling_labels_but_does_not_identify():
    # `[0]` writes both of these, which is why nothing keys or orders by the spelling.
    assert str(V(DictKey(0))) == str(V(SequenceKey(0)))
    assert V(DictKey(0)) != V(SequenceKey(0))


# ---------------------------------------------------------------- nodes

def node(name, reads, owns, **kw):
    # `name` is what a caller would bind this under; the node itself never carries it.
    return ImplementedFunction(
        Function(reads=tuple(V(GetAttrKey(r)) for r in reads),
                 owns=tuple(V(GetAttrKey(o)) for o in owns)),
        kw.pop('fn', lambda *a: None),
    )


def test_a_node_is_its_interface():
    # And nothing else: a name is a place, so nothing knows its own. `NestingGraph` binds it.
    n = node('d1', ['z', 'y2'], ['y1'])
    assert not hasattr(n, 'name')
    assert n.reads == (V(GetAttrKey('z')), V(GetAttrKey('y2')))
    assert n.owns == (V(GetAttrKey('y1')),)


def test_a_node_mints_variables():
    # A node that owns nothing is unreachable, and invisible to `ancestors`.
    with pytest.raises(ValueError, match='must own at least one variable'):
        node('nothing', ['z'], [])


def test_only_writes_must_be_unique():
    # `fn(x, x)` is legitimate; two outputs of one name are not.
    assert node('twice', ['x', 'x'], ['y']).reads == (V(GetAttrKey('x')),) * 2
    with pytest.raises(ValueError, match='duplicate output name'):
        node('dupe', [], ['y', 'y'])


def test_every_subclass_is_slotted():
    # A slotted base buys nothing if a subclass forgets, and forgetting is silent: it
    # carries a `__dict__` back with every test passing.
    def kinds(c):
        yield c
        for sub in c.__subclasses__():
            yield from kinds(sub)

    assert not [c.__name__ for c in kinds(Path)
                if c.__module__.startswith('cottax') and '__slots__' not in c.__dict__]


def test_an_error_shows_names_as_they_are_written():
    with pytest.raises(ValueError, match=r'\.s\.x'):
        Function(
            reads=(V(GetAttrKey('s'), GetAttrKey('x')),),
            owns=(V(GetAttrKey('s'), GetAttrKey('x')),),
        )
