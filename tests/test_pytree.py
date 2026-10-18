'''
Indexing a pytree by key path: `read_at` / `write_at` / `complete_at` and the checks.
'''


import equinox as eqx
import pytest
from jax.tree_util import (
    DictKey, GetAttrKey, SequenceKey, register_pytree_with_keys,
)

from cottax.pytree.crossings import check_antichain, complete_at, read_at, write_at
import purposes


def v(*keys) -> tuple:
    '''`pytree.py` deals in bare `KeyPath`s -- a tuple of key entries, nothing more.'''
    return keys


class Eq(eqx.Module):
    psi : object
    q   : object


class Labelled(eqx.Module):
    a     : object
    label : str = eqx.field(static=True)


@pytest.fixture
def cds():
    return {'eq': [Eq(psi=None, q=1.0)], 'k': 2.0, 'x': None}


X   = v(DictKey('x'))
K   = v(DictKey('k'))
EQ0 = v(DictKey('eq'), SequenceKey(0))
PSI = v(DictKey('eq'), SequenceKey(0), GetAttrKey('psi'))
Q   = v(DictKey('eq'), SequenceKey(0), GetAttrKey('q'))


# ---------------------------------------------------------------- reading

def test_a_name_standing_for_a_subtree_is_answered_whole(cds):
    # A name is a place, so `.eq[0]` is one place holding a module rather than the two
    # leaves inside it.
    assert read_at(cds, EQ0) is cds['eq'][0]
    assert read_at(cds, Q) == 1.0


def test_an_unset_place_is_still_a_place(cds):
    # `None` is an empty pytree node, so it is no leaf of a flatten -- but the walk goes
    # by name, which is what keeps a declared but uncomputed place addressable.
    assert read_at(cds, X) is None
    assert read_at(cds, PSI) is None


def test_a_name_with_no_place_raises_whatever_the_step_raised(cds):
    # `read_at` does not catch: an error here is a broken precondition, and saying which
    # key stopped is `check_pytree_readable`'s job -- see below.
    with pytest.raises(KeyError):
        read_at(cds, v(DictKey('typo')))

    with pytest.raises(AttributeError):
        read_at(cds, v(DictKey('k'), GetAttrKey('nope')))


def test_reading_walks_and_so_does_not_ask_the_precondition():
    # Reading is a walk per name, so neither of the write's refusals is raised here: two
    # nested names are both answered (above), and so is a place no flatten leads to. Both
    # are `write_at`'s to refuse -- see below.
    tree  = {'m': Labelled(a=1.0, label='hello')}
    assert read_at(tree, v(DictKey('m'), GetAttrKey('label'))) == 'hello'


def test_the_walk_trusts_the_key_kind():
    # `[0]` spells both kinds, and a trusting step takes the container at its word: the
    # dict answers a `SequenceKey`. That refusal is `write_at`'s -- below.
    tree = {0: 'by dict key'}
    assert read_at(tree, v(DictKey(0)))     == 'by dict key'
    assert read_at(tree, v(SequenceKey(0))) == 'by dict key'


def test_a_key_kind_nobody_registered_a_step_for_still_resolves():
    # jax's key vocabulary is open -- cottax's own `Mint` is one of them -- so the
    # dispatch falls back to matching by equality against what the container actually
    # flattens to. No step registered (`key_kind(cls, step)`) is needed, only to make it O(1).
    class Shadow:
        def __init__(self, v): self.v = v

    register_pytree_with_keys(
        Shadow, lambda s: ([(purposes.Copy(), s.v)], None), lambda _, vs: Shadow(*vs),
    )

    assert read_at({'s': Shadow(3.0)}, v(DictKey('s'), purposes.Copy())) == 3.0
    with pytest.raises(KeyError):
        read_at({'a': 1.0}, v(purposes.Copy()))


# ---------------------------------------------------------------- writing

def test_writing_leaves_the_original_alone(cds):
    before = {'eq': [Eq(psi=None, q=1.0)], 'k': 2.0, 'x': None}
    out    = write_at(cds, {X: 9.0, PSI: 5.0})

    assert out == {'eq': [Eq(psi=5.0, q=1.0)], 'k': 2.0, 'x': 9.0}
    assert cds == before


def test_a_subtree_is_written_where_it_is_named(cds):
    assert write_at(cds, {EQ0: Eq(psi=7.0, q=8.0)}) == \
           {'eq': [Eq(psi=7.0, q=8.0)], 'k': 2.0, 'x': None}


def test_writing_nothing_is_not_a_traversal(cds):
    assert write_at(cds, {}) is cds


def test_writing_refuses_what_it_cannot_place(cds):
    # A write that missed would be silent, so every refusal is asked of the one flatten
    # the write makes anyway.
    with pytest.raises(ValueError, match='lies inside'):
        write_at(cds, {EQ0: 1.0, PSI: 2.0})
    with pytest.raises(KeyError, match=r"\['typo'\] does not resolve"):
        write_at(cds, {v(DictKey('typo')): 1.0})
    with pytest.raises(KeyError, match=r"\['k'\].nope does not resolve"):
        write_at(cds, {v(DictKey('k'), GetAttrKey('nope')): 1.0})
    with pytest.raises(KeyError, match='not a place this pytree flattens to'):
        write_at({'m': Labelled(a=1.0, label='hello')},
                 {v(DictKey('m'), GetAttrKey('label')): 'bye'})

    # And the kind the trusting walk answers: only a flatten knows a dict yields
    # `DictKey`, so this is where `[0]` stops being one spelling for two names.
    with pytest.raises(KeyError, match='not a place this pytree flattens to'):
        write_at({0: 'by dict key'}, {v(SequenceKey(0)): 'x'})


# ---------------------------------------------------------------- the antichain

def test_the_antichain_is_a_question_about_the_names_alone():
    # Containment, not equality: `.eq[0]` already holds `.eq[0].psi`'s storage, so both
    # named means one value named twice. No tree is consulted -- it is a write's
    # precondition, and reading two nested names is only reading one value twice.
    assert check_antichain([K, X, PSI]) is None
    with pytest.raises(ValueError, match='lies inside'):
        check_antichain([EQ0, PSI])

    # Which is why it reaches what a tree cannot: a minted name has no place in any
    # CDS to be checked against, but two of them still nest.
    hat = v(purposes.Copy(), DictKey('u'))
    with pytest.raises(ValueError, match='lies inside'):
        check_antichain([v(purposes.Copy()), hat])


# ---------------------------------------------------------------- completing

def test_completing_adds_what_a_dict_lacks_and_changes_nothing_else(cds):
    new = v(DictKey('new'))
    deep = v(DictKey('a'), DictKey('b'), DictKey('c'))
    done = complete_at(cds, [X, PSI, new, deep])
    assert done == {**cds, 'new': None, 'a': {'b': {'c': None}}}
    assert done['eq'] is cds['eq'] and 'new' not in cds        # what it had is shared, untouched


def test_completing_reaches_a_dict_inside_another_container():
    tree = {'eq': [Eq(psi={}, q=1.0)]}
    done = complete_at(tree, [v(DictKey('eq'), SequenceKey(0), GetAttrKey('psi'), DictKey('r'))])
    assert done['eq'][0].psi == {'r': None} and tree['eq'][0].psi == {}


def test_a_name_and_one_inside_it_complete_as_the_inner_one():
    # whatever order handed: `.s` is made a dict holding `.s.u`, not a `None` it stops at
    for names in ([v(DictKey('s')), v(DictKey('s'), DictKey('u'))],
                  [v(DictKey('s'), DictKey('u')), v(DictKey('s'))]):
        assert complete_at({}, names) == {'s': {'u': None}}


def test_only_a_dict_grows_a_key(cds):
    for name in (v(DictKey('eq'), SequenceKey(0), GetAttrKey('nope')),   # a field
                 v(DictKey('eq'), SequenceKey(3)),                        # an index
                 v(DictKey('x'), DictKey('under'))):                      # under a `None`
        with pytest.raises(KeyError, match='only a dict grows a key'):
            complete_at(cds, [name])
