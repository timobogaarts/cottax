'''
`AbstractVar`: a level's demand on its finals, satisfied any way a name can be bound and
refused by ABC when it is not.
'''

import copy
import dataclasses
import inspect
import pickle
from typing import Any

from abc import abstractmethod
import pytest

from cottax.core import Violation, violations
from cottax.core.driven import DriverContract
from typing import ClassVar

from cottax.core.frozen import AbstractClassVar, AbstractVar, Frozen, OrderedMap, PerLevel, checked


class Level(Frozen):
    fn : AbstractVar[Any]


class AsField(Level):      fn : Any
class AsDefaulted(Level):  fn : Any = None
class AsAttribute(Level):  fn = staticmethod(lambda: 1)
class AsProperty(Level):
    @property
    def fn(self): return 2
class Forgot(Level):       other : Any = None
class Between(Level):      pass
class Below(Between):      fn : Any
class Inherits(AsField):   pass


def test_every_way_to_bind_is_concrete():
    assert AsField(fn=1).fn == 1 and AsDefaulted().fn is None
    assert AsAttribute().fn() == 1 and AsProperty().fn == 2
    assert Below(3).fn == 3 and Inherits(4).fn == 4
    assert not any(map(inspect.isabstract, (AsField, AsDefaulted, AsAttribute, AsProperty, Below)))


def test_the_level_gets_no_field_and_a_field_gets_no_default():
    assert 'fn' not in Level.__dataclass_fields__
    assert str(inspect.signature(AsField.__init__)) == '(self, fn: Any) -> None'
    assert not hasattr(AsField, 'fn')                  # per instance, honestly
    assert AsField(1) == AsField(1) and hash(AsField(1)) == hash(AsField(1))
    assert 'fn' in inspect.get_annotations(Level)      # restored for a reader


@pytest.mark.parametrize('cls', [Level, Forgot, Between])
def test_unbound_is_abstract_and_abc_refuses(cls):
    assert inspect.isabstract(cls)
    with pytest.raises(TypeError, match="abstract class .* 'fn'"):
        cls()


def test_a_level_may_not_give_its_abstract_var_a_value():
    with pytest.raises(TypeError, match='cannot have a value'):
        class Bad(Level):
            fn : AbstractVar[Any] = 1


def test_a_stringified_annotation_is_refused():
    with pytest.raises(NotImplementedError, match='stringified'):
        class Bad(Level):
            g : 'AbstractVar[Any]'


class PerClass(Frozen):
    table : AbstractClassVar[tuple]


def test_an_abstract_class_var_is_bound_per_class():
    class Attr(PerClass):     table = (1,)
    class Typed(PerClass):    table : ClassVar[tuple] = (2,)
    assert Attr().table == (1,) and Typed().table == (2,)
    assert 'table' not in {f.name for f in dataclasses.fields(Typed)}     # a ClassVar, not a field
    with pytest.raises(TypeError, match="abstract class .* 'table'"):
        PerClass()


@pytest.mark.parametrize('binding', ['table : tuple', 'table : tuple = ()'])
def test_an_abstract_class_var_may_not_be_a_field(binding):
    with pytest.raises(TypeError, match='per class'):
        exec(f'class Bad(PerClass):\n    {binding}', {'PerClass': PerClass})


# ---------------------------------------------------------------- checks, root-down

def _refuse(why):
    raise ValueError(why)

class Root(Frozen):
    n : int = 0
    checks = (lambda v: () if v.n >= 0 else _refuse('negative'),)
class Mid(Root):
    checks = (lambda v: () if v.n < 10 else _refuse('too big'),)
class Leaf(Mid):
    pass


def test_checks_accumulate_root_down_and_none_can_be_dropped():
    assert Leaf(3).n == 3
    with pytest.raises(ValueError, match='negative'):
        Leaf(-1)
    with pytest.raises(ValueError, match='too big'):
        Leaf(11)


def test_a_rule_in_checks_raises_its_first_violation_and_the_collector_yields_them():
    class Found(Violation):
        n : int
        error    = ValueError
        template = 'found {n}'
    class Ruled(Frozen):
        n : int
        checks = (lambda v: (Found(v.n) for _ in range(1) if v.n),)
    assert Ruled(0).n == 0
    with pytest.raises(ValueError, match='found 3'):
        Ruled(3)
    assert list(violations(Ruled, 0)) == []
    assert list(violations(Ruled, 3)) == [Found(3)]
    with pytest.raises(ValueError, match='negative'):     # a raising check still raises
        list(violations(Leaf, -1))


def checked_halving():
    '''A checked function, the record of its body running, and its two refusals: built per
    test, so no module-level `Violation` final escapes into the corpus `test_core` checks.'''
    class Low(Violation):
        n : int
        error    = ValueError
        template = 'low {n}'
    class Odd(Violation):
        n : int
        error    = ValueError
        template = 'odd {n}'
    def low(n, floor, *, step):
        if n < floor:
            yield Low(n)
    def odd(n, floor, *, step):
        if (n // step) % 2:
            yield Odd(n)
    ran = []
    @checked(low, odd)
    def halved(n, floor=0, *, step=1):
        ran.append(n)
        return n // 2
    return halved, ran, Low, Odd


def test_a_checked_function_raises_its_first_refusal_before_the_body_runs():
    halved, ran, _, _ = checked_halving()
    with pytest.raises(ValueError, match='low -1'):
        halved(-1)
    assert ran == []


def test_violations_of_a_checked_function_are_every_refusal_and_the_body_never_runs():
    halved, ran, Low, Odd = checked_halving()
    assert list(violations(halved, -1)) == [Low(-1), Odd(-1)]
    assert ran == []


def test_defaults_and_keywords_reach_the_checks():
    halved, _, Low, Odd = checked_halving()
    assert list(violations(halved, 3)) == [Odd(3)]                # floor=0, step=1 defaulted
    assert list(violations(halved, 3, 5, step=4)) == [Low(3)]
    assert list(violations(halved, n=6, floor=0, step=3)) == []


def test_a_checked_function_with_no_refusal_returns_its_result():
    halved, ran, _, _ = checked_halving()
    assert halved(4) == 2 and ran == [4] and halved.__name__ == 'halved'


def test_a_subclass_may_not_define_post_init():
    with pytest.raises(TypeError, match='bypass'):
        class Bad(Root):
            def __post_init__(self): pass


# ---------------------------------------------------------------- the tree's own demands

def test_a_driver_without_accepts_cannot_be_built():
    class Vague(DriverContract): ...
    with pytest.raises(TypeError, match="'accepts'"):
        Vague()


def test_violations_are_the_first_refusing_level_s_all_and_the_levels_below_unasked():
    '''A level is asked once every level above it holds, as a rule group is: what it
    refuses is raised as its first violation, carrying all of the level's.'''
    class Bad(Violation):
        why : str
        error    = ValueError
        template = '{why}'
    asked = []

    class Base(Frozen):
        n : int
        checks = (lambda v: [Bad('negative')] if v.n < 0 else [],
                  lambda v: [Bad('odd')] if v.n % 2 else [])
    class Below(Base):
        checks = (lambda v: asked.append(v.n) or ([Bad('big')] if abs(v.n) > 10 else []),)

    assert list(violations(Below, -11)) == [Bad('negative'), Bad('odd')] and asked == []
    try:                       # not `pytest.raises`: its record would keep `Bad` alive for test_core's corpus
        Below(-11)
    except ValueError as refused:
        assert str(refused) == 'negative' and refused.violation == Bad('negative')
        assert refused.violations == (Bad('negative'), Bad('odd'))
    else:
        pytest.fail('Below(-11) was built')
    assert list(violations(Below, 12)) == [Bad('big')] and asked == [12]


def test_a_refusal_of_another_value_built_inside_a_check_is_not_taken_for_its_own():
    class Bad(Violation):
        why : str
        error    = ValueError
        template = '{why}'
    class Inner(Frozen):
        n : int
        checks = (lambda v: [Bad('inner')] if v.n else [],)
    class Outer(Frozen):
        n : int
        checks = (lambda v: Inner(v.n) and [],)
    assert list(violations(Outer, 0)) == []
    with pytest.raises(ValueError, match='inner'):
        list(violations(Outer, 1))


# ---------------------------------------------------------------- OrderedMap

def test_an_ordered_map_is_equal_only_in_the_same_order():
    '''Order is observable (a graph's binding order breaks its ties), so it is in the
    equality and the hash; and a map never equals the plain dict of its items.'''
    a, b = OrderedMap({'x': 1, 'y': 2}), OrderedMap({'y': 2, 'x': 1})
    assert a != b and a == OrderedMap([('x', 1), ('y', 2)])
    assert hash(a) == hash(OrderedMap({'x': 1, 'y': 2}))
    assert a != {'x': 1, 'y': 2}
    assert list(a) == ['x', 'y'] and a['y'] == 2


def test_an_ordered_map_refuses_a_repeated_key():
    with pytest.raises(ValueError, match='duplicate'):
        OrderedMap([('x', 1), ('x', 2)])


@pytest.mark.parametrize('change', [
    lambda d: setattr(d, '_keys', ()), lambda d: d.__setitem__('x', 0),
    lambda d: d.update(z=0),
])
def test_an_ordered_map_cannot_change(change):
    d = OrderedMap({'x': 1})
    with pytest.raises((AttributeError, TypeError)):
        change(d)
    assert d == OrderedMap({'x': 1})


def test_an_ordered_map_stays_one_through_every_copy():
    '''Pickled, copied, joined: an `OrderedMap` out. `|` keeps a key's position.'''
    d = OrderedMap({'x': (1, 2), 'y': 0})
    for out in (pickle.loads(pickle.dumps(d)), copy.copy(d), copy.deepcopy(d)):
        assert type(out) is OrderedMap and out == d
    joined = d | {'x': 3, 'z': 4}
    assert type(joined) is OrderedMap and joined.items() == (('x', 3), ('y', 0), ('z', 4))


def test_an_ordered_map_hashes_when_asked_and_not_before():
    '''An unhashable value is built and refused on use, as a tuple holding a list is.'''
    d = OrderedMap({'x': []})
    with pytest.raises(TypeError, match='unhashable'):
        hash(d)


# ---------------------------------------------------------------- abstract or final, for methods

class Abstract(Frozen):
    @abstractmethod
    def f(self) -> int: ...

class Final(Abstract):
    def f(self) -> int: return 1


def test_a_final_implements_what_its_level_left_abstract():
    assert Final().f() == 1


def test_a_method_a_level_implemented_is_not_implemented_again():
    with pytest.raises(TypeError, match='overrides the one Final implements'):
        class Again(Final):
            def f(self) -> int: return 2


def test_nor_is_it_shadowed_by_a_property():
    with pytest.raises(TypeError, match='overrides the one Final implements'):
        class Shadow(Final):
            @property
            def f(self) -> int: return 2


def test_two_levels_beside_each_other_may_not_both_implement_a_name():
    class Left(Abstract):
        def f(self) -> int: return 1
    class Right(Abstract):
        def f(self) -> int: return 2
    with pytest.raises(TypeError, match='joins Left and Right, which both implement `f`'):
        class Both(Left, Right):
            pass


def test_a_join_of_one_implementation_and_an_abstract_level_is_final():
    class Other(Abstract):
        def g(self) -> int: return 2
    class Both(Final, Other):
        pass
    assert (Both().f(), Both().g()) == (1, 2)


def test_a_dunder_and_a_value_are_not_methods_the_rule_reads():
    class Repr(Final):
        checks = ()
        def __repr__(self) -> str: return 'final'
    assert repr(Repr()) == 'final'


def test_a_per_level_name_accumulates_and_is_no_field():
    class Level(Frozen):
        steps : PerLevel[tuple] = ('a',)
    class Final(Level):
        steps = ('b',)
    assert [f.name for f in dataclasses.fields(Final)] == []
    assert list(Final.per_level('steps')) == ['a', 'b']
    assert list(Final.per_level('steps', derived_first=True)) == ['b', 'a']


def test_only_a_name_declared_per_level_is_accumulated():
    class Level(Frozen):
        steps : ClassVar[tuple] = ('a',)
    with pytest.raises(TypeError, match='not declared `PerLevel`'):
        list(Level.per_level('steps'))
    with pytest.raises(TypeError, match='not declared `PerLevel`'):
        list(Frozen.per_level('chekcs'))



# ---------------------------------------------------------------- Answer and Choice

from cottax.core.frozen import Answer, Choice, answers


class Asked(Answer):
    about = object


class Also(Answer):
    about = object


class About:
    pass


def test_a_class_both_an_answer_and_a_choice_is_refused_at_creation():
    with pytest.raises(TypeError) as found:
        class Both(Asked, Choice):
            pass
    where = test_a_class_both_an_answer_and_a_choice_is_refused_at_creation.__qualname__
    assert str(found.value) == (f'{__name__}.{where}.<locals>.Both is both an `Answer` and a '
                                f'`Choice`: one right answer is registered, several are named by '
                                f'a study, and nothing is both')


def test_an_answer_holding_a_field_is_refused_at_registration():
    class Holding(Asked):
        n : int = 0
    with pytest.raises(TypeError) as found:
        answers(About)(Holding)
    assert str(found.value) == (f"{__name__}.{Holding.__qualname__} holds the fields ('n',): an "
                                f'answer is built by the registry, `cls()`, so whatever it needs is '
                                f'a class attribute or a method')


def test_an_answer_under_two_interfaces_is_refused_by_kind():
    class Twice(Asked, Also):
        pass
    with pytest.raises(TypeError) as found:
        answers(About)(Twice)
    assert str(found.value) == (f"{__name__}.{Twice.__qualname__} is under the interfaces "
                                f"('Asked', 'Also'): an answer is registered under one, a level "
                                f'directly under `Answer`')


def test_an_answer_is_registered_for_what_its_interface_is_about():
    '''An interface binds `about`; a class outside it is refused, a protocol met by shape, and
    an interface saying nothing is refused at the first registration.'''
    from cottax.core.frozen.answer import is_about
    from cottax.core import Label, Place

    class Numbers(Answer):
        about = int

    class Counted(Numbers):
        pass
    with pytest.raises(TypeError) as found:
        answers(bool, str)(Counted)
    assert str(found.value) == (f"{Numbers.__qualname__} is answered per int, and "
                                f"('builtins.str',) is not: an answer is registered for what its "
                                f'interface is about')

    class Unsaid(Answer):
        pass

    class Said(Unsaid):
        pass
    with pytest.raises(TypeError) as found:
        answers(About)(Said)
    assert str(found.value).startswith(f'{Unsaid.__qualname__} does not say what it is answered per')
    assert is_about(About, (Place, Label)) and is_about(bool, int) and not is_about(str, int)

    class Spelled:                              # a field without a default, and a property
        segments : tuple
        @property
        def prefixes(self): return ()

    class Half:
        segments : tuple
    assert is_about(Spelled, Place) and not is_about(Half, Place) and not is_about(int, Place)

    class Places(Answer):
        about = Place

    class Spelling(Places):
        pass
    with pytest.raises(TypeError, match=r'answered per Place, and .*Half.* is not'):
        answers(Half)(Spelling)


class Reloaded:
    pass


def test_an_answer_registered_again_under_its_own_name_replaces_the_first():
    '''A module reloaded in a notebook registers its answer again: same module and
    qualified name, a new class. Anything else is refused.'''
    from cottax.core.frozen import answer
    from cottax.core.frozen.answer import dotted

    def written() -> type[Asked]:
        class Mine(Asked):
            pass
        return Mine

    first, again = answers(Reloaded)(written()), answers(Reloaded)(written())
    assert first is not again and type(answer(Asked, Reloaded())) is again

    class Other(Asked):
        pass
    with pytest.raises(TypeError) as found:
        answers(Reloaded)(Other)
    assert str(found.value) == (f'{__name__}.Reloaded has the Asked answer {dotted(again)} already, '
                                f'so {dotted(Other)} is refused: which one answers would be chosen '
                                f'by import order')


from cottax.core.frozen import AbstractVar, Frozen, choice, choices


class Way(Frozen, Choice):
    step : AbstractVar[int]


class Sideways(Way):
    pass


def test_a_choice_final_is_marked_returned_unchanged_and_listed():
    '''Listed in marking order; marked again under its own name (a module reloaded), it
    replaces the first.'''
    class Listed(Frozen, Choice):
        step : AbstractVar[int]

    def written() -> type[Listed]:
        class Mine(Listed):
            step : int = 1
        return Mine

    class Yours(Listed):
        step : int = 2
    mine = written()
    assert choice(Listed)(mine) is mine and choice(Listed)(Yours) is Yours
    assert choices(Listed) == (mine, Yours)
    again = choice(Listed)(written())
    assert choices(Listed) == (again, Yours) and again is not mine
    assert choices(Way) == () or all(issubclass(c, Way) for c in choices(Way))


def test_a_choice_mark_names_a_choice_interface():
    for interface in (About, Sideways, Choice):
        with pytest.raises(TypeError) as found:
            choice(interface)
        assert str(found.value) == (f'{interface.__qualname__} is not a `Choice` interface: '
                                    f'`@choice` names a level directly under `Choice`')


def test_a_choice_mark_refuses_a_class_not_under_its_interface():
    with pytest.raises(TypeError) as found:
        choice(Way)(About)
    assert str(found.value) == 'About is marked a Way choice but does not extend Way'


def test_a_choice_mark_refuses_an_abstract_level():
    with pytest.raises(TypeError) as found:
        choice(Way)(Sideways)
    assert str(found.value) == (f"{__name__}.Sideways leaves ('step',) abstract, so it is not a Way "
                                f'a study can name: `@choice` marks a final')


def test_a_remembered_answer_is_of_the_value_and_outside_it():
    '''Asked once per value, as a `cached_property` is; equality, the hash and a copy leave
    it out, so a copy comes back cold.'''
    import copy
    from cottax.core.frozen import remembered
    asked = []

    class Pair(Frozen):
        a : int

    @remembered
    def doubled(p):
        asked.append(p.a)
        return 2 * p.a

    p = Pair(2)
    assert doubled(p) == doubled(p) == 4 and asked == [2]
    assert p == Pair(2) and hash(p) == hash(Pair(2))
    cold = copy.deepcopy(p)
    assert cold == p and doubled(cold) == 4 and asked == [2, 2]


def test_a_remembered_answer_keyed_by_a_constant_is_asked_once_per_key():
    '''`fn(value, key)`: one answer per key, each asked once.'''
    from cottax.core.frozen import remembered
    asked = []

    class Pair(Frozen):
        a : int

    @remembered
    def times(p, k):
        asked.append(k)
        return k * p.a

    p = Pair(2)
    assert times(p, 3) == times(p, 3) == 6 and times(p, 5) == 10 and asked == [3, 5]
