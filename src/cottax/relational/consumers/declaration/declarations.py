'''
The relational declarations: a function paired with a condition node over what it outputs
(an implicit model, a constraint), and a problem holding the constraints it is subject to.
'''
from collections.abc import Iterator
from typing import Any

from ....core import ConditionalNode, Label, NodeDefinition, Place, Violation, is_requirement, minted
from ....core.frozen import AbstractVar, Frozen, OrderedMap, violations, name_of
from ....consumers.surface.interface import DerivedNames
from ...mints import Cond, Require
from ....consumers.declaration.interface_types import (
    Declaration, Paired, PlaceWithValue, Solve, Tags, given_in, merged_tags,
)
from ... import Eq, Le, Objective, Relation, RelationalCondition


# ----------------------------------- Implicit Models
class Implicit[N: Label, V: Place](Paired[N, V]):
    '''An implicit model: a function writing `^cond.u` per unknown (its `outputs`), in that
    order, and at `^solve.at` the problem determining the unknowns.'''

    terms = ()
    condition_mint = Solve

    @property
    def function_outputs(self) -> tuple[V, ...]:
        return tuple(minted(Cond(), u) for u in self.outputs)

    def condition_node(self, at: N) -> ConditionalNode[V]:
        return self.condition

    @property
    def problem(self) -> N:
        '''Where its problem binds, the model handed to `declare` loose: `^solve.<fn>`.'''
        return self.loose_condition()


class Residual[N: Label, V: Place](Implicit[N, V]):
    '''The function returns a residual per unknown, each held at zero: `^cond.u = 0`.'''

    @property
    def condition(self) -> RelationalCondition[V]:
        return RelationalCondition(tuple(Relation(r, None, Eq) for r in self.function_outputs),
                                   (), self.outputs, ())


class FixedPoint[N: Label, V: Place](Implicit[N, V]):
    '''The function returns the next value of each unknown, each held equal to it: a fixed
    point, `u = ^cond.u`.'''

    @property
    def condition(self) -> RelationalCondition[V]:
        return RelationalCondition(
            tuple(Relation(u, g, Eq) for u, g in zip(self.outputs, self.function_outputs)),
            (), self.outputs, ())
# -------------------------------------- constraints

class Bounds[V: Place](Frozen):
    '''
    Per place, a lower side, an upper side, or both: each a place (with its value where a
    number was written), or `None` for zero; a place absent from a side is unbounded there.
    '''

    at_least : OrderedMap[V, PlaceWithValue[V] | None]
    at_most  : OrderedMap[V, PlaceWithValue[V] | None]

    def relations(self, among: tuple[V, ...]) -> tuple[Relation[V], ...]:
        '''`lo <= x` and `x <= hi` for each of `among` that a side bounds; a side of `None`
        is zero.'''
        out = []
        for x in among:
            if x in self.at_least:
                lo = self.at_least[x]
                out.append(Relation(None if lo is None else lo.place, x, Le))
            if x in self.at_most:
                hi = self.at_most[x]
                out.append(Relation(x, None if hi is None else hi.place, Le))
        return tuple(out)

    @property
    def bounded(self) -> tuple[V, ...]:
        '''Every place a side bounds, once.'''
        return tuple(dict.fromkeys((*self.at_least, *self.at_most)))

    @property
    def parameters(self) -> tuple[V, ...]:
        '''Every side that is a place, once: held fixed while the bounded places move.'''
        return tuple(dict.fromkeys(p.place for side in (self.at_least, self.at_most)
                                   for p in side.values() if p is not None))

    @property
    def terms(self) -> tuple[PlaceWithValue[V] | None, ...]:
        '''Each side as written, the lower then the upper.'''
        return (*self.at_least.values(), *self.at_most.values())


def check_held(constraint: 'Constraint') -> Iterator[Violation]:
    '''Every place held against something is an output, and every output is held.'''
    fn, held = name_of(constraint.fn), constraint.held_against
    if stray := tuple(x for x in held if x not in constraint.function_outputs):
        yield AgainstNoOutput(fn, stray)
    if free := tuple(o for o in constraint.function_outputs if o not in held):
        yield HeldAgainstNothing(fn, free)


class Constraint[N: Label, V: Place](Paired[N, V]):
    '''A function computing its outputs, and at `condition_at` (`^require.at`) the
    requirement holding each against something. It determines nothing: an architecture
    absorbs it into the problem its conditions reach.'''

    held_against : AbstractVar[tuple[V, ...]]
    '''Every place the constraint holds against something: each output, and nothing else.'''

    checks = (check_held,)
    condition_mint = Require

    @property
    def function_outputs(self) -> tuple[V, ...]:
        return self.outputs

    def condition_node(self, at: N) -> ConditionalNode[V]:
        return self.condition

    @property
    def requirement(self) -> N:
        '''Where its requirement binds, the constraint handed to `declare` loose:
        `^require.<fn>`.'''
        return self.loose_condition()


class InequalityConstraint[N: Label, V: Place](Constraint[N, V]):
    '''Each output within its bounds: `g <= g_max`, `q >= q_min`, `lo <= t <= hi`, zero
    where a side is `None`.'''

    bounds : Bounds[V]

    @property
    def held_against(self) -> tuple[V, ...]:
        return self.bounds.bounded

    @property
    def terms(self) -> tuple[PlaceWithValue[V] | None, ...]:
        return self.bounds.terms

    @property
    def condition(self) -> RelationalCondition[V]:
        return RelationalCondition(self.bounds.relations(self.function_outputs), (), (),
                                   self.bounds.parameters)


class EqualityConstraint[N: Label, V: Place](Constraint[N, V]):
    '''Each output equal to its target: `h = h_target`, or `h = 0` where it is `None`.'''

    targets : OrderedMap[V, PlaceWithValue[V] | None]
    '''Each output's target (with its value where a number was written), or `None` for
    zero.'''

    @property
    def held_against(self) -> tuple[V, ...]:
        return tuple(self.targets)

    @property
    def terms(self) -> tuple[PlaceWithValue[V] | None, ...]:
        return tuple(self.targets.values())

    @property
    def condition(self) -> RelationalCondition[V]:
        relations = tuple(Relation(o, None if (t := self.targets[o]) is None else t.place, Eq)
                          for o in self.function_outputs)
        return RelationalCondition(relations, (), (), tuple(dict.fromkeys(
            t.place for t in self.targets.values() if t is not None)))

# ---------------------------------------------------------------- a problem

def check_subject_to(problem: 'Problem') -> Iterator[Violation]:
    '''A problem is subject to constraint wrappers and requirements, nothing else.'''
    for c in problem.subject_to:
        if not (isinstance(c, Constraint) or (isinstance(c, RelationalCondition)
                                              and is_requirement(c))):
            yield NotHeldAs(repr(c))


def check_bounds_on_unknowns(problem: 'Problem') -> Iterator[Violation]:
    '''Every bounded place is an unknown: a bound on anything else would hold nothing.'''
    if stray := tuple(str(x) for x in problem.bounds.bounded if x not in problem.unknowns):
        yield BoundNotUnknown(stray)


def check_condition(problem: 'Problem') -> Iterator[Violation]:
    '''Its condition is a condition node: refused here, where it is written, not when declared.'''
    refused = (*check_subject_to(problem), *check_bounds_on_unknowns(problem))
    if not refused:                                      # its own refusals come first
        yield from violations(RelationalCondition, *problem._condition_fields)


class Problem[N: Label, V: Place](Declaration[N, V]):
    '''
    A relational condition node declared alone, with no body: the unknowns it determines,
    their bounds, its objectives, and the constraints and requirements it is subject to.
    '''

    unknowns   : tuple[V, ...]
    bounds     : Bounds[V]
    '''On the unknowns: `(lo, hi)` as written.'''
    objectives : tuple[Objective[V], ...]
    '''One for `optimise`, none for `satisfy`.'''
    subject_to : tuple[Constraint[N, V] | RelationalCondition[V], ...]
    '''Constraints and requirements held in this problem's one condition node.'''

    checks = (check_subject_to, check_bounds_on_unknowns, check_condition)

    @property
    def holds(self) -> tuple[Constraint[N, V] | RelationalCondition[V], ...]:
        return self.subject_to

    @property
    def outputs(self) -> tuple[V, ...]:
        return self.unknowns

    def nodes_when_held(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        '''Nothing: what holds it takes its condition.'''
        return ()

    @property
    def held(self) -> tuple[RelationalCondition[V], ...]:
        '''The requirement of each constraint it is subject to.'''
        return tuple(c.condition if isinstance(c, Constraint) else c for c in self.subject_to)

    @property
    def _condition_fields(self) -> tuple:
        return ((*self.bounds.relations(self.unknowns), *(r for h in self.held for r in h.relations)),
                self.objectives, self.unknowns,
                tuple(dict.fromkeys((*self.bounds.parameters,
                                     *(p for h in self.held for p in h.parameters)))))

    @property
    def condition(self) -> RelationalCondition[V]:
        return RelationalCondition(*self._condition_fields)

    @property
    def given(self) -> OrderedMap[V, Any]:
        return given_in(self.bounds.terms)

    @property
    def tags(self) -> Tags[V]:
        return merged_tags((t.place, t.tags) for t in self.bounds.terms if t is not None)

    def nodes(self, at: N) -> tuple[tuple[N, NodeDefinition[V]], ...]:
        return ((at, self.condition),)

    def loose_at(self, names: DerivedNames[N, V]) -> N | None:
        '''`^solve.<objective>`; none with no objective.'''
        if not self.objectives:
            return None
        return minted(Solve(), names.node_at(self.objectives[0].place))

# ----------------------------------------------------------------- violation finals

class AgainstNoOutput(Violation):
    fn     : str
    places : tuple
    error    = ValueError
    template = '{fn}: a limit given for {places!r}, which it does not output'


class HeldAgainstNothing(Violation):
    fn     : str
    places : tuple
    error    = ValueError
    template = ('{fn}: {places!r} held against nothing -- once any limit is given every '
                'output is named, `0` for zero')


class BoundNotUnknown(Violation):
    places : tuple
    error    = ValueError
    template = 'bounds on {places!r}, which are not unknowns of the problem'


class NotHeldAs(Violation):
    constraint : str
    error    = TypeError
    template = '{constraint}: a problem is subject to a constraint or a requirement'


class NotAPair(Violation):
    place   : object
    written : str
    error    = TypeError
    template = ('{place!r}: its bounds are a pair `(lo, hi)`, each a name, a value or None, and '
                '{written} is not one')
