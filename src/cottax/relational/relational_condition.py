'''
The condition written as relations: a `Rel` marker, a `Relation` between two places,
and `RelationalCondition`, relations and objectives over places with the unknowns they
determine and the parameters they hold fixed, and `join`, two stated as one.
'''

import dataclasses
from collections.abc import Iterator

from ..core import Condition, Place, Renaming, Violation, relabelled
from ..core.frozen import Frozen


class Rel:
    '''Marker for a relation, asked by identity: a relation holds the class itself. A new
    one subclasses `Rel` with a `symbol`.'''

    symbol : str
    '''How a relation writes it.'''

    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a marker, not a value')


class Eq(Rel):
    '''`lhs = rhs`.'''
    symbol = '='

class Le(Rel):
    '''`lhs <= rhs`.'''
    symbol = '<='

# -------------------- relation

def check_sides(relation: 'Relation') -> Iterator[Violation]:
    '''0=0 is not a relation. '''
    if relation.lhs is None and relation.rhs is None:
        yield NoSides()

def check_op(relation: 'Relation') -> Iterator[Violation]:
    '''The op is a `Rel` class, not an instance or a string.'''
    if not (isinstance(relation.op, type) and issubclass(relation.op, Rel)):
        yield NotARel(relation.op)

class Relation[V: Place](Frozen):
    '''`lhs op rhs` between two places; a side of `None` is zero.'''

    lhs : V | None
    rhs : V | None
    op  : type[Rel]

    checks = (check_sides, check_op)

    @property
    def terms(self) -> tuple[V, ...]:
        '''The sides that are places, left then right.'''
        return tuple(s for s in (self.lhs, self.rhs) if s is not None)

    def relabel(self, renaming: Renaming[V]) -> 'Relation[V]':
        return Relation(*(s if s is None else renaming.get(s, s)
                          for s in (self.lhs, self.rhs)), self.op)

    def __repr__(self) -> str:
        return (f'{"0" if self.lhs is None else self.lhs} {self.op.symbol} '
                f'{"0" if self.rhs is None else self.rhs}')

# -------------------- objective

class Sense:
    '''Marker for which way an objective is driven, asked by identity: an objective holds
    the class itself.'''

    symbol : str
    '''How an objective writes it.'''

    def __init__(self) -> None:
        raise TypeError(f'{type(self).__name__} is a marker, not a value')


class Min(Sense):
    '''Driven down.'''
    symbol = 'min'

class Max(Sense):
    '''Driven up.'''
    symbol = 'max'


def check_sense(objective: 'Objective') -> Iterator[Violation]:
    '''The sense is a `Sense` class, not an instance or a string.'''
    if not (isinstance(objective.sense, type) and issubclass(objective.sense, Sense)):
        yield NotASense(objective.sense)

class Objective[V: Place](Frozen):
    '''A place and the way it is driven: which way changes the answer, so it is the
    problem's, stored with the place it drives.'''

    place : V
    sense : type[Sense]

    checks = (check_sense,)

    def relabel(self, renaming: Renaming[V]) -> 'Objective[V]':
        return Objective(renaming.get(self.place, self.place), self.sense)

    def __repr__(self) -> str:
        return f'{self.sense.symbol} {self.place}'

#---------------------- relational condition

def check_relates_something(node: 'RelationalCondition') -> Iterator[Violation]:
    '''It relates something: a condition met by every value is not a condition.'''
    if not node.relations and not node.objectives:
        yield RelatesNothing(type(node).__name__)

def check_parameters(node: 'RelationalCondition') -> Iterator[Violation]:
    '''A parameter is held fixed: a term of some relation, and never the only kind of term
    in one. That it is never moved needs no rule here: a parameter is read outside, so one
    that is an unknown is refused as `ReadsOwn`, and one an objective is as
    `ReadOnBothSides`, a level above.'''
    terms = {t for r in node.relations for t in r.terms}
    if unrelated := tuple(p for p in node.parameters if p not in terms):
        yield ParameterUnrelated(unrelated)
    if fixed := tuple(r for r in node.relations
                      if r.terms and all(t in node.parameters for t in r.terms)):
        yield FixedRelation(fixed)

class RelationalCondition[V: Place](Condition[V]):
    '''A condition: relations and objectives over places, the unknowns it determines and
    the parameters it holds fixed. Every other term is read inside. Joins another (`join`).'''

    relations  : tuple[Relation[V], ...]
    objectives : tuple[Objective[V], ...]
    owns_inside   : tuple[V, ...]
    '''The unknowns it determines; none for a requirement.'''
    reads_outside : tuple[V, ...]
    '''The parameters: terms read and held fixed while the unknowns move: a bound, a
    limit. Read outside, so a proof asks them off the problem's cycle.'''
    owns_outside = ()

    checks = (check_relates_something, check_parameters)

    @property
    def unknowns(self) -> tuple[V, ...]:    return self.owns_inside
    @property
    def parameters(self) -> tuple[V, ...]:  return self.reads_outside
    @property
    def reads_inside(self) -> tuple[V, ...]:
        '''The objectives, then every relation's terms that are neither unknowns nor
        parameters, in order.'''
        return tuple(o.place for o in self.objectives) + tuple(
            t for r in self.relations for t in r.terms
            if t not in self.unknowns and t not in self.parameters)

    def join(self, other: 'RelationalCondition[V]') -> 'RelationalCondition[V]':
        '''The join with another relational condition: relations, objectives and unknowns
        concatenated, this side first; the parameters too, a place both hold fixed once.
        What `Combine` folds by, through this kind's `ModelReading`.'''
        return RelationalCondition(self.relations  + other.relations,
                                   self.objectives + other.objectives,
                                   self.unknowns   + other.unknowns,
                                   self.parameters + tuple(p for p in other.parameters
                                                           if p not in self.parameters))

    def relabel(self, renaming: Renaming[V]) -> 'RelationalCondition[V]':
        '''
        Relabels the relations, objectives, unknowns and parameters.
        '''
        return dataclasses.replace(
            self,
            relations  = tuple(r.relabel(renaming) for r in self.relations),
            objectives = tuple(o.relabel(renaming) for o in self.objectives),
            owns_inside   = relabelled(self.owns_inside,   renaming),
            reads_outside = relabelled(self.reads_outside, renaming),
        )

    def __repr__(self) -> str:
        stated = '; '.join([*(repr(o) for o in self.objectives),
                            *(repr(r) for r in self.relations)])
        over = (' over ' + ', '.join(str(u) for u in self.unknowns)
                if self.unknowns else ' (requirement)')
        fixed = (' fixing ' + ', '.join(str(p) for p in self.parameters)
                 if self.parameters else '')
        return f'{type(self).__name__}({stated}{over}{fixed})'


# ----------------------------------------------------------------- violation finals

class NoSides(Violation):
    error    = ValueError
    template = 'a relation with zero on both sides relates nothing'


class NotASense(Violation):
    sense : object
    error    = TypeError
    template = '{sense!r} is not a `Sense` class: an objective holds `Min`, `Max` or another'


class NotARel(Violation):
    op : object
    error    = TypeError
    template = '{op!r} is not a `Rel` class: a relation holds `Eq`, `Le` or another `Rel` itself'


class RelatesNothing(Violation):
    node : str
    error    = ValueError
    template = '{node}: relates nothing -- a condition met by every value is not a condition'


class ParameterUnrelated(Violation):
    places : tuple
    error    = ValueError
    template = '{places!r} held fixed, but no relation relates them: a parameter is a term'


class FixedRelation(Violation):
    relations : tuple
    error    = ValueError
    template = ('{relations!r}: every term held fixed, so no unknown moves the relation -- '
                'relate a parameter to an unknown or a condition')
