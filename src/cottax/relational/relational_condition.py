'''
The condition node written as relations: a `Rel` marker, a `Relation` between two places,
and `RelationalCondition`, relations and objectives over places with the unknowns they
determine and the parameters they hold fixed, and `+` the join.
'''

import dataclasses
from collections.abc import Iterator

from ..core import ConditionalNode, Place, Renaming, Violation, relabelled
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
    '''A parameter is held fixed: not moved (an unknown), not minimised (an objective), a
    term of some relation, and never the only kind of term in one.'''
    driven = {o.place for o in node.objectives}
    if moved := tuple(p for p in node.parameters if p in node.unknowns or p in driven):
        yield FixedAndMoved(moved)
    terms = {t for r in node.relations for t in r.terms}
    if unrelated := tuple(p for p in node.parameters if p not in terms):
        yield ParameterUnrelated(unrelated)
    if fixed := tuple(r for r in node.relations
                      if r.terms and all(t in node.parameters for t in r.terms)):
        yield FixedRelation(fixed)

class RelationalCondition[V: Place](ConditionalNode[V]):
    '''A condition node: relations and objectives over places, the unknowns it determines and
    the parameters it holds fixed. Every other term is a condition. Supports addition.'''

    relations  : tuple[Relation[V], ...]
    objectives : tuple[Objective[V], ...]
    unknowns   : tuple[V, ...]
    parameters : tuple[V, ...]
    '''Terms read and held fixed while the unknowns move: a bound, a limit. Read, not a
    condition, so a proof asks them off the problem's cycle.'''

    checks = (check_relates_something, check_parameters)

    @property
    def conditions(self) -> tuple[V, ...]:
        '''The objectives, then every relation's terms that are neither unknowns nor
        parameters, in order.'''
        return tuple(o.place for o in self.objectives) + tuple(
            t for r in self.relations for t in r.terms
            if t not in self.unknowns and t not in self.parameters)
    @property
    def reads(self) -> tuple[V, ...]:
        '''The conditions, then the parameters.'''
        return self.conditions + self.parameters

    @property
    def owns(self) -> tuple[V, ...]:
        return self.unknowns

    def __add__(self, other: 'RelationalCondition[V]') -> 'RelationalCondition[V]':
        '''The join: relations, objectives and unknowns concatenated, this side first; the
        parameters too, a place both hold fixed once.'''
        if not isinstance(other, RelationalCondition):
            return NotImplemented
        return RelationalCondition(self.relations  + other.relations,
                                   self.objectives + other.objectives,
                                   self.unknowns   + other.unknowns,
                                   self.parameters + tuple(p for p in other.parameters
                                                           if p not in self.parameters))

    def determined(self, unknowns: tuple[V, ...]) -> 'RelationalCondition[V]':
        '''Same ConditionalNode but determined by the unknowns.'''
        return dataclasses.replace(self, unknowns=unknowns)

    def relabel(self, renaming: Renaming[V]) -> 'RelationalCondition[V]':
        '''
        Relabels the relations, objectives, unknowns and parameters.
        '''
        return dataclasses.replace(
            self,
            relations  = tuple(r.relabel(renaming) for r in self.relations),
            objectives = tuple(o.relabel(renaming) for o in self.objectives),
            unknowns   = relabelled(self.unknowns,   renaming),
            parameters = relabelled(self.parameters, renaming),
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


class FixedAndMoved(Violation):
    places : tuple
    error    = ValueError
    template = ('{places!r} held fixed, but also an unknown or an objective: a parameter is '
                'read and never moved')


class ParameterUnrelated(Violation):
    places : tuple
    error    = ValueError
    template = '{places!r} held fixed, but no relation relates them: a parameter is a term'


class FixedRelation(Violation):
    relations : tuple
    error    = ValueError
    template = ('{relations!r}: every term held fixed, so no unknown moves the relation -- '
                'relate a parameter to an unknown or a condition')
