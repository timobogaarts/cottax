'''`OdeCondition`: each state with its rate and its initial value, `x' = r`, `x(t0) = x0`, over
one span.'''

from ..core import ConditionalNode, Place, Renaming, Violation
from ..core.frozen import Frozen


class Dt[V: Place](Frozen):
    '''`x' = r` from `x(t0) = x0`: `state` changes at the rate held at `rate`, starting from
    the value held at `initial`.'''

    state   : V
    rate    : V
    initial : V

    def relabel(self, renaming: Renaming[V]) -> 'Dt[V]':
        return Dt(*(renaming.get(p, p) for p in (self.state, self.rate, self.initial)))

    def __repr__(self) -> str:
        return f"{self.state}' = {self.rate} from {self.initial}"


class OdeCondition[V: Place](ConditionalNode[V]):
    '''`x' = r` from `x0` per relation, over the `span`: an initial-value problem, so one
    solution. It determines exactly its states, so it is always a problem: outside its cycle a
    state is its value at `t1`; inside, the current state. The initial values and the span
    are its parameters, held fixed while it is integrated. Supports addition.'''

    relations : tuple[Dt[V], ...]
    '''A state twice, or a state that is a rate or an initial value, is core's refusal
    (`OwnsTwice`, `ReadsOwn`).'''
    span      : V
    '''Where `(t0, t1)` is read: one for every state.'''

    @property
    def states(self) -> tuple[V, ...]:
        return tuple(d.state for d in self.relations)

    @property
    def rates(self) -> tuple[V, ...]:
        return tuple(d.rate for d in self.relations)

    @property
    def initials(self) -> tuple[V, ...]:
        return tuple(d.initial for d in self.relations)

    @property
    def parameters(self) -> tuple[V, ...]:
        '''The initial values, then the span.'''
        return (*self.initials, self.span)

    @property
    def unknowns(self) -> tuple[V, ...]:
        return self.states

    @property
    def conditions(self) -> tuple[V, ...]:
        return self.rates

    @property
    def reads(self) -> tuple[V, ...]:
        return self.rates + self.parameters

    @property
    def owns(self) -> tuple[V, ...]:
        return self.states

    def __add__(self, other: 'OdeCondition[V]') -> 'OdeCondition[V]':
        '''The join: the relations concatenated, this side first, over the one span both
        read. Two spans are refused: one integration runs over one.'''
        if not isinstance(other, OdeCondition):
            return NotImplemented
        if other.span != self.span:
            raise SpansDiffer(self.span, other.span).raised
        return OdeCondition(self.relations + other.relations, self.span)

    def determined(self, unknowns: tuple[V, ...]) -> 'OdeCondition[V]':
        '''Itself, handed exactly its states: what it determines is not edited.'''
        if tuple(unknowns) != self.states:
            raise DeterminesItsStates(self.states, tuple(unknowns)).raised
        return self

    def relabel(self, renaming: Renaming[V]) -> 'OdeCondition[V]':
        return OdeCondition(tuple(d.relabel(renaming) for d in self.relations),
                            renaming.get(self.span, self.span))

    def __repr__(self) -> str:
        return f'OdeCondition({"; ".join(map(repr, self.relations))}; over {self.span})'


def is_ode(node: object) -> bool:
    '''An ODE: what an integrator answers.'''
    return isinstance(node, OdeCondition)


# ----------------------------------------------------------------- violation finals

class DeterminesItsStates(Violation):
    states   : tuple
    unknowns : tuple
    error    = ValueError
    template = ('an ODE determines its states and nothing else: {states!r}, not {unknowns!r}')


class SpansDiffer(Violation):
    span  : object
    other : object
    error    = ValueError
    template = ('two ODEs over {span!r} and {other!r} are not one integration: read one span '
                'in both (`span=` the same place, or a `Rename`) before joining them')
