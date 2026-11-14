'''`OdeCondition`: each state with its rate and its initial value, `x' = r`, `x(t0) = x0`, over
one span, the place its time is given at, and the times the rate bends or jumps at.'''

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
    state is its value at `t1`; inside, the current state. Its time, where it names one, is
    owned alike: inside, the current time; outside, `t1`. The initial values, the span, the
    kinks and the jumps are its parameters, held fixed while it is integrated. Supports
    addition.'''

    relations : tuple[Dt[V], ...]
    '''A state twice, or a state that is a rate or an initial value, is core's refusal
    (`OwnsTwice`, `ReadsOwn`).'''
    span      : V
    '''Where `(t0, t1)` is read: one for every state.'''
    kinks     : tuple[V, ...] = ()
    '''Where times the rate bends at are read, each an array: continuous there, its derivative
    not -- a control interpolated between knots. An integrator lands its steps on them, or a
    step striding over one gets the derivatives across it wrong while the values stay right.'''
    jumps     : tuple[V, ...] = ()
    '''Where times the rate jumps at are read, each an array: a control held between knots.
    An integrator lands its steps on them and evaluates the rate afresh past each -- found
    exactly only by a rate reading `time`, not by a clock integrated as a state.'''
    time      : V | None = None
    '''Where the integration's time is given: owned, an unknown no rate determines -- the
    integrator gives it rather than solving for it. `None` for an autonomous rate.'''

    @property
    def states(self) -> tuple[V, ...]:
        return tuple(d.state for d in self.relations)

    @property
    def rates(self) -> tuple[V, ...]:
        return tuple(d.rate for d in self.relations)

    @property
    def initials(self) -> tuple[V, ...]:
        return tuple(d.initial for d in self.relations)

    owns_outside = ()

    @property
    def owns_inside(self) -> tuple[V, ...]:
        '''The states, then the time where there is one.'''
        return self.states + (() if self.time is None else (self.time,))

    @property
    def reads_inside(self) -> tuple[V, ...]:
        return self.rates

    @property
    def reads_outside(self) -> tuple[V, ...]:
        '''The initial values, the span, then the kinks and the jumps.'''
        return (*self.initials, self.span, *self.kinks, *self.jumps)

    def __add__(self, other: 'OdeCondition[V]') -> 'OdeCondition[V]':
        '''The join: the relations concatenated, this side first, over the one span both
        read, stepping on the kinks and jumps of both, at the one time either names. Two
        spans, or two times, are refused: one integration runs over one, on one clock.'''
        if not isinstance(other, OdeCondition):
            return NotImplemented
        if other.span != self.span:
            raise SpansDiffer(self.span, other.span).raised
        if None not in (self.time, other.time) and self.time != other.time:
            raise TimesDiffer(self.time, other.time).raised
        both = lambda a, b: tuple(dict.fromkeys(a + b))
        return OdeCondition(self.relations + other.relations, self.span,
                            both(self.kinks, other.kinks), both(self.jumps, other.jumps),
                            other.time if self.time is None else self.time)

    def determined(self, unknowns: tuple[V, ...]) -> 'OdeCondition[V]':
        '''Itself, handed exactly its states and time: what it determines is not edited.'''
        if tuple(unknowns) != self.owns_inside:
            raise DeterminesItsStates(self.owns_inside, tuple(unknowns)).raised
        return self

    def relabel(self, renaming: Renaming[V]) -> 'OdeCondition[V]':
        return OdeCondition(tuple(d.relabel(renaming) for d in self.relations),
                            renaming.get(self.span, self.span),
                            tuple(renaming.get(k, k) for k in self.kinks),
                            tuple(renaming.get(j, j) for j in self.jumps),
                            None if self.time is None else renaming.get(self.time, self.time))

    def __repr__(self) -> str:
        time = '' if self.time is None else f' in {self.time}'
        breaks = ''.join(f'; {name} at {", ".join(map(str, at))}'
                         for name, at in (('kinks', self.kinks), ('jumps', self.jumps)) if at)
        return (f'OdeCondition({"; ".join(map(repr, self.relations))}; over '
                f'{self.span}{time}{breaks})')


def is_ode(node: object) -> bool:
    '''An ODE: what an integrator answers.'''
    return isinstance(node, OdeCondition)


# ----------------------------------------------------------------- violation finals

class DeterminesItsStates(Violation):
    states   : tuple
    unknowns : tuple
    error    = ValueError
    template = ('an ODE determines its states and nothing else (its time with them, where it '
                'gives one): {states!r}, not {unknowns!r}')


class SpansDiffer(Violation):
    span  : object
    other : object
    error    = ValueError
    template = ('two ODEs over {span!r} and {other!r} are not one integration: read one span '
                'in both (`span=` the same place, or a `Rename`) before joining them')


class TimesDiffer(Violation):
    time  : object
    other : object
    error    = ValueError
    template = ('two ODEs in {time!r} and {other!r} are not one integration: one clock gives '
                'one time -- name one place in both (`time=` the same, or a `Rename`)')
