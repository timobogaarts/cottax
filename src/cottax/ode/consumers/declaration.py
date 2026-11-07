'''The ODE declarations, `OdeDeclaring`, chosen as a part of a study's surface, and its one kind, `Rate`.'''

from collections.abc import Iterator
from typing import Any

from ...core import ConditionalNode, Label, Mint, Place, PlaceToPlace, Violation, minted
from ...core.frozen import OrderedMap, choice, name_of
from ...core.driven import Driven
from ...consumers.declaration.interface import Declaring
from ...consumers.declaration.interface_types import Keyword, Paired, PlaceWithValue, Solve, declares
from ...consumers.surface.interface import Part
from .. import mints
from ..contracts import Integration
from ..ode_condition import Dt, OdeCondition


def check_named(rate: 'Rate') -> Iterator[Violation]:
    '''A history and an initial value each a state's, and a history recorded at times.'''
    fn = name_of(rate.fn)
    for keyword, named in (('record', rate.record), ('initial', rate.initial)):
        if stray := tuple(x for x in named if x not in rate.outputs):
            yield NotAState(fn, keyword, stray)
    if bool(rate.record) != (rate.times is not None):
        yield RecordUnpaired(fn, 'record' if rate.record else 'times',
                             'times' if rate.record else 'record')


class Rate[N: Label, V: Place](Paired[N, V]):
    '''A function writing `^rate.x` per state (its `outputs`), in that order, and at
    `^solve.at` the ODE determining the states from their initial values (`initial`, else
    `^initial.x`) over the span (`span`, else `^span` of the first state), giving its time
    where `time` names a place, stepping on the times it bends at (`kinks`) and jumps at
    (`jumps`). Where a study names
    a history (`record`, at `times`) the ODE is declared driven, by `Integration(trajectory=True)`,
    so a model reading a history is coupled to it from the declaration on.'''

    record  : OrderedMap[V, V]
    '''The place each recorded state's history is written to.'''
    times   : PlaceWithValue[V] | None
    '''Where the times a history is saved at are read.'''
    initial : OrderedMap[V, PlaceWithValue[V]]
    '''Where each named state's initial value is read.'''
    span    : PlaceWithValue[V] | None
    '''Where the span is read.'''
    kinks   : PlaceWithValue[V] | None = None
    '''Where the times the rate bends at are read.'''
    jumps   : PlaceWithValue[V] | None = None
    '''Where the times the rate jumps at are read.'''
    time    : V | None = None
    '''Where the integration's time is given, for the rate (or anything on its cycle) to read.'''

    checks = (check_named,)
    condition_mint = Solve

    @property
    def function_outputs(self) -> tuple[V, ...]:
        return tuple(minted(mints.Rate(), x) for x in self.outputs)

    @property
    def terms(self) -> tuple[PlaceWithValue[V], ...]:
        return tuple(t for t in (self.times, self.span, *self.initial.values(), self.kinks,
                                 self.jumps) if t is not None)

    @property
    def condition(self) -> OdeCondition[V]:
        def initial(x: V) -> V:
            return self.initial[x].place if x in self.initial else minted(mints.Initial(), x)
        first = self.outputs[0]
        span = self.span.place if self.span is not None else minted(mints.Span(), first)
        breaks = lambda at: () if at is None else (at.place,)
        return OdeCondition(tuple(map(Dt, self.outputs, self.function_outputs,
                                      map(initial, self.outputs))), span,
                            breaks(self.kinks), breaks(self.jumps), self.time)

    def condition_node(self, at: N) -> ConditionalNode[V]:
        '''The ODE, driven where a history is recorded: the times where `times` says, each
        history where `record` names it, else at `^trajectory.x` -- the time's too, the
        times themselves.'''
        if not self.record:
            return self.condition
        histories = tuple(self.record.get(x) or minted(mints.Trajectory(), x)
                          for x in self.condition.unknowns)
        return Driven(self.condition, Integration(trajectory=True), ((self.times.place,),),
                      (histories,))

    @property
    def problem(self) -> N:
        '''Where its ODE binds, the rate handed to `declare` loose: `^solve.<fn>`.'''
        return self.loose_condition()


def given_at[V](surface: Declaring, x: V, written: Any, value_at: PlaceToPlace[V]
                ) -> PlaceWithValue[V]:
    '''What is written where a place about `x` is due, a value given at `value_at(x)`: the
    surface's `term`, zero too a value here (a state starting at rest, a span from zero).'''
    at = surface.term(x, written, value_at)
    return PlaceWithValue(value_at(x), written) if at is None else at


def record_of[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], record: Any
                 ) -> OrderedMap[V, V]:
    '''`record=`: each recorded state's history, a mapping or one for the one state there is.'''
    if record is None:
        return OrderedMap()
    written = surface.per_place(outputs, record)
    return OrderedMap((x, surface.place(h)) for x, h in written.items())


def initial_of[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], initial: Any
                  ) -> OrderedMap[V, PlaceWithValue[V]]:
    '''`initial=`: each named state's initial value, a place or a value given at `^initial.x`.'''
    if initial is None:
        return OrderedMap()
    written = surface.per_place(outputs, initial)
    return OrderedMap((x, given_at(surface, x, w, mints.Initial()))
                      for x, w in written.items())


def per_problem(purpose: Mint):
    '''`times=`, `span=`, `kinks=` or `jumps=`: a place, or a value given at `purpose`'s name
    of the first state.'''
    def read[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], written: Any
                ) -> PlaceWithValue[V] | None:
        if written is None:
            return None
        return given_at(surface, outputs[0], written, purpose)
    return read


def time_of[V](surface: Declaring, fn: Any, outputs: tuple[V, ...], time: Any) -> V | None:
    '''`time=`: the place the integration's time is given at, a name or a place; a value
    is no time.'''
    return None if time is None else surface.place(time)


@choice(Part)
class OdeDeclaring(Declaring):
    '''The ODE declarations: a rate per state, and the ODE integrating it.'''

    rate = declares(
        Rate,
        '''A function returning the rate of each state, and the ODE determining the states:
        `rate(fn, *states)`, or `@rate(*states)`. `initial=` (a mapping, or one for the one
        state) and `span=` say where those are read, a value written there given; two rates
        joined by `combine` read one span. `time='t'` gives the integration's time at `t`, for
        the rate to read. `kinks=` and `jumps=` say where the times the rate bends or jumps at
        are read -- the knots of a control it interpolates or holds -- for the integrator to
        step on; a jump is found exactly only by a rate reading `time`. `record={'x': 'x_history'}` with `times='ts'`
        declares the ODE driven, writing each named state at every time.''',
        record=Keyword('record', record_of),
        times=Keyword('times', per_problem(mints.Times())),
        initial=Keyword('initial', initial_of),
        span=Keyword('span', per_problem(mints.Span())),
        kinks=Keyword('kinks', per_problem(mints.Kinks())),
        jumps=Keyword('jumps', per_problem(mints.Jumps())),
        time=Keyword('time', time_of))


# ----------------------------------------------------------------- violation finals

class NotAState(Violation):
    fn      : str
    keyword : str
    places  : tuple
    error    = ValueError
    template = '{fn}: `{keyword}=` names {places!r}, which are not its states'


class RecordUnpaired(Violation):
    fn      : str
    given   : str
    missing : str
    error    = ValueError
    template = ('{fn}: `{given}=` is given without `{missing}=` -- a history is saved at times, '
                'so name both or neither')
