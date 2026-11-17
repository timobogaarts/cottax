'''`DoeCondition`: each factor set to every point of its plan, each response recorded there --
a design of experiments.'''

from ..core import Condition, Place, Renaming
from ..core.frozen import Frozen


class Varied[V: Place](Frozen):
    '''`factor` set to each point of the plan held at `plan`: an array, its leading axis the
    points.'''

    factor : V
    plan   : V

    def relabel(self, renaming: Renaming[V]) -> 'Varied[V]':
        return Varied(*(renaming.get(p, p) for p in (self.factor, self.plan)))

    def __repr__(self) -> str:
        return f'{self.factor} in {self.plan}'


class Recorded[V: Place](Frozen):
    '''`response` at each point, stacked along a leading axis at `record`.

    Workaround, pending the history rework: a record is a driver's history in concept, as an
    integrator's `Trajectory` is, and the two stand apart until that rework unifies them.'''

    response : V
    record   : V

    def relabel(self, renaming: Renaming[V]) -> 'Recorded[V]':
        return Recorded(*(renaming.get(p, p) for p in (self.response, self.record)))

    def __repr__(self) -> str:
        return f'{self.response} into {self.record}'


class DoeCondition[V: Place](Condition[V]):
    '''Each factor set to every point of its plan, `x_i = p(i)` for `i = 1..n`, and each
    response recorded at each point, `h_i = F(x_i)`, `F` what its cycle computes. It owns the
    factors inside and reads the responses there; it reads the plans outside and owns the
    records outside. It combines with nothing.

    Workaround, pending the history rework: the records are the condition's own
    (`owns_outside`), and that rework may move them to the driver.

    Pending a check of the condition kind: outside its cycle a factor, and anything owned on
    the cycle, holds no answer -- what one point left -- and nothing refuses a model off the
    cycle reading it.

    Work in progress: a DOE is nested in nothing. Nesting one proves, and is unsupported until
    that check exists.'''

    factors   : tuple[Varied[V], ...]
    responses : tuple[Recorded[V], ...]

    @property
    def plans(self) -> tuple[V, ...]:
        return tuple(v.plan for v in self.factors)

    @property
    def records(self) -> tuple[V, ...]:
        return tuple(r.record for r in self.responses)

    @property
    def owns_inside(self) -> tuple[V, ...]:
        return tuple(v.factor for v in self.factors)

    @property
    def reads_inside(self) -> tuple[V, ...]:
        return tuple(r.response for r in self.responses)

    @property
    def reads_outside(self) -> tuple[V, ...]:
        return self.plans

    @property
    def owns_outside(self) -> tuple[V, ...]:
        return self.records

    def relabel(self, renaming: Renaming[V]) -> 'DoeCondition[V]':
        return DoeCondition(tuple(v.relabel(renaming) for v in self.factors),
                            tuple(r.relabel(renaming) for r in self.responses))

    def __repr__(self) -> str:
        return f'DoeCondition({"; ".join(map(repr, (*self.factors, *self.responses)))})'


def is_doe(node: object) -> bool:
    '''A design of experiments: what a sampler answers.'''
    return isinstance(node, DoeCondition)
