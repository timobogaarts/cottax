'''What a closing is built on: `At`, where its problem binds off the cycle and the group, and `AtFirstMember`.'''

from abc import abstractmethod
from collections.abc import Callable

from ...core import Label, Place, ToCut
from ...core.frozen import Frozen


type LabelToLabel[N: Label] = Callable[[N], N]


class At[N: Label, V: Place](Frozen):
    '''Where a closure binds, off the cycle it lies on (its members in binding order) and
    the group it closes.'''

    @abstractmethod
    def __call__(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> N:
        ...


class AtFirstMember[N: Label, V: Place](At[N, V]):
    '''
    At the cycle's first member in binding order, whatever the rule cut: one closure per
    cycle, so two groups on one cycle collide, which `closed` refuses by name.
    '''

    closure_of : LabelToLabel[N]

    def __call__(self, cycle: tuple[N, ...], group: ToCut[N, V]) -> N:
        return self.closure_of(cycle[0])

    def __repr__(self) -> str:
        return f'at_first_member({self.closure_of!r})'


__all__ = ['At', 'AtFirstMember']
