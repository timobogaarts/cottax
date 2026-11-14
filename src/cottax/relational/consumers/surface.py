'''
Relational's part of a `Surface`: `RelationalEditing`, a requirement determined and cuts
closed by a fixed point.
'''

from collections.abc import Mapping

from ...core import Closure, Cut, Label, Place, PlaceToPlace
from .. import Determine, FixedPointCut
from ...consumers.surface.interface import Editing


class RelationalEditing(Editing):
    '''The relational edits a study writes by hand: a requirement determined, and cuts
    closed by a fixed point.'''

    def determine(self, requirement: Label, *unknowns: Place) -> Determine:
        '''The requirement determines `unknowns`: a problem.'''
        return Determine(requirement, unknowns)

    def fixed_point_cut(self, cuts: Place | Mapping[Place, tuple[Label, ...]],
                        readers: tuple[Label, ...] | None = None, *, at: Label | None = None,
                        copy_of: PlaceToPlace | None = None) -> FixedPointCut:
        '''
        Cuts closed by one problem holding each copy equal to what it copies (`^copy.y = y`):
        one variable and its readers, `fixed_point_cut('y_1', D2)`, or several closed
        together, `fixed_point_cut({'y_1': D2, 'y_2': (D1,)}, at='mda')`. The problem binds
        at `^closure.<the first variable>` unless `at` names it; each copy where `copies`
        says unless `copy_of` does.
        '''
        if isinstance(cuts, Mapping):
            if readers is not None:
                raise TypeError('fixed_point_cut: give a mapping of each variable to its '
                                'readers, or one variable and its readers -- not both')
            written = tuple(cuts.items())
        elif readers is None:
            raise TypeError(f'fixed_point_cut({cuts!r}): name the readers that read the copy: '
                            f'fixed_point_cut({cuts!r}, reader)')
        else:
            written = ((cuts, readers),)
        made = tuple(Cut(var, by, copy_of or self.copies) for var, by in written)
        return FixedPointCut(made, at if at is not None else
                             Closure()(self.derived.node_at(made[0].var)))


__all__ = ['RelationalEditing']
