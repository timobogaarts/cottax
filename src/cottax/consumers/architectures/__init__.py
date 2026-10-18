r'''
Where the problems under a top go: MDF, IDF, SAND and MDA as one op each, over the cut
rules and a closing handed in, asking every problem met what it is.

Dependency:

interface      refusals          (interface: ModelReading, what a problem is to an
 ||  ||         ||  ||            architecture, three questions; Closing, how a group of
 ||  \/         ||  ||            cuts is closed, chosen. refusals: every refusal kind)
 ||  closing <--'   ||           (closing: closed, every group closed at a name of its own;
 ||   ||            ||            CloseFunctionalCycles, the op recording a rule)
 \/   \/            \/
architectures <-----'            (architectures: Architecture, its two scopes Monolithic and
 ||                               Coupled, the four levels and their finals MDF, IDF, SAND,
 ||                               MDA; Boxes: Bodies, Disciplines, Solves; Resolution)
 \/
part <-- declaration.interface   (part: Architecting, the four as a surface's methods, closed
                                  by the closing it holds)

interface_types                  (At, AtFirstMember: where a closure binds, what a closing
                                  is built on)

'''

from .interface_types import AtFirstMember
from .closing import CloseFunctionalCycles, closed
from .part import Architecting
from .architectures import (
    IDF, MDA, MDF, SAND, Analysis, Architecture, Bodies, Boxes, Coupled, Disciplines,
    IndividualFeasible, Monolithic, MultidisciplinaryFeasible, OnCycle, Resolution, Simultaneous,
    Solves,
)

__all__ = [
    'Architecture', 'Monolithic', 'Coupled',
    'MultidisciplinaryFeasible', 'IndividualFeasible', 'Simultaneous', 'Analysis',
    'MDF', 'IDF', 'SAND', 'MDA',
    'Boxes', 'Bodies', 'Disciplines', 'Solves', 'OnCycle', 'Resolution',
    'AtFirstMember', 'closed', 'CloseFunctionalCycles', 'Architecting',
]
