'''Every refusal kind the maths, the plan and the condition node define, for a study
matching a refusal by kind.'''

from .maths.definition import (
    OwnInsideRead, OwnsNested, OwnsTwice, ReadInsideOwn,
    ReadsOwn,
)
from .answers import NothingToAsk
from .maths.executable import (
    Constant, NoProblem, OutsideCycle, OwnReadInsideCycle, ReadInsideCycle, RequirementUnanswered,
    SeveralOutermost, StillCyclic,
)
from .maths.wellformed import (
    OwnAroundOwner, OwnedTwice, OwnerInsideRead, OwnInsideOwner, ReadInsideOwner,
)
from .maths.kinds import OwnsNothing, ReadOnBothSides, StatesNothing
from .maths.wellformed import EndOwnsNothingInside, NestingLoop, SelfNested, Unbound
from .maths.nesting import BlockTooSmall, BoxOwnsNothing, NodeInTwoBlocks, Unreachable
from .plan.ops import (
    AlreadyCut, AttributeAlreadySet, AttributeNotSet, DoesNotRead, NoAttributeNamed,
    NoReaders, NotProduced, NotRenamed,
)
from .conditional.ops import OwnedInsideTwice, OwnsInsideDriven, OwnsInsideNotAChoice
from .plan.primitive import (
    AlreadyNested, NameTaken, NotBound, NotNested, NothingDeleted, NothingInserted, Stranded,
)

__all__ = [
    'OwnInsideRead', 'OwnsNested', 'OwnsTwice', 'ReadInsideOwn', 'ReadsOwn',
    'NothingToAsk',
    'Constant', 'NoProblem', 'OutsideCycle', 'OwnReadInsideCycle', 'ReadInsideCycle',
    'RequirementUnanswered', 'SeveralOutermost', 'StillCyclic',
    'OwnAroundOwner', 'OwnedTwice', 'OwnerInsideRead', 'OwnInsideOwner', 'ReadInsideOwner',
    'OwnsNothing', 'ReadOnBothSides', 'StatesNothing',
    'BlockTooSmall', 'BoxOwnsNothing', 'NestingLoop', 'NodeInTwoBlocks', 'EndOwnsNothingInside',
    'SelfNested', 'Unbound', 'Unreachable',
    'AlreadyCut', 'DoesNotRead', 'NoReaders', 'NotProduced', 'NotRenamed', 'OwnedInsideTwice',
    'OwnsInsideDriven', 'OwnsInsideNotAChoice',
    'AttributeAlreadySet', 'AttributeNotSet', 'NoAttributeNamed',
    'AlreadyNested', 'NameTaken', 'NotBound', 'NotNested', 'NothingDeleted', 'NothingInserted',
    'Stranded',
]
