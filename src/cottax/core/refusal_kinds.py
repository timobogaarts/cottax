'''Every refusal kind the maths, the plan and the condition node define, for a study
matching a refusal by kind.'''

from .maths.definition import (
    AttributeDoesNotFit, OwnInsideRead, OwnsNested, OwnsTwice, ReadInsideOwn,
    ReadsOwn,
)
from .maths.dispatch import NothingToAsk
from .maths.executable import (
    Constant, NoProblem, OutsideCycle, OwnReadInsideCycle, ReadInsideCycle, RequirementUnanswered,
    SeveralOutermost, StillCyclic,
)
from .maths.graph import (
    OwnAroundOwner, OwnedTwice, OwnerInsideRead, OwnInsideOwner, ReadInsideOwner,
)
from .maths.kinds import OwnsNothing, ReadOnBothSides, StatesNothing
from .maths.nesting import (
    BlockTooSmall, BoxOwnsNothing, NestingLoop, NodeInTwoBlocks, NotAnEncloser, SelfNested, Unbound,
    Unreachable,
)
from .plan.ops import (
    AlreadyCut, AttributeAlreadySet, AttributeNotSet, DoesNotRead, NoAttributeNamed,
    NoReaders, NotProduced, NotRenamed,
)
from .conditional.ops import OwnedInsideTwice, OwnsInsideDriven, OwnsInsideNotAChoice
from .plan.primitive import (
    AlreadyNested, NameTaken, NotBound, NotNested, NothingDeleted, NothingInserted, Stranded,
)
from .maths.valued import InputMissing, NotAPlace, OwnedGiven

__all__ = [
    'AttributeDoesNotFit',
    'OwnInsideRead', 'OwnsNested', 'OwnsTwice', 'ReadInsideOwn', 'ReadsOwn',
    'NothingToAsk',
    'Constant', 'NoProblem', 'OutsideCycle', 'OwnReadInsideCycle', 'ReadInsideCycle',
    'RequirementUnanswered', 'SeveralOutermost', 'StillCyclic',
    'OwnAroundOwner', 'OwnedTwice', 'OwnerInsideRead', 'OwnInsideOwner', 'ReadInsideOwner',
    'OwnsNothing', 'ReadOnBothSides', 'StatesNothing',
    'BlockTooSmall', 'BoxOwnsNothing', 'NestingLoop', 'NodeInTwoBlocks', 'NotAnEncloser',
    'SelfNested', 'Unbound', 'Unreachable',
    'AlreadyCut', 'DoesNotRead', 'NoReaders', 'NotProduced', 'NotRenamed', 'OwnedInsideTwice',
    'OwnsInsideDriven', 'OwnsInsideNotAChoice',
    'AttributeAlreadySet', 'AttributeNotSet', 'NoAttributeNamed',
    'AlreadyNested', 'NameTaken', 'NotBound', 'NotNested', 'NothingDeleted', 'NothingInserted',
    'Stranded',
    'InputMissing', 'NotAPlace', 'OwnedGiven',
]
