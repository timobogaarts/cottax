'''Every refusal kind the maths defines, for a study matching a refusal by kind.'''

from .maths.cut_rules import CycleTooLarge, KeyRepeats, NoRule, OrderMismatch, SeveralRules
from .maths.definition import OwnInsideRead, OwnsNested, OwnsTwice, ReadInsideOwn, ReadsOwn
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
from .maths.plan.ops import (
    AlreadyCut, DoesNotRead, NoReaders, NotProduced, NotRenamed, OwnedInsideTwice,
    OwnsInsideNotAChoice,
)
from .maths.plan.primitive import (
    AlreadyNested, NameTaken, NodeTwice, NotAddable, NotBound, NotNested, NotOfKind,
    NothingDeleted, NothingInserted, NothingToCombine, Stranded,
)
from .maths.valued import InputMissing, NotAPlace, OwnedGiven

__all__ = [
    'CycleTooLarge', 'KeyRepeats', 'NoRule', 'OrderMismatch', 'SeveralRules',
    'OwnInsideRead', 'OwnsNested', 'OwnsTwice', 'ReadInsideOwn', 'ReadsOwn',
    'NothingToAsk',
    'Constant', 'NoProblem', 'OutsideCycle', 'OwnReadInsideCycle', 'ReadInsideCycle',
    'RequirementUnanswered', 'SeveralOutermost', 'StillCyclic',
    'OwnAroundOwner', 'OwnedTwice', 'OwnerInsideRead', 'OwnInsideOwner', 'ReadInsideOwner',
    'OwnsNothing', 'ReadOnBothSides', 'StatesNothing',
    'BlockTooSmall', 'BoxOwnsNothing', 'NestingLoop', 'NodeInTwoBlocks', 'NotAnEncloser',
    'SelfNested', 'Unbound', 'Unreachable',
    'AlreadyCut', 'DoesNotRead', 'NoReaders', 'NotProduced', 'NotRenamed', 'OwnedInsideTwice',
    'OwnsInsideNotAChoice',
    'AlreadyNested', 'NameTaken', 'NodeTwice', 'NotAddable', 'NotBound', 'NotNested',
    'NotOfKind', 'NothingDeleted', 'NothingInserted', 'NothingToCombine', 'Stranded',
    'InputMissing', 'NotAPlace', 'OwnedGiven',
]
