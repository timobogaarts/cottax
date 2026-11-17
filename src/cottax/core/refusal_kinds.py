'''Every refusal kind the maths defines, for a study matching a refusal by kind.'''

from .maths.cut_rules import CycleTooLarge, KeyRepeats, NoRule, OrderMismatch, SeveralRules
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
from .maths.conditional import DrivenOnOneSide
from .maths.kinds import OwnsNothing, ReadOnBothSides, StatesNothing
from .maths.nesting import (
    BlockTooSmall, BoxOwnsNothing, NestingLoop, NodeInTwoBlocks, NotAnEncloser, SelfNested, Unbound,
    Unreachable,
)
from .maths.plan.ops import (
    AlreadyCut, AttributeAlreadySet, AttributeNotSet, DoesNotRead, NoAttributeNamed,
    NoReaders, NotProduced, NotRenamed, OwnedInsideTwice, OwnsInsideDriven, OwnsInsideNotAChoice,
)
from .maths.plan.primitive import (
    AlreadyNested, CarriesAttributes, NameTaken, NodeTwice, NotAddable, NotBound,
    NotNested, NotOfKind, NothingDeleted, NothingInserted, NothingToCombine, Stranded,
)
from .maths.valued import InputMissing, NotAPlace, OwnedGiven

__all__ = [
    'CycleTooLarge', 'KeyRepeats', 'NoRule', 'OrderMismatch', 'SeveralRules',
    'AttributeDoesNotFit',
    'OwnInsideRead', 'OwnsNested', 'OwnsTwice', 'ReadInsideOwn', 'ReadsOwn',
    'NothingToAsk',
    'Constant', 'NoProblem', 'OutsideCycle', 'OwnReadInsideCycle', 'ReadInsideCycle',
    'RequirementUnanswered', 'SeveralOutermost', 'StillCyclic',
    'OwnAroundOwner', 'OwnedTwice', 'OwnerInsideRead', 'OwnInsideOwner', 'ReadInsideOwner',
    'DrivenOnOneSide', 'OwnsNothing', 'ReadOnBothSides', 'StatesNothing',
    'BlockTooSmall', 'BoxOwnsNothing', 'NestingLoop', 'NodeInTwoBlocks', 'NotAnEncloser',
    'SelfNested', 'Unbound', 'Unreachable',
    'AlreadyCut', 'DoesNotRead', 'NoReaders', 'NotProduced', 'NotRenamed', 'OwnedInsideTwice',
    'OwnsInsideDriven', 'OwnsInsideNotAChoice',
    'AttributeAlreadySet', 'AttributeNotSet', 'NoAttributeNamed',
    'CarriesAttributes',
    'AlreadyNested', 'NameTaken', 'NodeTwice', 'NotAddable', 'NotBound', 'NotNested',
    'NotOfKind', 'NothingDeleted', 'NothingInserted', 'NothingToCombine', 'Stranded',
    'InputMissing', 'NotAPlace', 'OwnedGiven',
]
