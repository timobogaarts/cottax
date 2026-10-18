'''
The value layer every class stands on: `Frozen`, the frozen dataclass base with its
abstract variables and its checks (`base.py`), `OrderedMap`, the mapping a value
stores (`ordered_map.py`), `Violation`, what a check yields and
how a refusal is raised or collected, `checked`, a function declaring its checks as a class
does, `name_of` and `dotted`, a callable or a class named in a message (`violation.py`), and how a feature's class gets into a consumer: an `Answer`
registered with `answers` and found by `answer`, or a `Choice`, its finals marked `choice`
(`answer.py`).
'''

from .base import AbstractClassVar, AbstractVar, Frozen, PerLevel, unchecked
from .ordered_map import OrderedMap
from .answer import Answer, Choice, answer, answer_for_class, answered, answers, choice
from .violation import Violation, checked, dotted, first, name_of, violations

__all__ = [
    'AbstractClassVar', 'AbstractVar', 'Frozen', 'PerLevel', 'unchecked', 'OrderedMap',
    'Answer', 'Choice', 'answer', 'answer_for_class', 'answered', 'answers', 'choice',
    'Violation', 'checked', 'first', 'violations', 'name_of', 'dotted',
]
