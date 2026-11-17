r'''
The condition node as data: a `Condition`, the value it states, and what is attached to it
(an `Attachment`, its driver), held by `ConditionalNode`. Above the maths, which reads a
condition node through `AbstractConditionalNode`'s four names alone, and above the plan,
since its own op (`SetOwnsInside`) is an edit.

node <- ops      (node: Condition, Attachment, ConditionalNode, kind_of, is_driven;
                  ops: SetOwnsInside, what a condition node owns inside)
'''

from .node import Attachment, Condition, ConditionalNode, is_driven, kind_of
from .ops import SetOwnsInside
