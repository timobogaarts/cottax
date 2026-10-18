.. _faq:

FAQ
===

Why the Frozen base?
--------------------

This package is designed to ensure that the graphs it builds are *always* valid, and the operations on them preserve that validity.
These graphs are therefore immutable: any change to a graph must prove that the new graph is actually valid too.
Furthermore, the checks are additive: every additional level (e.g. executable/runnable) only *adds* checks, never removes them.
This only holds if no subclass accidentally overrides checks or changes what a check reads. Python is rather permissive here, so to make that enforced
rather than convention, no implemented method may be overridden in a subclass.
You may extend such a class with additional methods (abstract or not), or add additional rules and checks though.

The ``Frozen`` base class implements this pattern; it is additionally used for the nodes, since they carry the same proof obligation.
