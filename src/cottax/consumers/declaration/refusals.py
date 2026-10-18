'''
Why a declaration is refused: a wrapper writing nothing, a signature it cannot read, a
declaration held but not in the model, a model with no place to find its names by or none of
its own, a value given where nothing reads it.
'''

from ...core import Violation


class NotPositional(Violation):
    fn        : str
    parameter : str
    kind      : str
    error    = TypeError
    template = ('{fn}: `{parameter}` is {kind}, so no read binds it -- every parameter is a '
                'read, bound by position')


class HeldByValue(Violation):
    fn        : str
    parameter : str
    error    = TypeError
    template = ('{fn}: `{parameter}` has a default, which is a setting held by value -- make '
                'it a place and read it')


class NoOutputs(Violation):
    fn   : str
    kind : str
    error    = TypeError
    template = ("{fn} writes no place: name its outputs, `@method('y')` or `method({fn}, 'y')` "
                "-- a bare `@method` hands it the function alone, so this {kind} names none")


class ReadsUnlike(Violation):
    fn         : str
    reads      : tuple
    parameters : tuple
    error    = ValueError
    template = ('{fn} takes {parameters!r} but reads are given for {reads!r}: one place per '
                'parameter')


class HeldNotInModel(Violation):
    held : tuple
    error    = ValueError
    template = ('{held!r} held by another declaration but not in the model: put each in a '
                'slot too, for what stays at its name')


class Nameless(Violation):
    models : tuple
    error    = TypeError
    template = ('{models!r} write no place, so nothing says how their nodes are named: a '
                'node writes something')


class Unnamed(Violation):
    model : str
    error    = TypeError
    template = ('{model} has no name of its own: put it in a slot, `{{"name": ...}}`, and '
                'write that name where its node name is due')


class NotAnInput(Violation):
    places : tuple
    error    = KeyError
    template = ('a value given at {places!r}, which the graph does not read from outside: only a '
                'boundary input takes a value')


__all__ = ['NoOutputs', 'NotPositional', 'HeldByValue', 'ReadsUnlike', 'HeldNotInModel', 'Nameless',
           'Unnamed', 'NotAnInput']
