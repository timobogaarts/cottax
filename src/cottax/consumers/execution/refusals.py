'''
Why a run is refused: a place a data structure does not index, an implementation it cannot
call, a problem `Drive` cannot give one, a body returning a miscount or failing when traced, a
graph proved below what runs, a node that cannot be a jit cache key, a value given twice or
none.
'''

from ...core import Violation






class ForeignPlace(Violation):
    places    : tuple
    classes   : tuple
    structure : str
    own       : str
    error    = TypeError
    template = ('{places!r}, of classes {classes}: {structure} indexes a {own} alone, so it '
                'neither reads nor writes them')


class NotCallable(Violation):
    node           : object
    implementation : object
    error    = TypeError
    template = ('{node!r} is implemented by {implementation!r}, which describes what runs it '
                'and is not itself something to call -- `Implement` it with a callable here')


class NotADriverImplementation(Violation):
    node           : object
    implementation : object
    error    = TypeError
    template = ('{node!r} is driven, and run by {implementation!r}, which is not a '
                '`DriverImplementation`: nothing says which driver it implements')


class ImplementsAnother(Violation):
    node       : object
    driver     : str
    implements : str
    error    = TypeError
    template = ('{node!r} is driven by a {driver}, and its implementation implements a '
                '{implements}: what runs it is not what was accepted')


class DataMissing(Violation):
    node           : object
    implementation : str
    roles          : tuple
    error    = TypeError
    template = ('{node!r} is run by {implementation}, which needs data its driver does not '
                'take: {roles}')


class ReportUnavailable(Violation):
    node           : object
    implementation : str
    roles          : tuple
    error    = TypeError
    template = ('{node!r} asks its driver for reports {implementation} cannot produce: '
                '{roles}')


class NoneAccepts(Violation):
    node    : object
    problem : object
    error    = ValueError
    template = '{node!r}: no implementation given accepts {problem!r}'


class NoneImplements(Violation):
    node     : object
    contract : str
    error    = ValueError
    template = '{node!r}: no implementation given implements its contract, a {contract}'


class NoDefaultContract(Violation):
    implementation : object
    implements     : str
    error    = TypeError
    template = ('{implementation!r} implements {implements}, which has no default contract: '
                '`Assign` it by hand')


class ReturnedUnlike(Violation):
    node    : object
    count   : int
    outputs : tuple
    got     : str
    error    = ValueError
    template = ('{node!r} returned {got} for its {count} outputs {outputs!r}: a body returns '
                'one value per output, in order -- one output takes the whole return value')


class TraceFailed(Violation):
    node   : object
    reason : str
    error    = TypeError
    template = ('{node!r} fails when traced at these shapes ({reason}): a body jax cannot '
                'trace, or shapes that do not match')


class ProvedBelow(Violation):
    proof : str
    error    = TypeError
    template = ('a run takes a `CallableGraph`, and the graph is a {proof}: prove it a '
                '`CallableGraph`')


class Unhashable(Violation):
    node   : object
    kind   : str
    count  : int
    leaves : str
    error    = TypeError
    template = ('{node!r}: a {kind} holding {count} unhashable leaf/leaves ({leaves}) cannot '
                'be a jit cache key. A value carried in the graph is an input the graph does '
                'not name -- give it an output on a node of its own, so it can be supplied '
                'without rebuilding')


class BodyByIdentity(Violation):
    node : object
    fn   : object
    error    = TypeError
    template = ('{node!r}: {fn!r} is a bound method, minted fresh on every access and '
                'comparing its receiver by identity, so re-assembling this graph gives a '
                'fresh jit cache key. Pass the object itself if it is callable, or a frozen '
                'dataclass wrapping the call')


class PartialBody(Violation):
    node : object
    fn   : object
    error    = TypeError
    template = ('{node!r}: {fn!r} is a `functools.partial`, which compares by identity, so '
                're-assembling this graph gives a fresh jit cache key. Pass a frozen '
                'dataclass whose fields are the bound arguments')


class NotDriverData(Violation):
    problem : object
    role    : str
    why     : str
    error    = KeyError
    template = ('`values[{role}]` names an unknown of {problem!r}, {why}: only a driven problem\'s '
                'own roles take data')


class NotAnUnknown(Violation):
    role    : str
    written : object
    why     : str
    error    = KeyError
    template = ('`values[{role}]` names {written!r}, {why}: a driver\'s data is keyed by an '
                'unknown of the problem it drives')


class TakesNoValue(Violation):
    problem : object
    role    : str
    place   : object
    error    = ValueError
    template = ('{problem!r}\'s `{role}` is at {place!r}, which a node computes: it takes no '
                'value')


class NotAnInput(Violation):
    places : tuple
    error    = KeyError
    template = ('`values[Inputs]` names {places!r}, which the graph does not read from outside: '
                'only a boundary input takes a value')


class ValuedTwice(Violation):
    place   : object
    sources : tuple
    error    = ValueError
    template = '{place!r} is given twice, by {sources}: an input has one source'


class Unvalued(Violation):
    missing : str
    error    = KeyError
    template = ('no value for {missing} -- by name in `values={{Inputs: {{name: value}}, '
                'Role: {{unknown: value}}}}`, or from a structure in `read_from`')


class NoSlotFor(Violation):
    role : str
    error    = TypeError
    template = ('{role} by structure: a {role} is one per problem, at no unknown\'s place in a '
                'structure -- give or read it by name')


class NoRoleAt(Violation):
    role    : str
    written : object
    error    = KeyError
    template = ('`out[{role}, {written!r}]`: no driver takes or reports a {role} for the problem '
                'determining it -- a role is read at an unknown of its driven problem')


class NoneTakes(Violation):
    role : str
    error    = KeyError
    template = '{role} by structure: no driver of the graph takes or reports a {role}'


__all__ = [
    'ForeignPlace', 'NotCallable', 'NotADriverImplementation',
    'ImplementsAnother', 'DataMissing', 'ReportUnavailable', 'NoneAccepts', 'NoneImplements',
    'NoDefaultContract',
    'ReturnedUnlike', 'TraceFailed', 'ProvedBelow', 'Unhashable',
    'BodyByIdentity', 'PartialBody',
    'NotDriverData', 'NotAnUnknown', 'TakesNoValue', 'NotAnInput', 'ValuedTwice', 'Unvalued', 'NoSlotFor', 'NoRoleAt',
    'NoneTakes',
]
