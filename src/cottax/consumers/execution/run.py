'''
What can run (`CallableGraph`), and running it (`run`).
'''

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from ...core import (
    FunctionalNode, GraphWithValues, Label, NodeDefinition, Place, Run, SteppedExecutableGraph,
    Violation,
)
from ...core.frozen import OrderedMap, checked
from ...core.driven import DrivenNode
from ...core.realisation import ImplementedDriven, RealisedGraph, is_implemented
from .interface import DriverImplementation
from .interface_types import ConditionMap, ConditionValues, Context, Posed, Unknowns
from .refusals import (
    DataMissing, ImplementsAnother, NotADriverImplementation, NotCallable, ProvedBelow,
    ReportUnavailable, ReturnedUnlike,
)




# ============================================================================ what can run
# Realised Graph that can be called (all implementations are callable for functional nodes
# and an implemented DriverContract for conditional nodes)

def check_function_callable[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                                ) -> Iterator[Violation]:
    '''
    Callable function for functional nodes
    '''
    if isinstance(node, FunctionalNode) and is_implemented(node) \
            and not callable(node.implementation):
        yield NotCallable(name, node.implementation)


def check_implements_its_driver[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                                    ) -> Iterator[Violation]:
    '''
    Check:
    - is the implementation a `DriverImplementation`
    - is the node implementing the driver contract?
    - Is the implementation capable of taking and reporting the contract's (chosen) input
    data and reports?
    '''
    if isinstance(node, DrivenNode) and is_implemented(node):
        implementation = node.implementation
        if not isinstance(implementation, DriverImplementation):
            yield NotADriverImplementation(name, implementation)
        elif not isinstance(node.contract, implementation.implements):
            yield ImplementsAnother(name, type(node.contract).__name__,
                                    implementation.implements.__name__)
        else:
            if missing := tuple(r for r in implementation.needs if r not in node.contract.takes):
                yield DataMissing(name, type(implementation).__name__, _names(missing))
            if absent := tuple(r for r in node.contract.reports if r not in implementation.provides):
                yield ReportUnavailable(name, type(implementation).__name__, _names(absent))


def _names(roles: tuple[type, ...]) -> tuple[str, ...]:
    return tuple(role.__name__ for role in roles)


class CallableGraph[N: Label, V: Place](RealisedGraph[N, V]):
    '''A realised graph whose every implementation is callable / driver implemented'''

    checks_node = (check_function_callable, check_implements_its_driver)



# ============================================================================ running it
# Running a CallableGraph with a value at each of its inputs.

def check_callable(valued: GraphWithValues) -> Iterator[Violation]:
    if not isinstance(valued.graph, CallableGraph):
        yield ProvedBelow(type(valued.graph).__name__)


@checked(check_callable)
def run[N: Label, V: Place](valued: GraphWithValues[CallableGraph[N, V], V]
                            ) -> OrderedMap[V, Any]:
    '''Every place with a value: the inputs, and everything the graph computes.'''
    env = Env(valued.values)
    walk(valued.graph, env, _solve)
    return OrderedMap.trusted(tuple(env.values), env.values.values())


# ------------------------------------------------ internal walking

class Env[V]:
    '''The values so far during a walk, by place.'''

    __slots__ = ('values',)

    def __init__(self, values: Iterable[tuple[V, Any]] | Mapping[V, Any] = ()) -> None:
        self.values : dict[V, Any] = dict(values)

    def read(self, places: Iterable[V]) -> tuple[Any, ...]:
        return tuple(self.values[p] for p in places)

    def write(self, places: tuple[V, ...], values: Iterable[Any]) -> None:
        self.values.update(zip(places, values, strict=True))

type AtSolve = Callable[[DrivenNode, SteppedExecutableGraph, Env], None]
'''What a walk does at a `Solve`, handed the problem, its interior and the values so far:
it writes the unknowns and the reports, and walks the interior if it wants its values.'''


def walk(graph: SteppedExecutableGraph, env: Env, at_solve: AtSolve) -> None:
    '''`graph`'s steps in order, a `Run` called and a `Solve` handed to `at_solve`: `run` is
    this loop with "solve it" at a solve. What a step raises names its node (`blamed`).'''
    for step in graph.steps:
        name = step.node if isinstance(step, Run) else step.problem
        try:
            if isinstance(step, Run):
                _call(name, graph[name], env)
            else:
                at_solve(graph[name], step.interior, env)
        except Exception as e:
            _blame(e, name)
            raise


def blamed(e: BaseException) -> Any:
    '''The node whose step raised `e`, the innermost; `None` if no step did.'''
    return getattr(e, '_cottax_node', None)


def _blame(e: Exception, name: Any) -> None:
    if blamed(e) is None:
        try:
            e._cottax_node = name
        except AttributeError:                    # an exception refusing attributes: the note alone
            pass
        e.add_note(f'raised at node {name!r}')


def _call(name: Any, node: Any, env: Env) -> None:
    '''A body returns the value at its one owned place, or one value per place.'''
    result = node.implementation(*env.read(node.reads))
    if len(node.owns) > 1:
        try:
            got = f'{len(result)} values'
        except TypeError:
            got = f'one {type(result).__name__}'
        if got != f'{len(node.owns)} values':
            raise ReturnedUnlike(name, len(node.owns), node.owns, got).raised
    env.write(node.owns, (result,) if len(node.owns) == 1 else result)


def _solve(node: ImplementedDriven, interior: CallableGraph, env: Env) -> None:
    conditions = InteriorConditionMap.posing(node, interior)

    # Solve: posed what the values mean and what is asked back, nothing of the graph
    values, reports = node.implementation(
        Posed(node.condition, node.contract.reports, conditions, conditions.context_places),
        env.read(conditions.context_places),
        dict(zip(node.contract.takes, map(env.read, node.data))))

    # Add unknowns and reports to data
    env.write(node.unknowns, values)
    for role, slot in zip(node.contract.reports, node.reports):
        env.write(slot, reports[role])
    # Walk interior again for consistent values in non-unknowns.
    walk(interior, env, _solve)


class InteriorConditionMap[N: Label, V: Place](ConditionMap):
    '''A condition map by one walk of a solve's interior. It holds a graph and places, so
    it hashes and compares by value.'''

    interior       : CallableGraph[N, V]
    unknowns       : tuple[V, ...]
    '''Where the unknowns' values go.'''
    context_places : tuple[V, ...]
    '''What the interior reads from outside, the unknowns aside: where `context` goes,
    in order. Fixed while the solve iterates.'''
    conditions     : tuple[V, ...]
    '''Where the answer is read: the condition's reads, its conditions then its parameters.'''

    @classmethod
    def posing(cls, node: DrivenNode[V], interior: CallableGraph[N, V]
               ) -> 'InteriorConditionMap[N, V]':
        '''The map `run` hands `node`'s driver, `interior` its `Solve`'s. Fixed: what the
        interior reads from outside, and what the condition reads that the interior does not
        produce (a parameter); the map answers at every condition read.'''
        solved   = set(node.unknowns)
        produced = set(interior.graph.graph.owned_variables)
        fixed    = tuple(dict.fromkeys(
            p for p in (*interior.graph.graph.boundary_inputs, *node.condition.reads)
            if p not in solved and p not in produced))
        return cls(interior, node.unknowns, fixed, node.condition.reads)

    def __call__(self, unknowns: Unknowns, context: Context) -> ConditionValues:
        env = Env(zip(self.context_places, context, strict=True))
        env.write(self.unknowns, unknowns)
        walk(self.interior, env, _solve)
        return env.read(self.conditions)
