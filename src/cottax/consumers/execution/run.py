'''
What can run (`Runnable`), and running it (`run`).
'''

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from ...core import (
    ConditionalNode, FunctionalNode,  Label, Graph, NodeDefinition,
    Place, per_node_check, Rules, Run, Schedule, Violation, is_driven)
from ...core.frozen import OrderedMap, checked
from ...core.conditional import DriverContract, Role
from ...core.realisation import Realised, implementation_of, is_implemented
from .interface import DriverImplementation
from .interface_types import ConditionMap, ConditionValues, Context, Posed, Unknowns
from .refusals import (
    DataMissing, DataUnused, ImplementsAnother, NotADriverImplementation, NotCallable,
    ReportUnavailable, ReturnedUnlike,
)
from .valued import GraphWithValues




# ============================================================================ what can run
# Realised Graph that can be called (all implementations are callable for functional nodes
# and an implemented DriverContract for conditional nodes)

@per_node_check
def check_function_callable[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                                ) -> Iterator[Violation]:
    '''
    Callable function for functional nodes
    '''
    if isinstance(node, FunctionalNode) and is_implemented(node) \
            and not callable(implementation_of(node)):
        yield NotCallable(name, implementation_of(node))


@per_node_check
def check_implements_its_driver[N: Label, V: Place](name: N, node: NodeDefinition[V]
                                                    ) -> Iterator[Violation]:
    '''
    Check:
    - is the implementation a `DriverImplementation`
    - is the node implementing the driver contract?
    - Is the implementation capable of taking and reporting what the driver is given and
    asked for? Data the contract requires an implementation may leave unread (a solver
    starting itself); optional data was asked for, so one it does not use is refused.
    '''
    if is_driven(node) and is_implemented(node):
        implementation, driver = implementation_of(node), node.driver
        contract = driver.contract
        if not isinstance(implementation, DriverImplementation):
            yield NotADriverImplementation(name, implementation)
        elif not isinstance(contract, implementation.implements):
            yield ImplementsAnother(name, type(contract).__name__,
                                    implementation.implements.__name__)
        else:
            yield from serves(name, implementation, contract, tuple(driver.data), tuple(driver.reports))


def serves[N: Label](name: N, implementation: DriverImplementation, contract: DriverContract,
                     data: tuple[type[Role], ...], reports: tuple[type[Role], ...]
                     ) -> Iterator[Violation]:
    '''Whether `implementation` can run a driver of `contract` holding `data` and asked for
    `reports`: it is given what it needs, uses each datum beyond what the contract requires
    for those reports, and can make each report. Asked of a driven node, and by `Drive` of the
    driver it is about to place.'''
    kind = type(implementation).__name__
    if missing := tuple(r for r in implementation.needs if r not in data):
        yield DataMissing(name, kind, _names(missing))
    required, _ = contract.roles_for(reports)
    if unused := tuple(r for r in data if r not in required
                       and r not in implementation.needs | implementation.uses):
        yield DataUnused(name, kind, _names(unused))
    if absent := tuple(r for r in reports if r not in implementation.provides):
        yield ReportUnavailable(name, kind, _names(absent))


def _names(roles: tuple[type, ...]) -> tuple[str, ...]:
    return tuple(role.__name__ for role in roles)


class RunnableNodes(Rules):
    '''Every function's implementation callable, every driver's implementing its contract.'''
    rules = (check_function_callable, check_implements_its_driver)


class Runnable(Realised, RunnableNodes):
    '''A realised graph whose every implementation is callable / driver implemented.'''



# ============================================================================ running it
# Running a Runnable with a value at each of its inputs.

@checked(Runnable.check)
def run[N: Label, V: Place](valued: GraphWithValues[Graph[N, V], V]
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

type AtEnclosed = Callable[[ConditionalNode, Graph, Env], None]
'''What a walk does at a `Drive`, handed the encloser, its interior and the values so far:
it writes the unknowns and the reports, and walks the interior if it wants its values.'''


def walk(graph: Graph, env: Env, at_enclosed: AtEnclosed) -> None:
    '''`graph`'s steps in order, a `Run` called and an `Enclosed` handed to `at_enclosed`: `run` is
    this loop with "solve it" at one. What a step raises names its node (`blamed`).'''
    for step in Schedule.of(graph).steps:
        name = step.node if isinstance(step, Run) else step.encloser
        try:
            if isinstance(step, Run):
                _call(name, graph[name], env)
            else:
                at_enclosed(graph[name], step.interior.graph, env)
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
    result = implementation_of(node)(*env.read(node.reads))
    if len(node.owns) > 1:
        try:
            got = f'{len(result)} values'
        except TypeError:
            got = f'one {type(result).__name__}'
        if got != f'{len(node.owns)} values':
            raise ReturnedUnlike(name, len(node.owns), node.owns, got).raised
    env.write(node.owns, (result,) if len(node.owns) == 1 else result)


def _solve(node: ConditionalNode, interior: Graph, env: Env) -> None:
    conditions = InteriorConditionMap.posing(node, interior)
    driver = node.driver

    # Solve: posed what the values mean and what is asked back, nothing of the graph
    values, reports = implementation_of(node)(
        Posed(node.condition, driver.reports.keys(), conditions, conditions.context_places),
        env.read(conditions.context_places),
        {role: env.read(places) for role, places in driver.data.items()})

    # Add unknowns and reports to data
    env.write(node.owns_inside, values)
    for role, places in driver.reports.items():
        env.write(places, reports[role])
    # Walk interior again for consistent values in non-unknowns.
    walk(interior, env, _solve)


class InteriorConditionMap[N: Label, V: Place](ConditionMap):
    '''A condition map by one walk of a solve's interior. It holds a graph and places, so
    it hashes and compares by value.'''

    interior       : Graph[N, V]
    unknowns       : tuple[V, ...]
    '''Where the unknowns' values go.'''
    context_places : tuple[V, ...]
    '''What the interior reads from outside, the unknowns aside: where `context` goes,
    in order. Fixed while the solve iterates.'''
    conditions     : tuple[V, ...]
    '''Where the answer is read: the condition's reads, its conditions then its parameters.'''

    @classmethod
    def posing(cls, node: ConditionalNode[V], interior: Graph[N, V]
               ) -> 'InteriorConditionMap[N, V]':
        '''The map `run` hands `node`'s driver, `interior` its `Enclosed`'s. Fixed: what the
        interior reads from outside, and what the condition reads that the interior does not
        produce (a parameter); the map answers at every condition read.'''
        solved   = set(node.owns_inside)
        stated   = node.condition.reads
        produced = set(interior.owned_variables)
        fixed    = tuple(dict.fromkeys(
            p for p in (*interior.boundary_inputs, *stated)
            if p not in solved and p not in produced))
        return cls(interior, node.owns_inside, fixed, stated)

    def __call__(self, unknowns: Unknowns, context: Context) -> ConditionValues:
        env = Env(zip(self.context_places, context, strict=True))
        env.write(self.unknowns, unknowns)
        walk(self.interior, env, _solve)
        return env.read(self.conditions)
