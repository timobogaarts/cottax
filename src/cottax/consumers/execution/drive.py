'''`Drive`: every undriven problem driven, and implemented, by the first implementation accepting
it; a driven one implemented by the first implementing its contract.'''

from collections.abc import Callable

from ...core import Graph, ComposedOp, GraphOp
from ...core.frozen import OrderedMap
from ...core.conditional import Assign, AssignByRule
from ...core import conditions, is_driven, kind_of
from ...core.conditional import DriverContract, Role
from ...core.realisation import Implement, is_implemented
from .interface import DriverImplementation
from .refusals import NoDefaultContract, NoneAccepts, NoneImplements


class Drive(ComposedOp):
    '''Every problem no driver answers yet, driven by the first implementation whose
    contract accepts its shape, under the first of `contracts` it implements, at the places
    the rule `places` gives each role (as `AssignByRule` takes it), each role of `asks` wired
    where that contract offers it, and implemented by it. The places are recorded: an
    `Assign` each. A problem already driven and not implemented is implemented
    by the first implementation whose `implements` covers its contract; one already
    implemented is left alone.'''

    implementations : tuple[DriverImplementation, ...]
    contracts       : tuple[DriverContract, ...]
    places          : Callable[..., tuple]
    asks            : tuple[type[Role], ...] = ()

    def composed(self, graph: Graph) -> tuple[GraphOp, ...]:
        undriven = {p for p in conditions(graph.definitions) if not is_driven(graph[p])}
        picked = OrderedMap((p, self.chosen(p, node) if p in undriven
                             else self.covering(p, node.driver.contract))
                            for p in conditions(graph.definitions)
                            if not is_implemented(node := graph[p]))
        def assigned(p, implementation: DriverImplementation) -> Assign:
            contract = self.contract_for(implementation)
            offered = (*contract.may_report, *contract.may_be_given)
            asks = tuple(r for r in self.asks if r in offered)
            (assign,) = AssignByRule(p, contract, self.places, asks).composed(graph)
            return assign
        return (*(assigned(p, i) for p, i in picked.items() if p in undriven),
                *((Implement(picked),) if picked else ()))

    def chosen(self, name, problem) -> DriverImplementation:
        '''The first implementation whose contract accepts the problem's condition.'''
        for implementation in self.implementations:
            if implementation.implements.accepts(kind_of(problem)):
                return implementation
        raise NoneAccepts(name, problem).raised

    def covering(self, name, contract: DriverContract) -> DriverImplementation:
        '''The first implementation of `contract`, a problem's already.'''
        for implementation in self.implementations:
            if isinstance(contract, implementation.implements):
                return implementation
        raise NoneImplements(name, type(contract).__name__).raised

    def contract_for(self, implementation: DriverImplementation) -> DriverContract:
        '''The default contract an implementation answers: the first it implements.'''
        for contract in self.contracts:
            if isinstance(contract, implementation.implements):
                return contract
        raise NoDefaultContract(implementation, implementation.implements.__name__).raised


__all__ = ['Drive']
