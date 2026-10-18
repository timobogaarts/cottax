'''`Drive`: every undriven problem driven, and implemented, by the first implementation accepting
it; a driven one implemented by the first implementing its contract.'''

from collections.abc import Callable

from ...core import NestingGraph, ComposedOp, GraphOp
from ...core.frozen import OrderedMap
from ...core.driven import Assign, DriverContract, is_driven
from ...core.realisation import Implement, is_implemented
from .interface import DriverImplementation
from .refusals import NoDefaultContract, NoneAccepts, NoneImplements


class Drive(ComposedOp):
    '''Every problem no driver answers yet, driven by the first implementation whose
    contract accepts its shape, under the first of `contracts` it implements, at the places
    `places(at, unknowns, contract)` gives (its data and its reports, as `Assign` takes
    them), and implemented by it. A problem already driven and not implemented is implemented
    by the first implementation whose `implements` covers its contract; one already
    implemented is left alone.'''

    implementations : tuple[DriverImplementation, ...]
    contracts       : tuple[DriverContract, ...]
    places          : Callable[[object, tuple, DriverContract], tuple[tuple, tuple]]

    def composed(self, graph: NestingGraph) -> tuple[GraphOp, ...]:
        undriven = {p for p in graph.problems if not is_driven(graph[p])}
        picked = OrderedMap((p, self.chosen(p, node) if p in undriven
                             else self.covering(p, node.contract))
                            for p in graph.problems
                            if not is_implemented(node := graph[p]))
        def assigned(p, implementation: DriverImplementation) -> Assign:
            driver = self.contract_for(implementation)
            return Assign(p, driver, *self.places(p, graph[p].unknowns, driver))
        return (*(assigned(p, i) for p, i in picked.items() if p in undriven),
                *((Implement(picked),) if picked else ()))

    def chosen(self, name, problem) -> DriverImplementation:
        for implementation in self.implementations:
            if implementation.implements.accepts(problem):
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
