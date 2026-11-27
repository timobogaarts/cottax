'''`Drive`: every undriven problem driven, and implemented, by the first implementation accepting
it; a driven one implemented by the first implementing its contract.'''

from collections.abc import Callable, Iterator

from ...core import Graph, Composite, GraphOp, Violation
from ...core.frozen import OrderedMap
from ...core.conditional.assign import assigned_by_rule
from ...core import conditions, is_driven, kind_of
from ...core.conditional import DriverContract, Role
from ...core.realisation import Implement, is_implemented
from .interface import DriverImplementation
from .refusals import NoDefaultContract, NoneAccepts, NoneImplements, NotOfferedAnywhere
from .run import serves


def check_asks_offered_somewhere(op: 'Drive', graph: Graph) -> Iterator[Violation]:
    '''Each role asked is an option of the contract of some problem it drives: one no problem
    offers would be asked for and placed nowhere.'''
    if wired := op.wired(graph):
        offered = {r for contract, _ in wired.values() for r in contract.options}
        if extra := tuple(r.__name__ for r in op.asks if r not in offered):
            yield NotOfferedAnywhere(repr(op), extra)


def check_implementations_serve(op: 'Drive', graph: Graph) -> Iterator[Violation]:
    '''Each implementation picked can run the driver it is picked for -- the one placed here,
    or the one already on the problem -- asked before the graph is touched.'''
    wired = op.wired(graph)
    for p, implementation in op.picked(graph).items():
        if p in wired:
            contract, asks = wired[p]
            data, reports = contract.roles_for(asks)
        else:
            driver = graph[p].driver
            contract, data, reports = driver.contract, tuple(driver.data), tuple(driver.reports)
        if isinstance(contract, implementation.implements):
            yield from serves(p, implementation, contract, data, reports)


class Drive(Composite):
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
    '''The options wired on each problem whose contract offers them; one no problem offers
    is refused.'''

    checks_graph = (check_asks_offered_somewhere, check_implementations_serve)

    def picked(self, graph: Graph) -> OrderedMap:
        '''Per problem not yet implemented, its implementation.'''
        return OrderedMap((p, self.covering(p, node.driver.contract) if is_driven(node)
                           else self.chosen(p, node))
                          for p in conditions(graph.definitions)
                          if not is_implemented(node := graph[p]))

    def wired(self, graph: Graph) -> OrderedMap:
        '''Per problem it drives, the contract it is placed under and the asks it offers.'''
        def under(contract: DriverContract) -> tuple:
            return contract, tuple(r for r in self.asks if r in contract.options)
        return OrderedMap((p, under(self.contract_for(i))) for p, i in self.picked(graph).items()
                          if not is_driven(graph[p]))

    def _ops(self, graph: Graph) -> tuple[GraphOp, ...]:
        picked = self.picked(graph)
        return (*(assigned_by_rule(graph, p, contract, self.places, asks)
                  for p, (contract, asks) in self.wired(graph).items()),
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
