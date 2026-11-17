'''A graph and a value at exactly each of its boundary inputs: what a run is handed. The
execution's, not the maths': a value at a place is a run's input, not the graph's.'''

from collections.abc import Iterator
from typing import Any

from ...core import Graph, Place, dependencies
from ...core.frozen import Frozen, OrderedMap, Violation
from .refusals import InputMissing, NotAPlace, OwnedGiven


def check_every_input_given(g: 'GraphWithValues') -> Iterator[Violation]:
    if missing := tuple(p for p in dependencies(g.graph).boundary_inputs if p not in g.values):
        yield InputMissing(missing)


def check_nothing_owned_given(g: 'GraphWithValues') -> Iterator[Violation]:
    owned = set(g.graph.owned_variables)
    if given := tuple(p for p in g.values if p in owned):
        yield OwnedGiven(given)


def check_only_inputs_given(g: 'GraphWithValues') -> Iterator[Violation]:
    known = set(g.graph.variables)
    if extra := tuple(p for p in g.values if p not in known):
        yield NotAPlace(extra)


class GraphWithValues[G: Graph, V: Place](Frozen):
    '''
    A graph and a value at exactly each of its boundary inputs. Its checks read the graph's
    structure alone, so any proof holds; a consumer says which it takes (`proved`).
    '''

    graph  : G
    values : OrderedMap[V, Any]

    checks = (check_every_input_given, check_nothing_owned_given, check_only_inputs_given)
