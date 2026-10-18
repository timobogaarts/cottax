'''A proved graph and a value at exactly each of its boundary inputs.'''

from collections.abc import Iterator
from typing import Any

from ..frozen import Frozen, OrderedMap, Violation
from .nesting import ProvedGraph
from .places import Place


def check_every_input_given(g: 'GraphWithValues') -> Iterator[Violation]:
    if missing := tuple(p for p in g.graph.graph.graph.boundary_inputs if p not in g.values):
        yield InputMissing(missing)


def check_nothing_owned_given(g: 'GraphWithValues') -> Iterator[Violation]:
    owned = set(g.graph.graph.graph.owned_variables)
    if given := tuple(p for p in g.values if p in owned):
        yield OwnedGiven(given)


def check_only_inputs_given(g: 'GraphWithValues') -> Iterator[Violation]:
    known = set(g.graph.graph.graph.variables)
    if extra := tuple(p for p in g.values if p not in known):
        yield NotAPlace(extra)


class GraphWithValues[G: ProvedGraph, V: Place](Frozen):
    '''
    A proved graph and a value at exactly each of its boundary inputs. Its checks read
    the graph's structure alone, so any proof holds; a consumer says which it takes.
    '''

    graph  : G
    values : OrderedMap[V, Any]

    checks = (check_every_input_given, check_nothing_owned_given, check_only_inputs_given)


class InputMissing(Violation):
    places : tuple
    error    = KeyError
    template = 'no value given at {places!r}, which the graph reads and nothing in it computes'


class OwnedGiven(Violation):
    places : tuple
    error    = ValueError
    template = ('a value given at {places!r}, which the graph computes: an owned place only '
                'ever means the value the run computes')


class NotAPlace(Violation):
    places : tuple
    error    = KeyError
    template = 'a value given at {places!r}, which the graph neither reads nor computes'
