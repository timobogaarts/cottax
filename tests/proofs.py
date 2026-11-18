'''What the tests read of a graph's proofs without asking: what proving left remembered.'''

from cottax.core.maths.rules import _satisfies


def recalled(graph, group) -> bool:
    '''Whether `graph` is remembered to satisfy `group` -- read, never asked: False if it
    fails or was never asked.'''
    return graph.__dict__.get('_remembered', {}).get((_satisfies, group), False)
