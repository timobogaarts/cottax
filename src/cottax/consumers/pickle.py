'''A graph written as a pickle: every node pickles and refers to nothing in `__main__`.'''

import io
import pickle
import types
from collections.abc import Iterator

from ..core import Label, NestingGraph, Place, Violation
from ..core.frozen import checked


class _Scan(pickle.Pickler):
    '''Records every class and function the pickle refers to by name.'''

    def __init__(self) -> None:
        super().__init__(io.BytesIO())
        self.modules : set[str] = set()

    def reducer_override(self, obj: object) -> object:
        if isinstance(obj, (type, types.FunctionType)):
            self.modules.add(obj.__module__)
        return NotImplemented


def check_nodes_pickle[N: Label, V: Place](graph: NestingGraph[N, V]) -> Iterator[Violation]:
    '''Each node pickles, and names nothing in `__main__`'''
    for name in graph.nodes:
        scan = _Scan()
        try:
            scan.dump(graph[name])
        except Exception as e:
            yield Unpicklable(name, f'{type(e).__name__}: {e}')
            continue
        if '__main__' in scan.modules:
            yield DefinedInMain(name)


@checked(check_nodes_pickle)
def write_pickle[N: Label, V: Place](graph: NestingGraph[N, V]) -> bytes:
    '''A graph whose every node another process can unpickle, pickled; `pickle.loads` gives
    it back.'''
    return pickle.dumps(graph)


# ----------------------------------------------------------------- violation finals

class Unpicklable(Violation):
    node   : object
    reason : str
    error    = TypeError
    template = ('{node!r} does not pickle ({reason}): a lambda or a local function has no '
                'name to be found by; bind it at module level')


class DefinedInMain(Violation):
    node : object
    error    = TypeError
    template = ('{node!r} holds something defined in `__main__`, which loads back only in the '
                'process that wrote it; move it into a module')
