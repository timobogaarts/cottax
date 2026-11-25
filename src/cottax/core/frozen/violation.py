"""
Violations of the checks: supporting diagnostics as well as raising.
"""

import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import fields
from functools import wraps

from .base import AbstractClassVar, Frozen


class Violation(Frozen):
    """
    Violation: can raise or used to debug.
    """

    error    : AbstractClassVar[type[Exception]]
    """Exception raised for this violation"""
    template : AbstractClassVar[str]
    """Wording: e.g. 'output {place!r} is owned by both {other!r} and {node!r}'"""

    @property
    def where(self) -> dict[str, object]:
        """The fields, by role: places, node names, or tuples of either."""
        return {f.name: getattr(self, f.name) for f in fields(self)}   # type: ignore[arg-type]

    @property
    def message(self) -> str:
        return self.template.format(**self.where)

    @property
    def raised(self) -> Exception:
        '''The exception a caller expects for this violation, carrying it as `violation`, so a
        refusal caught is still data.'''
        exception = self.error(self.message)
        exception.violation = self
        return exception


def first(violations: Iterable[Violation]) -> None:
    '''Raise the first violation, if any -- the boundary between rules-as-data and the
    exceptions a constructor owes its caller.'''
    for violation in violations:
        raise violation.raised


def checked[F: Callable](*checks: Callable[..., Iterable[Violation]]) -> Callable[[F], F]:
    '''A function declaring its checks as a class does: each is called with the function's
    arguments as it receives them, and the first violation is raised before the body runs.
    The signature is read once, at decoration (`fn.signature`).'''
    def decorate(fn: F) -> F:
        signature = inspect.signature(fn)

        @wraps(fn)
        def run(*args, **kwargs):
            bound = _bound(signature, args, kwargs)
            for check in checks:
                first(check(*bound.args, **bound.kwargs))
            return fn(*bound.args, **bound.kwargs)
        run.checks, run.signature = checks, signature                   # type: ignore[attr-defined]
        return run                                                      # type: ignore[return-value]
    return decorate


def _bound(signature: inspect.Signature, args: tuple, kwargs: dict) -> inspect.BoundArguments:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def violations(cls: type[Frozen] | Callable, *args, **kwargs) -> Iterator[Violation]:
    '''
    Violatations from a class initialisation. Uses the DAG from the rules (lowest asked first, if 
    that raise it just only outputs those from the level)    
    '''
    if isinstance(cls, type):
        try:
            cls(*args, **kwargs)
        except Exception as e:
            if type(getattr(e, 'refused', None)) is not cls:
                raise
            yield from e.violations                                     # type: ignore[attr-defined]
        return
    bound = _bound(cls.signature, args, kwargs)                         # type: ignore[attr-defined]
    for check in cls.checks:                                            # type: ignore[attr-defined]
        yield from check(*bound.args, **bound.kwargs)


def name_of(implementation: object) -> str:
    '''A callable's name, or its type's, for a message.'''
    return getattr(implementation, '__name__', type(implementation).__name__)


def dotted(cls: type) -> str:
    '''A class's module and qualified name, for a message.'''
    return f'{cls.__module__}.{cls.__qualname__}'
