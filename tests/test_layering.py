'''
The layering, as rules over the tree (see `consumers_and_features.md`).

The static rules read each module's imports with `ast`: an import inside a function is not a
dependency. The runtime rules run once, in one interpreter forking a child per module, so
each module is imported first into a process holding no cottax (`python test_layering.py`).
'''

import ast
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import subprocess
import sys

import pytest


SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'cottax'


def _name(path: pathlib.Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


MODULES = {_name(p): p for p in sorted(SRC.rglob('*.py'))}
FEATURES = tuple(sorted(p.parent.parent.name for p in SRC.glob('*/consumers/__init__.py')))


def _entries(package: pathlib.Path) -> set[str]:
    '''A package's modules and subpackages, by name.'''
    return {p.stem for p in package.iterdir()
            if p.suffix == '.py' and p.stem != '__init__' or (p / '__init__.py').exists()}


CONSUMERS = tuple(sorted(_entries(SRC / 'consumers')))
DOORS = tuple(f'cottax.consumers.{c}' for c in CONSUMERS)
'''A feature is a top-level package with a `consumers/` of its own; a consumer is an entry
of `cottax/consumers/`, its front door the module or package itself.'''

SHARED = tuple(f'cottax.{f}.consumers.{e}' for f in FEATURES
               for e in sorted(_entries(SRC / f / 'consumers')) if e not in CONSUMERS)
'''Under a feature's `consumers/`, what is named after no consumer: support its answers share.'''


def _under(name: str, *prefixes: str) -> bool:
    return any(name == p or name.startswith(p + '.') for p in prefixes)


def _resolved(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ''
    base = module.split('.')[:len(module.split('.')) - node.level + is_package]
    return '.'.join([*base, node.module] if node.module else base)


@functools.cache
def _imports(module: str, lazy_too: bool = False) -> frozenset[str]:
    '''What `module` imports at import time (and inside its functions, if `lazy_too`), dotted,
    relative imports resolved.'''
    path = MODULES[module]
    tree = ast.parse(path.read_text())
    lazy = set() if lazy_too else {
        n for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
        for n in ast.walk(f)}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and node not in lazy:
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node not in lazy:
            found.add(_resolved(node, module, path.name == '__init__.py'))
    return frozenset(found)


def _offences(modules, allowed, lazy_too: bool = False) -> dict[str, set[str]]:
    '''Per module, each import `allowed(module, name)` refuses.'''
    return {m: bad for m in modules
            if (bad := {n for n in _imports(m, lazy_too) if not allowed(m, n)})}


def _is_answer_module(module: str) -> bool:
    return any(_under(module, f'cottax.{f}.consumers') for f in FEATURES)


# ================================================================ 1. core

MATHS = ('cottax.core.maths', 'cottax.core.frozen')
CORE = (('cottax.core.maths', MATHS), ('cottax.core.frozen', ('cottax.core.frozen',)),
        ('cottax.core.refusal_kinds', (*MATHS, 'cottax.core.refusal_kinds')),
        ('cottax.core.driven', (*MATHS, 'cottax.core.driven')),
        ('cottax.core.realisation', (*MATHS, 'cottax.core.driven', 'cottax.core.realisation')))
'''Each layer of core and what of cottax it may import; `refusals`, the maths' kinds, the maths.
`driven` and `realisation` may also take names from the front door `cottax.core`, which is
the maths alone.'''


def _core_allows(module: str) -> tuple[str, ...]:
    if module == 'cottax.core':
        return MATHS
    return next((a for package, a in CORE if _under(module, package)), ())


def _core_import(module: str, name: str) -> bool:
    if not name.startswith('cottax'):
        return name.split('.')[0] in (*sys.stdlib_module_names, 'networkx')
    front = name == 'cottax.core' and module != 'cottax.core' and not _under(module, *MATHS)
    return front or _under(name, *_core_allows(module))


def test_core_imports_its_layers_in_order_and_only_the_stdlib_and_networkx():
    '''Rule 1. `frozen` imports nothing else of cottax; the maths `frozen`; `driven` the maths;
    `realisation` the maths and `driven`; the front door the maths. Every module of core is in
    one of these layers, and imports nothing outside the standard library but networkx.'''
    core = [m for m in MODULES if _under(m, 'cottax.core')]
    assert [m for m in core if not _core_allows(m)] == []
    assert _offences(core, _core_import) == {}


def test_core_refusals_holds_every_kind_the_maths_defines():
    '''Rule 1. `core.refusal_kinds` is a study's door to the maths' refusal kinds: its `__all__` is
    every `Violation` defined under `core/maths/`, and each is the class defined there.'''
    from cottax.core import Violation
    import cottax.core.refusal_kinds as refusals
    defined = {n: v for m in MODULES if _under(m, 'cottax.core.maths')
               for n, v in vars(importlib.import_module(m)).items()
               if isinstance(v, type) and issubclass(v, Violation) and v.__module__ == m}
    assert sorted(refusals.__all__) == sorted(defined)
    assert {n: getattr(refusals, n) for n in refusals.__all__} == defined


def test_a_name_the_core_door_gives_is_imported_from_it():
    '''Rule 1. Outside the maths, a name the front door `cottax.core` gives is imported from
    the door, never from the maths module defining it (a lazy import too); a name it does not
    give, a rule, is imported from its own module.'''
    import cottax.core as door
    given = {k for k in vars(door) if not k.startswith('_')}
    found = {}
    for m, path in MODULES.items():
        if m == 'cottax.core' or _under(m, 'cottax.core.maths'):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.ImportFrom)
                    and _under(_resolved(node, m, path.name == '__init__.py'), 'cottax.core.maths')
                    and (bad := {a.name for a in node.names} & given)):
                found.setdefault(m, set()).update(bad)
    assert found == {}


# ================================================================ 2. the features

DRIVEN = {'cottax.relational.contracts': 'the shipped contracts are `DriverContract`s',
          'cottax.ode.contracts':        'the integrator and its roles'}
'''The modules of a feature outside its `consumers/` that also import `core.driven`.'''


def _feature_import(module: str, name: str) -> bool:
    feature = module.split('.')[1]
    if _under(name, ASSEMBLY, *(f'cottax.{f}' for f in FEATURES if f != feature)):
        return False
    if _is_answer_module(module) or not name.startswith('cottax'):
        return True
    return (name == 'cottax.core' or _under(name, *MATHS)
            or _under(name, f'cottax.{feature}') and not _under(name, f'cottax.{feature}.consumers')
            or module in DRIVEN and _under(name, 'cottax.core.driven'))


def test_a_feature_imports_no_other_and_outside_its_consumers_the_maths_alone():
    '''Rule 2. No feature imports another, nor the assembly. Outside its `consumers/` it imports of
    cottax the maths and itself, not its own `consumers/` and no consumer; `DRIVEN` also
    `core.driven`.'''
    assert _offences([m for m in MODULES if _under(m, *(f'cottax.{f}' for f in FEATURES))],
                     _feature_import) == {}


# ================================================================ 3, 4. consumers and answers

def test_neither_a_consumer_nor_core_names_a_feature():
    '''Rule 3. No module of a consumer or of core imports a feature, inside a function either:
    a feature's answers register when the study imports its `consumers` (the assembly does).'''
    modules = [m for m in MODULES if _under(m, 'cottax.consumers', 'cottax.core')]
    feature = lambda m, n: not _under(n, *(f'cottax.{f}' for f in FEATURES))
    assert _offences(modules, feature, lazy_too=True) == {}


LAYERED = {'declaration': ('surface',)}
'''Each consumer layered on others: an answer to it meets those through their doors too. A
declaring part is a part of a surface.'''


def _answer_import(module: str, name: str) -> bool:
    _, feature, _, entry = module.split('.')[:4]
    others = (f'cottax.{feature}.consumers.{c}' for c in CONSUMERS if c != entry)
    met = (entry, *LAYERED.get(entry, ())) if entry in CONSUMERS else ()
    return (not _under(name, 'cottax.consumers', *others)
            or any(name in (f'cottax.consumers.{c}', *(f'cottax.consumers.{c}.{door}'
                            for door in ('interface', 'interface_types', 'refusals')))
                   for c in met))


def test_an_answer_meets_its_consumer_in_its_interface_alone():
    '''Rule 4. Under `<feature>/consumers/<c>`, a module imports of the consumers only `c`'s
    doors, and those of the consumers `c` is `LAYERED` on: its front door (the study's API),
    `interface`, `interface_types` and the kinds it refuses with, `refusals`; support the
    answers share (`SHARED`) imports no consumer. Neither imports an answer to another
    consumer.'''
    answers = [m for m in MODULES if _is_answer_module(m) and len(m.split('.')) > 3]
    assert _offences(answers, _answer_import) == {}


USED_THROUGH = ('interface', 'interface_types', 'refusals')
'''A consumer uses another through `USED_THROUGH`.'''


def _through_doors(module: str, name: str) -> bool:
    door = next((d for d in DOORS if _under(name, d)), None)
    own  = next(d for d in DOORS if _under(module, d))
    return (door is None or door == own
            or name in (f'{door}.{f}' for f in USED_THROUGH))


def test_a_consumer_uses_another_through_its_doors():
    '''Rule 4. A module of one consumer imports of another, inside a function too, only what
    it is used through (`USED_THROUGH`), never its front door, a study's.'''
    modules = [m for m in MODULES if _under(m, *DOORS)]
    assert _offences(modules, _through_doors, lazy_too=True) == {}


USES_NONE = ('cottax.consumers.surface',)
'''The consumers using no other: reading what a study writes into names is apart from what
declares, answers or runs a model, each a part of a surface.'''


def test_a_consumer_using_none_imports_no_other():
    '''Rule 4. A module of a consumer in `USES_NONE` imports, inside a function too, nothing
    of another consumer.'''
    modules = [m for m in MODULES if _under(m, *USES_NONE)]
    own = lambda m, n: _under(n, next(d for d in DOORS if _under(m, d))) or not any(
        _under(n, d) for d in DOORS)
    assert _offences(modules, own, lazy_too=True) == {}


# ================================================================ 5, 6. what a consumer asks

def test_every_consumer_package_says_what_it_asks():
    '''Rule 5, statically: every consumer that is a package has an `interface.py` (a module,
    `pickle.py`, asks nothing). Its classes are held to `Answer` or `Choice` at runtime.'''
    assert [c for c in CONSUMERS if (SRC / 'consumers' / c).is_dir()
            and not (SRC / 'consumers' / c / 'interface.py').is_file()] == []


def test_every_choice_says_how_it_is_handed_in():
    '''Rule 5. Every class directly under `Choice` states its slot in a `Handed in as:` block
    of its docstring: what a study writes to name one.'''
    choices = {(m, n.name): ast.get_docstring(n) or '' for m, p in MODULES.items()
               for n in ast.walk(ast.parse(p.read_text())) if isinstance(n, ast.ClassDef)
               if any(getattr(b, 'id', getattr(b, 'attr', None)) == 'Choice' for b in n.bases)}
    assert len(choices) >= 5
    assert [c for c, doc in choices.items() if 'Handed in as:' not in doc] == []


def test_a_features_answers_are_named_after_consumers_and_load_no_other():
    '''Rule 6. Under a feature's `consumers/`, every module and package has a consumer's name,
    or is support (`SHARED`) imported by the answers to at least two consumers; `consumers/`
    and each package directly under it import nothing outside themselves.'''
    users = lambda s: {m.split('.')[3] for m in MODULES if _is_answer_module(m)
                       and len(m.split('.')) > 3 and m.split('.')[3] in CONSUMERS
                       and any(_under(n, s) for n in _imports(m))}
    assert {s: u for s in SHARED if len(u := users(s)) < 2} == {}
    for f in FEATURES:
        entries = _entries(SRC / f / 'consumers')
        packages = [m for e in ('', *(f'.{e}' for e in entries))
                    if MODULES[m := f'cottax.{f}.consumers{e}'].name == '__init__.py']
        assert _offences(packages, lambda m, n: _under(n, m)) == {}


def _all(module: str) -> list[str] | None:
    '''A module's `__all__`, read statically.'''
    return next((ast.literal_eval(n.value) for n in ast.parse(MODULES[module].read_text()).body
                 if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == '__all__'
                                                      for t in n.targets)), None)


def test_an_answer_package_says_what_it_offers_and_an_interface_defines_no_function():
    '''Rule 6. A package under a feature's `consumers/` holding modules is one answer: its
    `__all__` is what the single file would define. An `interface.py` defines classes alone.'''
    answers = [m for m, p in MODULES.items() if _is_answer_module(m) and len(m.split('.')) > 3
               and p.name == '__init__.py' and any(q.suffix == '.py' and q.name != '__init__.py' for q in p.parent.iterdir())]
    assert [m for m in answers if _all(m) is None] == []
    interfaces = [m for m in MODULES if _under(m, 'cottax.consumers') and m.endswith('.interface')]
    assert {m: f for m in interfaces for n in ast.parse(MODULES[m].read_text()).body
            if isinstance(n, ast.FunctionDef) if (f := n.name)} == {}


# ================================================================ 7, 8. exports and the assembly

def _bound_from_outside(module: str) -> set[str]:
    '''The public names an `__init__` binds from outside its package, and a lazy
    `__getattr__`.'''
    bad = set()
    for node in ast.parse(MODULES[module].read_text()).body:
        if isinstance(node, ast.ImportFrom):
            source = _resolved(node, module, True)
            if not _under(source, module):
                bad |= {f'{source}.{n}' for a in node.names
                        if not (n := a.asname or a.name).startswith('_')}
        elif isinstance(node, ast.Import):
            bad |= {a.name for a in node.names if not _under(a.name, module)
                    and not (a.asname or a.name).startswith('_')}
        elif isinstance(node, ast.FunctionDef) and node.name == '__getattr__':
            bad.add('__getattr__')
    return bad


ASSEMBLY = 'cottax.assemblies.jax'
'''The assembly: the one module naming both features, what it defines itself pinned
(`ASSEMBLY_DEFINES`).'''

ASSEMBLY_DEFINES = {'Scope', 'ShippedSurface', '__getattr__', '_drawn', 'render_html',
                    'render_xdsm_svg', 'render_matrix_svg', 'draw', 'to_dot', 'render'}
'''Every class and function the assembly defines itself: the shipped surface and a scope of
it, `Diffrax` imported when asked for, and the drawings taking a `Declared`. Anything else it
holds is an import, or a value (`flat`, `absolute`, `CONTRACTS`, each shipped purpose).'''


def test_an_init_exports_only_what_its_package_defines():
    '''Rule 7, statically: no `__init__` but the assembly's binds a public name from another
    package, and `cottax/__init__` is its docstring. At runtime: what each holds was defined
    under it.'''
    assert {m: b for m, p in MODULES.items() if p.name == '__init__.py'
            if (b := _bound_from_outside(m))} == {}
    assert [type(n) for n in ast.parse(MODULES['cottax'].read_text()).body] == [ast.Expr]


def test_the_assembly_defines_only_what_is_pinned():
    '''Rule 8. The assembly is imports and a few values; the classes and functions it defines
    itself are `ASSEMBLY_DEFINES`, so logic cannot grow there silently.'''
    import cottax.assemblies.jax as cx
    assert {a for a, v in vars(cx).items()
            if callable(v) and getattr(v, '__module__', None) == ASSEMBLY} == ASSEMBLY_DEFINES


CONDITION_NODES = {'declare', 'requirement', 'satisfy', 'optimise', 'run', 'valued',
                   'copy', 'reading', 'parameter_reads', 'term', 'per_place'}
'''What a part offers beside the kinds it declares (`declares`) and the ops it builds: the
model declared, relational's condition nodes, a run on values written by name or held in a
caller's structure, where its cuts put a copy, and a declaring
part's conventions, changed or asked.'''


def test_a_part_writes_or_runs_the_model_and_never_draws_it():
    '''Rule 8. Every method a shipped `Part` offers is a kind `declares` made, one of
    `CONDITION_NODES`, or one returning an op: a surface writes the model and runs it on values
    written by name, and what draws or exports one takes the finished model.'''
    import typing
    import cottax.assemblies.jax  # noqa: F401  the shipped parts
    import cottax.ode.consumers.declaration  # noqa: F401
    from cottax.core import GraphOp
    from cottax.consumers.surface.interface import Part
    parts, todo = [], [Part]
    while todo:
        parts.append(cls := todo.pop())
        todo.extend(cls.__subclasses__())
    shipped = [c for c in parts if _under(c.__module__, 'cottax') and c is not Part]
    assert len(shipped) >= 6

    def writes(method) -> bool:
        if getattr(method, '__declares__', None) is not None:
            return True
        returned = typing.get_type_hints(method).get('return')
        return isinstance(returned, type) and issubclass(returned, GraphOp)

    assert {c.__qualname__: bad for c in shipped
            if (bad := {w for w, v in vars(c).items() if not w.startswith('_')
                        and w not in {f.name for f in dataclasses.fields(c)}
                        and (inspect.isfunction(v) or hasattr(v, '__declares__'))
                        and not writes(getattr(c, w))}
                - CONDITION_NODES)} == {}


def test_nothing_imports_the_assembly():
    '''Rule 8. A study imports the assembly; nothing in the package does, inside a function
    either.'''
    assert _offences(MODULES, lambda m, n: not _under(n, ASSEMBLY), lazy_too=True) == {}


# ================================================================ 9. numerics

RUNS = tuple(f'cottax.{f}.consumers.execution' for f in FEATURES)
'''What runs numerics: a feature's answer to the run.'''

NUMERICS = {
    'jax':           (*RUNS, *SHARED, 'cottax.pytree.consumers.declaration',
                      'cottax.consumers.execution.jax_jit'),
    'optimistix':    RUNS,
    'diffrax':       RUNS,
    'scipy':         RUNS,
    'jax.tree_util': (*RUNS, *SHARED, 'cottax.pytree', 'cottax.consumers.execution'),
}
'''Where each library may be imported. jax also by the support a feature's answers share
(`SHARED`) and by a declared model crossing a
trace (`traced`, each given value an array) and the abstract walk (`jax_jit.Traced`);
`jax.tree_util`, jax as a pytree library, also
by the pytree place and the run (the jit proof).'''


def _numerics_import(module: str, name: str) -> bool:
    library = 'jax.tree_util' if _under(name, 'jax.tree_util') else name.split('.')[0]
    return _under(module, *NUMERICS.get(library, ('cottax',)))


def test_numerics_are_imported_only_where_something_runs():
    '''Rule 9. jax beyond `jax.tree_util` only in `RUNS`, `SHARED` and `traced`, optimistix, diffrax
    and scipy only in `RUNS`.'''
    assert _offences(MODULES, _numerics_import) == {}


# ================================================================ 10. at runtime

EXTRAS = ('diffrax',)
'''The extras a module may need: one importing a missing one is skipped, not refused.'''


def _dotted(obj) -> str:
    return f'{obj.__module__}.{obj.__qualname__}'


def _table(interfaces=None) -> dict:
    from cottax.core.frozen import answered
    return {_dotted(i): {_dotted(c): _dotted(type(a)) for c, a in t.items()}
            for i, t in answered().items() if interfaces is None or i in interfaces}


ANSWERS = tuple(f'cottax.{f}.consumers' for f in FEATURES)
'''What registers every answer the shipped features give: their `consumers`.'''


def _first(module: str) -> tuple[str, dict]:
    '''In a process holding no cottax: import `module`, say what loaded and what a feature
    answers, then every front door, the assembly and `ANSWERS`, and what is answered.'''
    found = {}
    try:
        importlib.import_module(module)
        found = {'loaded': sorted(m for m in sys.modules if _under(m, 'cottax')),
                 'by_a_feature': [k for t in _table().values() for k in t.values()
                                  if _under(k, *(f'cottax.{f}' for f in FEATURES))]}
        for door in (*DOORS, ASSEMBLY, *ANSWERS):
            importlib.import_module(door)
        found['table'] = _table()
    except Exception as e:
        skipped = isinstance(e, ImportError) and getattr(e, 'name', '') in EXTRAS
        found = {'skipped': True} if skipped else {**found, 'error': f'{type(e).__name__}: {e}'}
    return module, found


def _marked_choice(module: str, name: str) -> bool:
    '''Whether the class `name` is defined in `module` under a `@choice(...)`, read statically:
    the mark leaves no trace on the class.'''
    return any(isinstance(n, ast.ClassDef) and n.name == name
               and any(isinstance(d, ast.Call) and getattr(d.func, 'id', '') == 'choice'
                       for d in n.decorator_list)
               for n in ast.parse(MODULES[module].read_text()).body)


def _refusals_home(module: str) -> str:
    '''Where a consumer's refusal kinds are defined: its `refusals.py`, or a consumer that is
    one module (`pickle.py`) itself.'''
    door = next(d for d in DOORS if _under(module, d))
    return f'{door}.refusals' if MODULES[door].name == '__init__.py' else door


def _runtime() -> dict:
    '''Each module first, in a child forked from a process holding the third-party libraries
    and no cottax; then, here, what holds of everything imported.'''
    import multiprocessing
    for library in ('jax', 'jax.numpy', 'networkx', 'numpy', 'optimistix', 'scipy.optimize',
                    'diffrax'):
        if importlib.util.find_spec(library.split('.')[0]):
            importlib.import_module(library)
    assert not any(_under(m, 'cottax') for m in sys.modules)
    with multiprocessing.get_context('fork').Pool(os.cpu_count(), maxtasksperchild=1) as pool:
        first = dict(pool.map(_first, MODULES, chunksize=1))

    report = {'first': first, 'exported': {}, 'unasked': {}, 'unlisted': {}, 'for_a_study': {},
              'refusals_elsewhere': {}, 'answer_and_choice': {}, 'unmarked': {}}
    if any('error' in r for r in first.values()):
        return report
    from cottax.core import Violation
    from cottax.core.frozen import Answer, Choice
    asked = {v for m in MODULES if m.endswith('.interface') and _under(m, 'cottax.consumers')
             if 'skipped' not in first[m] for v in vars(importlib.import_module(m)).values()
             if isinstance(v, type) and issubclass(v, (Answer, Choice))}
    for name in (m for m in MODULES if 'skipped' not in first[m]):
        held = vars(module := importlib.import_module(name))
        if MODULES[name].name == '__init__.py':
            report['exported'][name] = [a for a, v in held.items() if not a.startswith('_')
                                        and (inspect.isclass(v) or inspect.isfunction(v))
                                        and not _under(v.__module__, name)]
        if name.endswith('.interface') and _under(name, 'cottax.consumers'):
            report['unasked'][name] = sorted(
                {a for a, v in held.items() if isinstance(v, type) and v.__module__ == name
                 and not issubclass(v, (Answer, Choice))})
        report['unlisted'][name] = [a for a in getattr(module, '__all__', ()) if a not in held]
        defined = {a: v for a, v in held.items() if isinstance(v, type) and v.__module__ == name}
        report['answer_and_choice'][name] = [a for a, v in defined.items()
                                             if issubclass(v, Answer) and issubclass(v, Choice)]
        report['unmarked'][name] = [a for a, v in defined.items()
                                    if issubclass(v, Choice) and v is not Choice
                                    and Choice not in v.__bases__ and not inspect.isabstract(v)
                                    and not _marked_choice(name, a)]
        if _under(name, 'cottax.consumers') and name != 'cottax.consumers':
            report['refusals_elsewhere'][name] = [a for a, v in defined.items()
                                                  if issubclass(v, Violation)
                                                  and name != _refusals_home(name)]
        if name in DOORS:
            report['for_a_study'][name] = [a for a in getattr(module, '__all__', ())
                                           if isinstance(v := held.get(a), type)
                                           and (v in asked or issubclass(v, Violation))]
    return {k: {m: v for m, v in d.items() if v} if k != 'first' else d
            for k, d in report.items()}


@pytest.fixture(scope='module')
def runtime() -> dict:
    '''`_runtime`'s report. One thread per BLAS, so the parent forks single-threaded.'''
    single = {v: '1' for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    out = subprocess.run([sys.executable, __file__], capture_output=True, text=True,
                         env={**os.environ, **single, 'PYTHONPATH': str(SRC.parent)})
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.splitlines()[-1])


@pytest.mark.slow
def test_every_module_imports_first_and_the_answers_register_alike(runtime):
    '''Rule 10. Every module imports first; whichever came first, importing every front door
    and `ANSWERS` afterwards registers the same answers, and a front door alone none of a
    feature's.'''
    first = runtime['first']
    assert {m: r['error'] for m, r in first.items() if 'error' in r} == {}
    tables = [r['table'] for r in first.values() if 'table' in r]
    assert tables[0] and all(t == tables[0] for t in tables)
    assert {m: r['by_a_feature'] for m, r in first.items() if m in DOORS
            if r.get('by_a_feature')} == {}


@pytest.mark.slow
def test_cottax_loads_nothing_and_a_feature_or_core_no_consumer(runtime):
    '''Rule 10. `import cottax` loads no other module; a module of core loads only what its
    layer, or the front door `cottax.core` imported before it, may import; a module of a
    feature outside its `consumers/` loads no consumer, no answer and not the assembly.'''
    first = runtime['first']
    assert first['cottax']['loaded'] == ['cottax']
    core = {m: bad for m, r in first.items() if _under(m, 'cottax.core')
            if (bad := [n for n in r.get('loaded', ()) if n not in ('cottax', 'cottax.core')
                        and not _core_import(m, n) and not _core_import('cottax.core', n)])}
    assert core == {}
    consumer = ('cottax.consumers', ASSEMBLY, *(f'cottax.{f}.consumers' for f in FEATURES))
    features = {m: bad for m, r in first.items()
                if _under(m, *(f'cottax.{f}' for f in FEATURES)) and not _is_answer_module(m)
                if (bad := [n for n in r.get('loaded', ()) if _under(n, *consumer)])}
    assert features == {}


@pytest.mark.slow
def test_what_is_imported_holds_what_it_says(runtime):
    '''Rules 5 and 7, at runtime: every class an `interface.py` defines is an `Answer` or a
    `Choice`; an `__init__` holds no class or function defined outside its
    package; every name in an `__all__` is there.'''
    assert runtime['unasked'] == {}
    assert runtime['exported'] == {}
    assert runtime['unlisted'] == {}


@pytest.mark.slow
def test_a_consumers_refusals_are_in_its_refusals_and_nothing_answers_and_is_chosen(runtime):
    '''Rules 4 and 5, at runtime: every `Violation` a consumer defines is in its
    `refusals.py` (`pickle.py`, one module, in itself); no class is both an `Answer` and a
    `Choice`.'''
    assert runtime['refusals_elsewhere'] == {}
    assert runtime['answer_and_choice'] == {}


@pytest.mark.slow
def test_every_shipped_choice_is_marked(runtime):
    '''Every final of a `Choice` interface defined under `cottax/` carries `@choice(...)`, so
    what a study may name is found by searching, as an answer is by `@answers`.'''
    assert runtime['unmarked'] == {}


@pytest.mark.slow
def test_a_front_door_exports_for_a_study(runtime):
    '''Rule 4, at runtime: a consumer's `__all__` holds no class its `interface.py` asks for
    (an `Answer` or `Choice` level, imported from there) and no refusal kind (`refusals`).'''
    assert runtime['for_a_study'] == {}


if __name__ == '__main__':
    print(json.dumps(_runtime()))
