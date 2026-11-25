# cottax hand-off — 2026-11-19 (maths core done)

Never commit this file (it is tracked in git today: it went into `8d95f21`; whether to
untrack it is the owner's call). Review progress is tracked per module in the sheet
https://claude.ai/artifact/Qi6kWCKCEkRXBPs19uMRzu (tabs "Modules" and "Open issues"): a
Claude Sheet, read and written only through the Claude Docs connector (load its guide
`topic.sheets` first); container `{"kind":"project","id":"bfff3a33-8957-4d44-a09d-a01a993a16de"}`,
workbook node `a8872111-751b`. The owner crosses rows off there; read it before starting
and update its Status / Notes as work lands.

**How the owner works** (also in the session memory):
- When asked for an opinion or a question ("what do you think", "why"), answer only; do not
  edit. Build only on "do it" / an explicit request.
- Leave every change uncommitted for review; edit files in place (the owner edits too).
- Targeted test files only, throttled (section 9); never the full suite unless asked.
- processax is the owner's: report breaks with file:line, never edit (unless the owner
  explicitly asks for a conversion, as once on 2026-11-19).
- Check `git status` / `git log` first: the owner commits between sessions, and this file
  may lag the code.
The older published copy (https://claude.ai/artifact/G2NE5XNBiBxHXpqFcov9RN) has a
section 7 (operator kinds, partial evaluation, switches, repositories) not copied here.

## 1. Where things stand

- Branch **`tower`**, committed and pushed by the owner up to `8d95f21` ("core maths done").
  The whole maths core (`core/maths`) is reviewed. Uncommitted: the owner's own edit to
  `frozen/base.py` (the `remembered` docstring), and the deletion of
  `consumers_and_features.md` (leave it alone).
- `main` has PRs #7 (condition node's two faces), #8 (node attributes), #9 (driver as a field).
- **processax** (`~/processax`) is the owner's: never edit it, only report breaks. It was
  converted to the tower API once this session (uncommitted there, by an agent, at the
  owner's request). Since then these broke (owner to fix):
  - `benchmark/structure.py`: imports `Loop` (:9, now `Enclosed`); uses `step.interior` as a
    graph (:24, :40-41; now `step.interior.graph`, or walk `step.interior.steps`).
  - `Dependencies.of(..).owners` -> `.owner`: `process/study.py:112`,
    `process/graphs/kinds.py:109,118`, `preprocess/tests/test_coupled_reads.py:38,56,68`.
  - Untested by the conversion: `benchmark/capture.replay`, the edited example notebooks.

## 2. The maths core as it now is

One stored value, facts as views, rules as groups, edits as ops (`plan/`).

- **`Graph(definitions, within)`** (`graph.py`): checks nothing at construction. Literal facts
  true of any graph: `nodes`, `variables`, `owned_variables`, `read_variables`, `readers` and
  `owners` (place -> tuple of nodes), `boundary_inputs` / `boundary_outputs` (set
  differences), `subgraph` (a projection, `dataclasses.replace`; not an op).
- **`View`** (`graph.py`): a Frozen holding `graph`, `checks = (Group.check,)`, facts as
  `cached_property`s, got only as `View.of(graph)` (`@remembered _view`, one per class per
  graph; a test forbids constructing one directly in `src`). Anything depending on values
  is not a view. Views: `Dependencies` (asks `ValidDependencies`: `owner` single, edges,
  `components`, `cycles`, `is_acyclic` property, reachability, `feedback_variables` /
  `closing_readers` kept for `docs/source/theory/theory_6_i.ipynb`), `Nesting` (asks `Valid`:
  `outermost`, `enclosing`, `cycle_of`, `interior`; `_cycles` built once per graph),
  `Schedule` (asks `Executable`: `steps`, a tree of `Run(node) | Enclosed(encloser,
  interior: Schedule)`).
- **Rule groups** (`rules.py`): classes never instantiated; bases mean "implies".
  `X.check(graph)` yields violations (implied groups first; a group whose base fails is not
  asked), `X.prove(graph)` raises the first. Takes a graph or a value holding one
  (`HasGraph`: a view, `Traced`). `@checked(X.check)` only on one-argument functions.
  `per_node_check` lifts a `(name, node)` rule. Memo: pure `@remembered _satisfies(graph,
  group)`. Groups: `ValidNesting`, `ValidDependencies`, `Valid` (both), `Executable(Valid)`,
  then above maths `DrivenNodes`/`Driven`, `RealisedNodes`/`Realised`,
  `RunnableNodes`/`Runnable`, `JitKeyNodes`/`Jittable`.
- **Memo**: `frozen.remembered`, `fn(value, *key)` kept in the value's own `__dict__` (dies
  with it; copies and pickles start empty). Keys are kinds (a group, a view class), never
  varying inputs. `test_layering.REMEMBERED` pins every use: `_satisfies`, `_view`.
- **Vocabulary**: an *encloser* is a condition node owning something inside
  (`kinds.enclosers(definitions)`, `owns_inside(d)`). The maths says nothing of drivers or
  problems: what an encloser does with its interior (iterate, map, choose) is a layer up.
  `body(graph)` = the graph without condition nodes. `outline(graph)` prints
  `encloser[members]`.
- **Ops**: `GraphOp.apply(graph, keeps=Valid)` proves `keeps` of input and result; op-only
  `checks`, `(op, graph)` `checks_graph` collected by `refusals`. `Contract(blocks)` is a
  `ComposedOp` (was a `Nesting` method).
- Blind review (Fable, old maths vs new, unlabelled): old 5/10, new 8.5/10. It withdrew
  "the old type tower made invalid graphs unrepresentable" as a net virtue (that needs the
  wrapper tower, which does not compose). What a 9 needs is in section 3.

## 3. Next, in order

1. **Theorem tests** (small, from the review): a subgraph of a `Valid` graph is `Valid`; an
   interior of an `Executable` graph is `Executable`. Both are relied on (`Nesting._cycles`
   re-proves per level; `Schedule` proves each interior as its view's check) and would fail
   late, out of a cached property, if a future rule broke them.
2. **`frozen/` review: done 2026-11-25** (base, violation, answer, ordered_map all reviewed).
   - `unchecked`, `_carried`, `skip=` gone; a `Frozen`'s levels are asked as rule groups (the
     first refusing level raises its first violation, carrying `violations` and `refused`);
     `violations(cls, ...)` constructs and catches. `check_traces` guards on `Runnable.check`.
   - A check returns an iterable everywhere (no `or ()`). `checked` reads its signature once.
   - Pickle and copy rebuild through the constructor (`__reduce__` -> `_rebuilt`), so checked.
   - Subclass machinery: the generated-repr shadowing fixed (199 classes printed their parent's
     fields); a written `__eq__` is carried down and needs a `__hash__` beside it; `__init__`
     refused like `__post_init__`; a stringified `PerLevel` refused.
   - `answer.py`: registration refusals are plain `TypeError`s (`Unanswered` stays a
     `Violation`); a protocol `about` is checked by shape; `choices(interface)` registry;
     `checks_answer` hook kept.
   - Open: `FixedAndMoved` (relational) looks unreachable -- a moved parameter breaks a base
     level first (`ReadsOwn` / `ReadOnBothSides`); owner's call. Theorem tests (step 1):
     deferred until memo seeding.
3. **`plan/` round** (deferred by the owner, all from the review):
   - `refusals(op, graph)` can raise on an invalid graph: it proves nothing first, and
     `Cut`'s check builds a view. Option: yield `Valid`'s violations first and stop.
   - `Redefine.rebinding` raises `NotBound` via `first` instead of a `checks_graph` entry, so
     `refusals` never reports it; `Rewire`'s check does a raw `graph[name]` first.
   - `Contract.composed` reads the private `graph._position`.
   - `Plan.__add__` writes into `__dict__` (primes `trunk` / `graph`; else each `+` rebuilds
     the chain); `Plan.graph` recurses through `trunk` (deep unpickled plans).
   - `refusals` depends on derived-first ordering. (`None` returns: done 2026-11-24, a check
     returns an iterable everywhere; `or ()` dropped in base, violation, algebra.)
   - `Nest`'s check duplicates loop detection `Valid` proves on the result.
   - `Cut.copy_of` is a callable inside a record meant to hash and pickle.
   - Declined (do not reopen): `keeps` threaded through `Plan`.
4. **The rest of core**, by the sheet: `conditional/`, `mints.py`, `driven/`, `realisation/`,
   `answers.py`, `refusal_kinds.py`.
5. **Consumers**, then the domains (`relational`, `ode`, `pytree`) and `assemblies/jax.py`.
   Known there: architecture domain names still say problem (`TopNotAProblem`,
   `body_of_problem`, `Solve.problem` in the declaration mint) -- a separate naming
   decision; `visualization/sequencing.py` has its own `Run` / `Solve` / `Cycle` entries
   (a drawing's, distinct from the maths steps).
6. **Roadmap step 4: `run(graph, values)`**, replacing `GraphWithValues` (section 5).
7. **Roadmap step 3: tags to a convention**, plus a place-keyed table (section 6).
8. Later: `About` facts mixins composed in the assembly, when the first higher-layer view
   lands; a `UserFacingGraph` only if really needed.

## 4. Principles and decisions (do not reopen without cause)

- A node's class says what it is, its fields how it connects, its attributes everything
  else; no wrapper node classes. A driver changes edges, so it is a field.
- Values stay outside the graph; the graph defines the computation.
- Mechanism should shrink; violation classes are not counted as bulk. Checks that only
  restate a type annotation do not belong in core.
- Node checks stay on the node when a node breaking them is never a meaningful mid-plan
  state; relational and per-stage rules go to a group.
- A view holds only facts of the graph alone; facts true of any graph sit on `Graph`.
  No forwarding of view facts onto `Graph`; bind `deps = Dependencies.of(g)` locally.
- Proofs are not carried to subgraphs (each new value is proved when asked; cheap). Seeding
  the memo by a theorem is possible later, only after the theorem tests exist.
- A failing group runs its rules twice (accepted). networkx stays in `Dependencies`.
- `Rules.check`'s `graph if isinstance(graph, Graph) else graph.graph` stays: views'
  `checks` hand the check the view itself.
- Rejected: the `ProvedGraph` wrapper tower and any group-as-view; a metaclass making
  `Executable(g)` yield violations (truthy generator, mypy, `violations()`); caching
  violation lists; keyed `remembered` over node sets (`_level`); a separate `Cycles` view;
  type-level validity via `NewType`/`TypeIs` brands (do not compose; ops returning `Self`
  would forge the brand); flat storage checking eagerly; ops that stop checking with a lazy
  proof; `@requires` per query function; a `hereditary` flag; a global registry of
  contributed rules.
- Open idea (from the older design): rules a node kind contributes (`in_graph` on the
  condition class, read through `kind_of`); nothing needs it yet. A repository graph (several
  producers of one place before selection) is why `Graph` checks nothing at construction.

## 5. Step 4 mapped: `run(graph, values)`

- `run` is `@checked(Runnable.check)` over `run(valued: GraphWithValues)`; `Traced` holds a
  `GraphWithValues` and has a `graph` property for its `checks = (Runnable.check, ...)`.
- Target: `run(graph, values)` with the three value checks
  (`check_every_input_given`, `check_nothing_owned_given`, `check_only_inputs_given`) on the
  function, `Traced(graph, values)`; delete `GraphWithValues`. Then `@checked(Runnable.check)`
  on a two-argument function needs `checked` to hand a check only the first argument (today
  it hands all; `Rules.check` takes one).
- Open: where the value checks live besides `run` (working answer: `Traced` gets them
  through `run`). `walk(graph, env, at_enclosed)` hands an `Enclosed`'s interior graph to the
  driver; it could take the `Schedule` tree directly.
- processax uses `GraphWithValues` in `study.py`, `wiring/graph.py`, `benchmark/capture.py`,
  `vmcon/tests/test_vmcon_scaling.py`.

## 6. Step 3 mapped: tags to a convention

`Declared` is a set of side tables beside a plan (`given`, `place_tags`, `node_tags`; ops do
not touch them, stale entries tolerated). `Static` is the only tag, read only by the jit
registration; nothing reads `node_tags`; processax uses no tags.

Agreed: a convention in `pytree/consumers/declaration.py` -- floats, complex numbers and
arrays (and array leaves of any pytree value) are traced; ints, bools and everything else
static. No escape hatch for a static float; JAX-only. Together with it (owner's decision):
a place-keyed table on the graph for per-variable static data; shape undecided, no reader
yet. Verify that one place can hold a whole pytree (an `eqx.Module`) and add a test that
differentiates through one.

## 7. Driver contracts: open design

A contract is `accepts` + `takes` + `reports` (role classes); `data` / `reports` on the
node are the places, by position. Wiring (roles, shareable, compared on join) is sound.
Family identity is the weird part: ten classes in two tiers, boolean flags (`steps`,
`scaled`, `trajectory`), four overlapping matchers (`entry_for`, `Drive.covering`,
`contract_for`, `chosen`), roles stated three to five times, `needs` cannot say optional.

Discussed, not decided:

```python
Wiring(family=RootFinding, takes=(Start,), reports=(Steps,))     # shareable, no places
Driver(wiring, places={Start: (...), Steps: (...)})              # on the node
```

Flags and the level/final hierarchy go; `ImplementDrivers` folds into `Drive`. Questions
for the owner: should "which family answers this node" be in the backend-free graph? Is it
deliberate that Newton cannot run a node assigned as fixed-point iteration? Is asking for a
role per `drive` call, per node, or a surface default?

## 8. Small known leftovers

- `implemented(node, fn)` replaces an existing implementation silently outside a plan.
- Refusing an implementation on a bare condition no longer hints "takes a driver first".
- `Undrive` no longer refuses an implemented node (the rule is the realised layer's); the
  node it leaves is refused by `Realised`. Owner to confirm.

## 9. How to run things

- Never the full suite unless the owner asks that turn: only the touched test files,
  `cd ~/cottax && nice -n 19 taskset -c 0,1 ~/miniconda3/envs/fullcoupling/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`.
- processax: `JAX_PLATFORMS=cpu` (the GPU runs out of memory), same throttle, the
  fullcoupling python; targeted files only.
