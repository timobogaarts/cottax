# cottax hand-off — 2026-11-14 (end of day)

Untracked on purpose: never commit this file. It replaces the earlier `handoff.md`, which
no longer exists. The older published copy (https://claude.ai/artifact/G2NE5XNBiBxHXpqFcov9RN)
is superseded by this file, except for its section 7 (later work: operator kinds,
partial evaluation, switches, repositories), which is not copied here.

## 1. Where things stand

- **`main`** has PR #7 (`condition-node-two-faces`): the four names `owns_inside`,
  `reads_inside`, `reads_outside`, `owns_outside`; `encloser`; core `SetOwnsInside`.
- **`node-attributes`** is merged into `main` (PR #8, `c1c1e8a`). Suite at that commit:
  1173 passed, 32 deselected.
- **`driver-field`**: step 1 built, uncommitted, for review (section 5).
- **processax** needs no change for `node-attributes` (946 passed, 820 skipped against it).
  It still has seven files of uncommitted edits from the PR #7 rename
  (`owns_inside`, `is_encloser`, `.graph.enclosers`); those are the owner's to commit.
- Local-only branch `backup/two-faces-before-squash` can be deleted.
- **`tower`** (2026-11-17): pieces 1-3 and the maths cleanup built, uncommitted, for review
  (sections 5b-5e). Next session starts at **5f**.

## 2. What `node-attributes` built (as merged, not as first designed)

- `NodeDefinition.attributes : OrderedMap[type[Attribute], object]`. The key is an
  `Attribute` subclass that is never instantiated; the value is the raw thing:
  `{Implementation: fn}`.
- `Attribute.fits(value, node) -> bool` (default `True`) is a rule of the node: a node
  carrying something that does not fit cannot be built. `misfits(node, attributes)` is the
  one expression of it, used by the node check, by `SetAttribute` and by `Undrive`.
- `Redefine` carries a node's attributes through every edit unless the op sets
  `edits_attributes = True` (`Replace`, `SetAttribute`, `RemoveAttribute`). `Replace`
  hands over the definition as given. `Combine` refuses nodes that carry attributes.
- `SetAttribute` / `RemoveAttribute` are abstract over a class-level `kind`;
  `Implement` / `Unimplement` are `kind = Implementation` and nothing else. `Implement`'s
  field is `values`.
- `Function` is merged into `FunctionalNode`. The wrapper classes `Implemented*` are gone.
- Dropped on the way, by the owner's decisions: `still_valid` (a transition question),
  the keyed-by-class check (a type checker's job), `with_attribute`, `NodeNamedTwice`.
- Known leftovers: `Driven` refuses a condition that already carries attributes
  (temporary, goes with the driver change); `without_attribute` has one caller;
  `ImplementDrivers` sits next to the key in `implemented.py`.

## 3. Principles settled today

- **A node's class says what it is; its fields say how it connects; its attributes say
  everything else.** No wrapper node classes at any level.
- A field is anything an edge, a check, `+` or a structural rule reads. An attribute is
  opaque to those. A driver changes edges, so it is a field, not an attribute.
- Nodes are objects, so per-node data lives in the node. Places are only names, so
  per-place data has to be a side table on the graph.
- **Values stay outside the graph.** The graph defines the computation; values (and
  equinox module parameters given at a place) are per-run data passed to the run.
- The owner does not count violation classes as bulk; mechanism is what should shrink.
- Checks that only restate a type annotation do not belong in core.

## 4. Agreed order of what comes next

Each on its own branch off `main`, started only when the owner says so, left uncommitted
for review.

1. **Driver as one field on the condition node** (removes `Driven` / `DrivenNode`).
2. **Tower unification**: rule groups replace the `ProvedGraph … JitGraph` subclass chain. Section 6.
3. **Tags to a convention, plus a place-keyed table on the graph**, done together.
4. **`run(graph, values)`** with checks on the function, replacing `GraphWithValues`.

The owner asked whether to do the tower first; the reasoning for the driver first is that
it finishes what `node-attributes` left half-done and fixes the node shapes the tower's
rules read. The owner then agreed the driver change makes sense.

## 5. Step 1: driver as a field

**Built on `driver-field`** (off `main` at `c1c1e8a`, uncommitted, for review; 2026-11-17).
Built twice: first as a driver field on every kind (suite 1177), then reshaped by "stage A"
(the condition as a value, mapped with the peer session in `~/`). As it stands now:

- **Layers** (`core/maths`): `kinds` holds `FunctionalNode` and `AbstractConditionalNode`
  (the four names as `AbstractVar`s, `check_states_something`, `check_reads_on_one_side`,
  `is_encloser`). It knows no condition value and no attachment. `conditional`, above it,
  holds `Condition`, `Attachment`, the concrete `ConditionalNode`, the join, `kind_of`,
  `addable`, `is_driven` and `DrivenOnOneSide`. It has to stay in maths while `plan/` is
  in maths (`Combine`'s driver check, `SetOwnsInside`).
- **`Condition`** (a `Frozen` value, not a node): the four names, abstract `relabel`,
  optional `join` (class attribute, `None` = no join), `+` = its `join`. Its checks are a
  node's place rules (`check_owns_are_separate`, `check_reads_outside_owns`) and the two
  condition checks, so a condition is well formed before anything is attached.
  `RelationalCondition` and `OdeCondition` are `Condition`s with plain names (`reads_outside`,
  `relabel`). The `condition_…` names, `relabel_condition`, `condition_of` and `carried_text`
  are gone.
- **`ConditionalNode(condition, *, driver=None, attributes=...)`**: concrete, final. The four
  names forward to the condition; outside, it adds `driver.reads` / `driver.owns`. Its
  `relabel` renames condition and driver; `+` joins the conditions, then
  `driver_join_violations`, then `driver.joined`. Its repr is the condition's repr, then
  ` by <Contract>`, then the attributes. Only check of its own: `check_driver_fits`.
- **`Attachment`** (protocol): `reads`, `owns`, `relabel`, `violations_on(condition)`,
  `join_violations(other, condition, other_condition)`, `joined(..., joined)`.
  `core/driven/driver.py`'s `Driver` is the one implementation; `accepts` and `arity` take a
  `Condition`. Owner's framing: an additive attachment, possibly on `FunctionalNode` later.
- **Kind-keyed answers ask the condition.** `kind_of(node)` gives a condition node's
  condition, else the node itself. `NodeDrawing` / `ModelReading` are `about =
  (Condition, AbstractConditionalNode)` and `Spelling` is `(NodeDefinition, Condition)`; the
  lookups (`drawn`, architectures, JSON writer) pass `kind_of(node)`. The JSON reader wraps
  a read `Condition`; wire format unchanged. `Combine(at, nodes, kind)` takes the condition
  class (`RelationalCondition`) and checks `kind_of`. Shape predicates (`is_root_find`, ...)
  and `accepts` take conditions (`Drive.chosen` passes `kind_of(problem)`).
- Built from a condition: declarations' `condition_node`, `FixedPointCut.definition`, and
  `declare` (a bare `Condition` handed to it is bound as `ConditionalNode(c)`). `driven(x, ...)`
  takes a node or a condition. `SetOwnsInside` replaces `node.condition`'s `owns_inside`.
  `driving_violations` is `violations(ConditionalNode, node.condition, driver=..., ...)`.
- `check_reads_on_one_side` is in the prefix order (`overlaps`), not equality: reading `r`
  inside and `r.a` outside is refused. `ReadOnBothSides(inside, outside)`, pairwise.
- Structural checks (`executable`, `is_encloser`, architectures, drawing) use
  `isinstance(_, AbstractConditionalNode)`. **No structural check mentions a driver.**
- Tests: `tests/graphs.py`'s `graph_of` binds a bare condition as its node (`node_of`). Toy
  kinds are `Condition`s; nodes are wrapped where built. Every cottax test file passes, run in
  batches (not one full run).
- **processax**: the `.condition` reads that failed after the first build work again. New:
  `process/src/processax/process/study.py:153` reads `graph[problem].unknowns`, which needs
  `.condition.unknowns` (9 tests in `process/tests/test_scaling.py`, `vmcon/tests/
  test_vmcon_scaling.py`, `process/test_smoke.py`). Not edited: processax is the owner's.
- processax's `.venv/bin/python` link is gone (base env `process_port` no longer exists);
  run it with `fullcoupling`'s python3.12 `-S` and `site.addsitedir('.venv/lib/python3.12/site-packages')`.
- **Stage B, open, not started** (owner's decision each): `Attachment` down to places and
  `relabel` (fit as a driven-layer rule); `join` / `Combine` / `addable` out of core;
  `Attribute.fits` out of core; a base hook for whether the interior may be read outside;
  the plan as a consumer, after which `conditional` could leave maths.

The map as written before building:

```python
class ConditionalNode(NodeDefinition):
    driver : Driver | None = None          # kw-only, like `attributes`
```

| Today | After |
|---|---|
| `Driven(condition, contract, data, reports)` | `replace(condition, driver=Driver(...))`, plus a `driven(node, ...)` helper |
| `Assign` wraps, `Undrive` unwraps | set and clear the field |
| `is_driven` = `isinstance(DrivenNode)` | `node.driver is not None` |
| `node.contract`, `node.data_of(role)` | `node.driver.…` |
| `condition_of(node)` (7 uses) | deleted; dispatch on the node's own class |
| pairing and slot checks on `Driven` | asked of the condition node when it has a driver |
| `Implementation.fits` | a function, or a condition with a driver |
| JSON `driven` spelling | wire format unchanged |

Removed: `Driven`, `DrivenNode`, `undriven()`, `condition_of`, the forwarding properties,
the `ConditionCarries` guard.

**Four decisions still open (my recommendation in each):**

1. *Which names include the driver's places.* Keep the four names as what the condition
   states; add the driver's places in the base class's `reads` / `owns`; the two
   executable rules and the read-on-one-side check ask for "everything outside" through
   one helper.
2. *Relabelling.* Make `ConditionalNode.relabel` concrete: it asks the kind for its
   statement, then relabels the driver and keeps the attributes. (`OdeCondition.relabel`
   rebuilds from scratch and would drop both.)
3. *Joining.* The base class joins statements through the kind and joins the drivers
   once; `Driven.__add__`'s slot logic moves there.
4. *The problem without its solver.* `run.py` reads `node.condition` in three places
   (the condition's own reads, and `Posed.condition`). They need a projection that strips
   driver and attributes.

Size: `Driven(` 6 in src / 30 in tests; `DrivenNode` 17 in src; `.contract` 38 / 23;
`is_driven` + `condition_of` 19 / 5. processax: none found by grep.

## 5b. Layering before the tower (branch `tower`, uncommitted, 2026-11-17)

PR #9 (`driver-field`) is merged (`450739c`). Before the tower, core was re-layered so each
layer can be reviewed on its own (the owner: "maths will be graph + functions that can be
defined *just on the graph*, with nothing else"). Core, bottom up:

- `frozen`, then `maths`: places, mints, definition, kinds (`FunctionalNode`,
  `AbstractConditionalNode`), graph, nesting, dispatch, valued, executable, steps.
- `plan/` (was `maths/plan/`): the op algebra and the primitive and place ops.
- `conditional/` (was `maths/conditional.py`): `node.py` (`Condition`, `Attachment`,
  `ConditionalNode`, `kind_of`, `is_driven`), `ops.py` (`SetOwnsInside` and its finals).
- `mints` (was `maths/mints.py`): `Mint`, `Minting` and the purposes, core's one interface
  over the places, beside the maths since nothing in maths, plan or conditional asks it.
- `driven`, then `realisation`, as before.
- The front door `cottax.core` gives maths, mints, plan and conditional; `refusal_kinds` holds
  their kinds. `test_layering` names them `MATHS` < `MINTS`, `PLAN` < `DOOR`.

Out of core, to `consumers/architectures/`:

- **`Combine`** (`combine.py`), folding by `ModelReading.join(condition, other)`. `join = None`
  on the interface; relational and ODE answer it with their condition's own `join`. No `+`
  on a condition or node any more.
- **The driver join is gone**: `Combine` refuses a driven node (`FoldsDriven`); undrive,
  fold, drive the fold (test_ode_trajectory test 9 does this). Removed: `Condition.join`
  / `__add__`, `ConditionalNode.__add__`, `Attachment.join_violations` / `joined`,
  `Driver.join_violations` / `joined`, `DrivenOnOneSide`, `DriversDiffer`, `RoleAtTwoPlaces`.
- **The ODE join was kept**, against the earlier plan to drop it: studies fold ODEs by hand
  (test_ode, test_ode_trajectory test 9). `OdeModel.combines_as` stays `None`, so no
  architecture folds them.
- **`cut_rules.py`**, with its finals in `refusals.py`.
- The surface `combine` verb (no callers) is removed. `Combine` and the cut rules are in
  `cottax.assemblies.jax` as before.

processax: `study.py:19` imports `GaussSeidelMinimal` from `cottax.core`; that now lives in
`cottax.consumers.architectures.cut_rules` (or `cottax.assemblies.jax`). Owner's to change.

## 5c. Tower piece 1: rules where they belong (uncommitted, on `tower` after `5b45a75`)

The layering is committed as `5b45a75`. Piece 1 is built on top, uncommitted:

- Maths asks nothing of an attribute: `Attribute.fits`, `check_attributes_fit`, `misfits`
  and `AttributeDoesNotFit` are gone. `check_settable` keeps only `AttributeAlreadySet`.
- `Attachment` is `reads`, `owns`, `relabel`; `ConditionalNode` has no checks of its own.
- Driven layer: `check_driver_fits(name, node)` in `driven/proof.py`, beside `check_driven`;
  `DrivenGraph.checks_node` asks both. `Assign` asks the fit through `driving_violations`.
- Realised layer: `check_implementation_fits(name, node)` and the final
  `ImplementationWithoutDriver` in `realisation/implemented.py`; `RealisedGraph.checks_node`
  asks it after `check_implemented`, and `Implement.checks_graph` asks it of the result.
- Decision 1 as built: a misfit node can be built by hand (`driven(...)`, `implemented(...)`)
  and exists until its layer is proved.
- **`Undrive` no longer refuses an implemented node**: the rule is the realised layer's and
  `driven` is below it. The node it leaves is refused by `RealisedGraph`. Owner to confirm.
- **The JSON reader asks both node rules of each node it reads**, so a file is still refused
  as it is read.
- `in_graph` is dropped from the tower: no kind needs it yet (owner agreed 2026-11-17).

## 5d. Tower piece 2: rule groups replace the subclass chain (uncommitted)

Built on piece 1, storage untouched (still eager; `Valid` is piece 3).

- `core/maths/rules.py`: `Rules` (whole graph) and `NodeRules` (each node), fields `name`,
  `rules`, `implies`; `unproved`, `prove` (returns `None`: nothing new is made, so it is a
  statement, owner's choice 2026-11-17), `is_proved`, `proved(group, of=0, at='')`, `proves(fn_or_class)`, `carried`.
  The memo is `_satisfies` in the graph's `__dict__`, outside equality, hash and pickle.
  A node group may imply node groups only (`NodeRulesImplyWhole`).
- Groups: `ExecutableGraph` (the seven rules); per node `DrivenNodes`, `RealisedNodes`,
  `CallableNodes`, `JitKeyNodes` (implies `RealisedNodes`); composites `DrivenGraph`,
  `RealisedGraph`, `CallableGraph`, `JitGraph` keep the old names and only imply.
- `steps(graph)` replaces `SteppedExecutableGraph.steps`, remembered on the graph; each
  interior is `carried` (node groups) and proved executable.
- Gone: `ProvedGraph`, `SteppedExecutableGraph`, `checks_node`, `ProvedBelow` (a run proves
  what it needs). `outline(graph)` replaces the proof's `__repr__`. Drawings take
  `blocks=True` only.
- What is proved is declared, not buried: `run` is `@checked(proved(CallableGraph,
  at='graph'))`, `Traced.checks` and `steps` likewise; `proves(run) == (CallableGraph,)`.
- `unchecked(...)` does not reach a group's rules (a skipped rule would be remembered as
  proved); a test asks a smaller group instead.
- `in_graph` dropped; `carried(also=)` exists but nothing passes `also` yet.
- Open: `core/maths/dispatch.py` (`answer_for_places` / `answer_for_labels`) is a registry
  lookup, not maths. Fold into `frozen/answer` as one function over a sequence (10 src call
  sites, ~14 tests, 2 processax), or keep the two helpers beside their users. Owner to choose.

processax, owner's to change: `JitGraph(graph)` -> `prove(graph, JitGraph)` then pass `graph` (study.py:170,
benchmark capture.py:123, structure.py:33, preprocess wiring/graph.py:56, vmcon tests 83/138);
`stepped.steps` -> `steps(graph)` (benchmark structure.py:22/41); `g.graph.graph.components`
-> `g.graph.components` (preprocess test_milestone1.py:90). `cottax.assemblies.jax` exports
`prove`, `steps`, `unproved`.

## 5e. Tower piece 3, and the cleanups before it (uncommitted, 2026-11-17)

Owner asked: memo cleanup, `dispatch` and `valued` out of maths, then piece 3. All green
(default 1181 + slow 32).

- **One memo mechanism.** `frozen.remembered`: a function of a `Frozen` value and hashable
  arguments, remembered in the value's one memo slot (outside eq, hash, pickle), with
  `known`, `answers` and `remember`. It replaces `_satisfies`, `_steps` and `NestingGraph._levels`.
  `rules.broken(graph, group)` is a group's own violations, remembered. `closure(group)`,
  `unproved`, `is_proved` (known, asks nothing) and `carried` are read off it.
  `prove` returns `None` (owner's choice).
- **Out of maths.** `dispatch.py` is now `core/answers.py`, a layer beside `mints` that is
  in `DOOR` and `DEFINING`; `test_layering` has `ANSWERS`. `valued.py` is now
  `consumers/execution/valued.py`, and its three refusals are in `execution/refusals.py`.
  `GraphWithValues` comes from `cottax.consumers.execution`.
- **Piece 3, as §6.3, with these choices:**
  - One stored `NestingGraph(definitions, within=OrderedMap())` in `maths/graph.py`. It
    holds the three nesting rules (eager) and the bindings' reads: `nodes`, `variables`,
    `owned_variables`, `read_variables`, `readers`, `enclosers`, `outermost_enclosers`,
    `enclosing`, `subgraph`. `AbstractGraph` is gone.
  - `maths/dependencies.py` has `Valid` (the two binding rules, lazy) and the
    `Dependencies` view, with `owners`, `components`, `cycles`, `topological_order`,
    `ancestors`, `descendants`, `reach`, `boundary_*`, `feedback_variables` and
    `closing_readers`. `dependencies(graph)` is remembered, and building the view is the proof.
  - `maths/nesting.py` has the `Nesting` view (`cycle_of`, `interior`, `contract`) and
    `nesting(graph)`. Levels and interiors carry `Valid` through
    `carried(also=(Valid,))`, and so does `body_of`. `contract` proves `Valid` of
    its result.
  - `ExecutableGraph` implies `Valid`.
  - `GraphOp.apply(graph, keeps=Valid)` proves `keeps` of the input and of the result.
    Composed and staged ops apply their parts with `keeps=None`. Not on `Plan`: no
    repository plan exists yet, so it would be an unused field.
  - The doors prove `Valid` of what they build: `read_json` and `declare`.
  - Not done: `run(graph, values)` (step 4), and the `in_graph` slot (nothing needs it).
- processax, owner's to change:
  - `GraphWithValues` is no longer in `cottax.core` but in `cottax.consumers.execution`:
    benchmark `capture.py:15`, `study.py:19`, preprocess `wiring/graph.py:13`,
    vmcon `tests/test_vmcon_scaling.py:15`.
  - `graph.graph.owners` / `.boundary_inputs` become `dependencies(graph).…`, where `graph`
    is the `NestingGraph`: `study.py:111`, process `graphs/kinds.py:109,118,137`,
    preprocess `tests/test_coupled_reads.py:38,56,68`.
  - `g.graph.graph.components` (`test_milestone1.py:90`) becomes `dependencies(g.graph).components`.
  - `.graph.owned_variables` loses one `.graph`.
- **After 5e (owner asked):**
  - `Place` is a protocol with two attributes, `segments` and `prefixes`, and nothing else.
    The maths only reads a place and never builds one. `places.prefixes()` is gone: the
    rules read `p.prefixes`. The test toys take theirs from `tests/toy_places.Segmented`.
  - `Attribute` is now `type Attribute = type`. The key of a node's `attributes` is a plain
    class, and `text` is gone. A node's repr names the kinds it carries (` +Implementation`).
    `Implementation` and the test `Note`s are plain classes.
  - `attributes_text`, `with_attributes` and `without_attribute` are gone from maths. Node
    reprs no longer show attributes. The five edit sites (`SetAttribute`, `RemoveAttribute`,
    `implemented`, `Redefine`, the JSON writer) use `dataclasses.replace(node, attributes=…)`.
    Maths keeps only the opaque `attributes` field and the `Attribute` alias.
  - `NestingGraph` is renamed `Graph`. processax: `process/graphs/kinds.py`.
  - **Memo settled (owner, after two agent reviews):**
    - `frozen.remembered` is a 10-line `cached_property` for a function written outside the
      class: no arguments, no `known`/`answers`/`remember`.
    - Proofs are a set per graph (`rules.proofs`), written only by `unproved`.
    - `carried`, `also=` and `NodeRulesImplyWhole` are gone. Nothing is copied to a
      subgraph; an interior is proved when asked. This also fixes a history-dependent memo:
      a node group proved before `steps` used to reach interiors, one proved after did not.
    - `Nesting._levels` is a dict on the view.
    - Inventory: `test_layering.REMEMBERED` pins every `@remembered` in src; `Graph`'s module
      docstring lists them. The pin is a review aid, not a gate on outside users.
    - Left as is: `Plan.__add__` primes `trunk` / `graph` (its own `cached_property`s;
      without it each `+` rebuilds the chain).
    - Offered, not done: `Solve` holding its interior's steps (one tree, no memo lookup in
      host loops); `checked` precomputing its signature (~8 µs/call).
  - `Graph` checks nothing at construction. The three nesting rules joined `Valid`
    (now five rules), so ops (`apply` keeps it) and the doors still refuse at once.
  - `readers_of` inlined into `Graph.readers`. `owners_of` stays: the binding check must not
    read the view it guards.
  - **`wellformed.py`** (between `rules` and `dependencies`) holds the five rules, `owners_of`,
    `Valid` and their refusals. `graph.py` imports neither `kinds` nor any rule.
  - **No "encloser" any more.** `kinds.owns_inside(d)` (a condition node's `owns_inside`, `()`
    otherwise) and `kinds.conditions(defs)` replace `is_encloser` / `enclosers` /
    `Graph.enclosers`. `outermost` and `enclosing(p)` moved to `nesting(g)`.
    `NotAnEncloser` -> `EndOwnsNothingInside` (same message). Post-executable sites (driven,
    roles, drive, structures) use `conditions`; architectures keep a `problems(graph)` helper
    (owns inside: requirements still differ there).
  - **rules.py, one group class.** `NodeRules` is gone: a node rule is lifted with
    `each_node(rule)`, a small value, inside a plain `Rules`. `closure` is no longer
    `remembered` (`REMEMBERED` is now four, all on a graph). `proved` is a value
    (`group`, `of`, `at`) with `__call__`, so `proves` reads `isinstance` and nothing is
    written onto a function. `unproved` skips a group implying one that failed, so a group's
    rules may assume what it implies (the `is_proved` guard in
    `check_nesting_ends_on_one_cycle` is gone). `at=` stays a dotted path: `jax_jit` uses
    `at='valued.graph'`.
  - **Groups are classes.** `class ExecutableGraph(Valid): rules = (...)`; the bases are
    what a group implies, `closure` is computed from them at class creation (each once,
    in the order written), `asked` reads the class's own `rules` only. A group with no rules
    of its own (`DrivenGraph(ExecutableGraph, DrivenNodes)`) is proved once its bases are.
    Never instantiated; annotated `type[Rules]`. No more `Rules('Name', ...)` strings.
  - processax breaks: `is_encloser` in `process/graphs/kinds.py:13,93,126` and
    `preprocess/tests/test_coupled_reads.py` -> `owns_inside` (truthy); `g.graph.enclosers` in
    `preprocess/tests/test_milestone1.py:129,149` -> `conditions(g.graph.definitions)`.
  - Tests at end of day: the touched files only (never the full suite), all green, slow ones
    too. Last batch: 23 files, 572 passed + 18 slow.

### 5f. Next session: start here

**Settled in discussion (not code changes):**
- *Class groups vs the old `ProvedGraph` tower.* Same names, and inheritance still means
  "implies", but a group is never instantiated: the proof is a fact in the graph's set, not
  the value's type. So no wrapper, no forwarding, no `.graph.graph`, one jit key per graph,
  a DAG instead of a chain, and node and whole-graph rules can mix in any group. A
  combination needs a class only to be *named*; proving its bases proves it.
- *Making a group the view* (`class Valid(Frozen): graph`) would be the tower again:
  rejected.
- *Node checks stay on the nodes, at construction.* A rule one node alone decides, and
  whose breaking is never a meaningful intermediate state, belongs on the node. Relational
  rules go to a group (`Valid`), and so do per-node rules about a stage (driven,
  implemented, jit). Test for moving one: is a node breaking it ever meaningful mid-plan?
  (A requirement is, which is why "owns something inside" is `check_no_requirements`.)
- `definition`, `kinds`, `places` and `graph` are fine as they are (owner).

**Built (uncommitted, 2026-11-18): `rules.py` trimmed, 137 -> about 120 lines with docstrings.**
Decided after a blind review (the metaclass idea was rejected, see below).
- A group is a check through a classmethod: `@checked(Executable.check)`,
  `checks = (Valid.check,)`. `check(held, *_)` proves `held` if it is a `Graph`, else
  `held.graph` (one hop; anything else is a `TypeError` naming the group). Extra arguments
  of a checked function are ignored. `Traced` gained a `graph` property for this.
- `proves(f_or_cls)` = the checks that are some group's `.check` (bound classmethods compare
  equal, so `unchecked`/`skip=` still work). `test_execution.py:286` still holds.
- `@node_check` marks a rule `(name, node)` as per-node (an attribute, the function is
  unchanged, so `check_driver_fits(name, node)` etc. still work directly); `Rules.asked`
  loops the nodes for it. `each_node` gone. Works on a lambda: `node_check(lambda ...)`.
- `proved`, `at=` and `is_proved` gone; tests read `X in proofs(g)`. `proofs` is exported
  (kept public: it is the documented memo). `__new__` guard kept (clear message).
- Groups renamed to adjectives, so they read as properties, not constructions:
  `ExecutableGraph`->`Executable`, `DrivenGraph`->`Driven`, `RealisedGraph`->`Realised`,
  `CallableGraph`->`Runnable`, `CallableNodes`->`RunnableNodes`, `JitGraph`->`Jittable`.
  `*Nodes` groups keep their names.
- Rejected: a metaclass `__call__` making `Executable(g)` yield violations. A generator is
  truthy, so `if Executable(g):` would always pass; mypy does not honour a metaclass
  `__call__` return; `violations()` would take the Frozen-class branch; `checked` passes
  every argument. Rejected: strict groups plus named helper checks (`proves` loses them).
- `closure` (a ClassVar set in `__init_subclass__`) replaced by the classmethod
  `Rules.implied()`, walking the bases each ask (no cache: a handful of classes). The
  failed-skip in `unproved` is `issubclass(g, f)`; no MRO read anywhere.
- `node_check` (a marker) became `per_node_check`, a class wrapping `rule(name, node)`:
  itself a graph rule (`__call__(graph)` loops the nodes), the one-node rule at `.node`
  (`json/format.py`, `implemented.py`, `test_contracts`, `test_realisation` call it so).
  `Rules.rules` is typed `Callable[[Graph], Iterator[Violation]]`; `asked` folded into
  `unproved` (reads `g.__dict__['rules']`, own only). `test_incremental`'s probe counts
  `Valid`'s first rule instead of patching `asked`. `held` is typed
  `Graph | HasGraph` (a Protocol, owner's), no runtime TypeError.
- Then (2026-11-18, later): the free functions went. `Rules` carries `check(graph)` (list)
  and `prove(graph)` (raise); `unproved`, `prove`, `proves` removed (`proves` test reads
  `run.checks == (Runnable.check,)`). Exports from rules: `Rules`, `per_node_check` only.
- The memo is pure: `@remembered def _satisfies(graph, group) -> bool` (bases first, own rules
  to the first violation; no violation kept), replacing the mutable `_proofs` set.
  `remembered` takes hashable keys: `fn(value, *key)`, cached at `(fn, *key)`
  (`test_frozen` has a keyed test). `check` lists a group's own violations only when it
  fails and its bases hold; a failing group's rules run twice (once to the first, once to
  list), on the failure path only. Tests read the memo via `tests/proofs.py:recalled(g, X)`.
- Tests: 30 touched files, 845 passed + 26 slow.
- processax breaks (owner's to fix): `JitGraph` imported and *called as a wrapper*
  (`JitGraph(graph)`) in `process/study.py:15,170`, `benchmark/capture.py:123`,
  `benchmark/structure.py:9,33`, `preprocess/wiring/graph.py:10,56`,
  `vmcon/tests/test_vmcon_scaling.py:13,83,138`. Already broken since groups became
  classes (never instantiated); the fix is `prove(graph, Jittable)` then pass `graph`.

**Built (uncommitted): `WellFormed` split and renamed after the paper's "valid graph".**
`maths/wellformed.py` -> `maths/valid.py`. `ValidNesting` (ends bound, no loops, ends own
inside) and `ValidDependencies` (one producer, reads exact) are independent halves;
`Valid(ValidNesting, ValidDependencies)` has no rules of its own (nesting first, so the
order violations are reported in is unchanged). `Dependencies.checks` asks only
`ValidDependencies`, `Nesting.checks` only `ValidNesting`; everything else (`Executable`,
`apply`'s `keeps`, `declare`, `read_json`, `Nesting`'s rewrites) asks `Valid`. Proving
`Valid` caches both halves. `test_incremental`'s probe wraps `ValidNesting`'s first rule.

**Next: `dependencies.py` (discussed 2026-11-18, nothing built).** Everything up to and
including `valid.py` is committed and pushed (owner). Findings:
- Caching is two-level: `dependencies(g)` is `@remembered` (one view per graph, proving
  `ValidDependencies` at construction), each fact a `cached_property` on the view.
- networkx is used only for SCC condensation + tie-broken topological sort and BFS
  reachability (~30 lines of plain Python), held as a *mutable* `nx.DiGraph` inside a frozen
  view (`test_cache` special-cases it; `test_visualization:304` pokes `_nx_dependencies`).
  Outside core only `consumers/visualization` uses networkx. Proposal: the view's data is the
  edges (`successors` / `predecessors` maps), Tarjan + Kahn (tie-break by binding order) and
  BFS in plain Python; networkx becomes an optional `[viz]` extra.
- Unused surface: `topological_order` (0 src uses), `feedback_variables` and `closing_readers`
  (tests only; owner to decide), `is_acyclic` (= `not cycles`, 2 uses), `components_order`
  (1). `boundary_inputs` / `boundary_outputs` need no edges or proof (set differences).
- Use counts in src: `.owners` 16, `.boundary_inputs` 12, `.components` 5,
  `.boundary_outputs` 3; `nesting(g).cycle_of` 9.
- The view pattern (Frozen with `graph`, `checks = (Group.check,)`, cached properties, one
  `@remembered` accessor) is the template for any further view; a generic keyed
  `view(graph, cls)` was considered and not recommended.
- Built (uncommitted): `boundary_inputs` / `boundary_outputs` moved from `Dependencies` to
  `Graph` (`cached_property`, next to `read_variables`); 25 call sites `dependencies(x).boundary_*`
  -> `x.boundary_*`. They no longer prove `ValidDependencies` on the way (they need no
  edges). processax `process/graphs/kinds.py:137` already reads `graph.graph.boundary_inputs`.
- Decided: no forwarding/patching of view facts onto `Graph`; bind `deps = dependencies(g)`
  locally where several facts are read; a `UserFacingGraph` later only if really needed.
- Open question from the owner: `dependencies(g).owners` is verbose; can the facts be
  reached from the graph? See the answer given at the end of the 2026-11-18 session
  (options: local binding, delegating free functions, forwarding on `Graph`, patching).

**Built (uncommitted): views are `View`s, got with `.of`.** `graph.py` holds `View(Frozen)`
(field `graph`; `View.of(graph)` -> `_view(graph, cls)`, `@remembered`, keyed by the class) next
to `Graph`: a view is a pure function of the graph alone, cached on it. `Dependencies` and
`Nesting` subclass it; the accessors `dependencies(g)` / `nesting(g)` are gone (call sites
`Dependencies.of(g)` / `Nesting.of(g)`, ~48 src + ~118 tests). Inventory: `_view` replaces the
two accessors (`_satisfies`, `_view`, `steps` remain). New `test_layering` rule: no view class
is called directly in `src` outside `graph.py`. Rule recorded in `View`'s docstring: anything
that also depends on values is not a view. `steps(g)` stays a remembered function (a tuple).
processax uses neither accessor.

**Next: split the views by what they read (agreed in discussion 2026-11-18, not built).**
`Nesting` mixes three things: `outermost` / `enclosing` (need `ValidNesting` only),
`cycle_of` / `interior` (need both halves: they build `Dependencies.of(subgraph)` inside,
so the dependency proof is asked deep down), and `contract` (an edit, not a fact; one use,
`architectures.py:470`). Its `_levels` dict is a mutable cache filled by `cycle_of` (the
`_proofs` smell again). Plan:
1+2 BUILT (uncommitted, 2026-11-19): no separate `Cycles` view -- no caller wants a valid
   nesting without valid dependencies (every `Nesting.of` in src is on a `Valid` or
   `Executable` graph). `Nesting.checks = (Valid.check,)` (honest: `cycle_of` / `interior`
   read both halves). The `_levels` dict is gone: `@remembered def _level(graph, out:
   frozenset)` in `nesting.py`, pure and keyed (inventory: `_satisfies`, `_view`, `_level`,
   `steps`). `ValidNesting` kept as a group though no view asks it alone (owner: it makes
   the two halves of `Valid` explicit); order unchanged.
   THEN (2026-11-19, uncommitted): `_level` dropped again -- a keyed `remembered` over sets
   of node names stretches its "constant keys" rule. `Nesting._cycles` (a `cached_property`,
   complete when made: one entry per encloser, enclosers nested alike share a level built
   once) replaces it; `cycle_of(e)` = `enclosing(e)` (to refuse a non-encloser) then
   `_cycles[e]`. Inventory back to `_satisfies`, `_view`, `steps`. Cycles stay on `Nesting`,
   not an executable view: the executable rules use `cycle_of` to decide executability.
   Vocabulary: "problem" -> "encloser" (works for DoE / sweep / switch, not only solves).
   `kinds.enclosers(definitions)` (condition nodes owning something inside; exported)
   replaces `executable._problems` and `architectures.problems`; `Nesting` params and
   docstrings, `body_of`, the layer diagram renamed. NOT renamed (separate decisions):
   violation fields / messages in `executable.py` (`problem : Label`, "declares several
   problems ..."), `Solve.problem` in `steps.py`, architecture domain names
   (`TopNotAProblem`, `body_of_problem`). processax: `is_encloser` sites (5e) can use
   `owns_inside(d)` / `enclosers(defs)`.
3. BUILT (uncommitted, 2026-11-19): `contract` is the op `Contract(blocks)` in `plan/ops.py`, a
   `ComposedOp` of `Unnest` (pairs touching a block), `Delete` (members but the first, in
   binding order) and `Replace` (first member -> its box); no graph-building code of its own,
   `Valid` kept by `apply`. `checks = (check_blocks_partition,)` (op alone: `BlockTooSmall`,
   `NodeInTwoBlocks`), `checks_graph = (check_blocks_bound, check_blocks_own_something)`
   (`NotBound` -- a KeyError as before -- and `BoxOwnsNothing`); the three finals moved to
   `plan/ops.py`. `check_nesting_ends_on_one_cycle` folded into `executable`'s
   `check_nestings_share_a_cycle` (it only wrapped it), `Unreachable` moved to
   `executable.py`. `nesting.py` now holds only the view. Call sites: `architectures.py`
   (`Contract(blocks).apply(graph)`) and 6 test sites.
4. `Steps(View)`, `checks = (Executable.check,)`, holding today's `steps(g)` tuple (property
   name open: `.all` or `.order`); `steps` loses `@remembered @checked`. Inventory then
   `_satisfies`, `_view`, `_level`. `steps(` is used in 2 src, ~19 tests, 2 processax sites.
5. `body_of` is not a nesting fact (never reads `within`): keep a free function next to the
   executable rules, renamed `body(graph)`, defined as "without condition nodes"
   (`conditions(graph.definitions)`) rather than "without nodes owning inside" -- equal on
   every graph the executable rules accept. Not a `Graph` property: that would bring `kinds`
   back into `graph.py`. 5 uses.
Also still open for `dependencies.py`: trim `topological_order`, `is_acyclic`
(owner's call on `feedback_variables`, `closing_readers`); networkx stays (judged not worth
replacing; only the mutable `nx.DiGraph` in a frozen view is a minor smell). User-facing
facts: per-layer facts mixins composed in the assembly (`About`), built when the first
higher-layer view lands.

**Built (uncommitted, 2026-11-19): items 4 and 5, `Dependencies` trim, the encloser wording.**
- `body_of` -> `body(graph)`, "the graph without its condition nodes" (`conditions`), next to
  the executable rules. `sequencing.py`'s local `body` renamed `body_graph` (it shadowed).
- Steps are a view: `Schedule(View)`, `checks = (Executable.check,)`, `Schedule.of(g).steps`
  (not `Steps`: taken by the driven role and the surface mint). The step `Solve` -> `Loop`,
  field `problem` -> `encloser` (three classes were named `Solve`; the drawing entry in
  `sequencing` and the declaration mint keep theirs). Inventory: `_satisfies`, `_view`.
  `assemblies/jax.py` imported core `steps` and rebound the name to a mint: dead import and
  duplicate `__all__` entry removed.
- Executable violations: fields `problem(s)` -> `encloser(s)`, `NoProblem` -> `NoEncloser`,
  messages reworded ("several outermost enclosers", "has no encloser", "no encloser on it").
  `maths/` now says "encloser" throughout except two `valid.py` messages.
- `Dependencies`: `topological_order` removed by the owner (tests moved to `components_order`
  / `cycles`); `is_acyclic` a plain `@property` (the owner's plain method was read as an
  attribute everywhere, so it was always truthy). `feedback_variables` / `closing_readers`
  kept: `docs/source/theory/theory_6_i.ipynb` uses them (through an outdated API).
- processax converted to this API (2026-11-19, uncommitted in ~/processax): JitGraph wrappers ->
  `Jittable.prove(g)` then `g`; `is_encloser` -> `owns_inside`; `g.graph.X` (old NestingGraph)
  -> `Dependencies.of(g).X` or `g.X`; `GraphWithValues` from `consumers.execution`,
  `GaussSeidelMinimal` from `consumers.architectures.cut_rules`; `structure.py` on
  `Schedule`/`Loop`. Targeted tests pass (79 + 26 slow skipped). `capture.replay` and the
  edited example notebooks are not exercised.

**Recursive schedule (2026-11-19, uncommitted):** the step `Loop` -> `Drive(encloser,
interior: Schedule)`: a tree, the interior is `Schedule.of(interior graph)` (`.graph` for the
driver's posing), so the old explicit `Executable.prove(interior)` is the view's own check,
stated as following from the whole's proof. `walk`'s `AtSolve`/`at_solve` -> `AtDrive`/
`at_drive`, still handed the interior graph. processax breaks (owner's): `benchmark/
structure.py` imports `Loop` (:9, now `Enclosed`) and uses `step.interior` as a graph (:24, :40-41).

**Blind re-review: 8.5/10** (old 5). Done after it (uncommitted): `Dependencies.owners` ->
`.owner` (single, the shape in the name; `Graph.owners` the tuples); rules docstrings say
`@checked(X.check)` is for one-argument functions; `subgraph` uses `dataclasses.replace`
(a graph asks nothing, so `_carried`'s ambient switch suppressed nothing); `cycle_of` asks
`enclosing` only on a miss; "problem" gone from `valid.py`'s messages; `remembered`
docstring states the lifetime and the kinds-not-inputs convention. Open from it: tests
pinning the two theorems (subgraph of a Valid graph is Valid; interior of an Executable
one is Executable); plan items later. DONE: the step is `Enclosed(encloser, interior: Schedule)` (`Drive` named
the driver, a `ConditionalNode` thing, and clashed with execution's `Drive` op);
`AtEnclosed`/`at_enclosed` in `walk`; `executable.py`'s messages say encloser, not driver.
processax breaks: `Dependencies.of(..).owners` at `study.py:112`, `kinds.py:109,118`,
`test_coupled_reads.py:38,56,68`.

**From the blind review (2026-11-19), deferred by the owner:**
- `refusals(op, graph)` on an invalid graph: asks no `Valid` first, so a `checks_graph`
  rule can meet a broken graph (`Cut`'s builds a view and raises). And `Redefine.rebinding`
  raises `NotBound` through `first` instead of a `checks_graph` entry, so `refusals` never
  reports it. Later.
- `Contract.composed` reads the private `graph._position`: make it public, or stop. Later.
- Done: `Graph.owners` (place -> every owner, a tuple like `readers`, true of any graph);
  `Dependencies.owners` is the single owner, unpacked (`(n,)`) under its proof; `owners_of`
  gone. `check_reads_are_exact` names every owner (`ReadInsideOwner` / `OwnerInsideRead`
  field `owner` -> `owners`); `nx_dependencies(graph)` takes the graph.
- Left: `Nesting._cycles` re-proves `ValidDependencies` per level subgraph -- a theorem
  (removing nodes keeps it), cheap, not worth carrying proofs through `subgraph`.
- Declined: a failing group runs its rules twice (`_satisfies` then `_own`); threading
  `keeps` through `Plan`.
- Done: stale text (`outline` prints `encloser[members]`, the `SeveralOutermost` message
  no longer names `Combine` (an architectures op), `body` formula `G \ N_C`, `rules.py` and
  `maths/__init__` docstrings, empty section in `graph.py`); `Rules.check`/`prove` take the
  graph alone (no `*_, **__`: every caller passes one argument); `Attribute` documented.

**Still open from earlier:**
- `body_of` into the nesting view (`nesting(g).body`), recommended.
- Optional: `Solve` holding its interior's steps (one tree); `checked` precomputing its
  signature (~8 µs/call).
- processax: the breaks listed in 5e are the owner's to fix.

## 6. Step 2 mapped: the tower becomes rule groups

Design agreed on 2026-11-13, **nothing built**. Copied here from the older artifact and
checked against the code at `ca88b52`; names follow PR #7 (`enclosers`, not `problems`).
Numbers marked *reviewer* come from a review agent's experiments and were not re-run.

### 6.1 Today

```
AbstractGraph            definitions; checks one producer per place, exact reads   (graph.py)
NestingGraph             graph + within; three nesting rules                       (nesting.py)
ProvedGraph              wraps a NestingGraph, forwards 11 names by hand           (nesting.py)
  ExecutableGraph        seven whole-graph rules; __repr__ / _outline              (executable.py)
    SteppedExecutableGraph   steps; from here down only `checks_node`              (steps.py)
      DrivenGraph        check_driven                                              (driven/proof.py)
        RealisedGraph    check_implemented                                         (realisation/proof.py)
          CallableGraph  check_function_callable, check_implements_its_driver      (execution/run.py)
            JitGraph     check_hashable, check_reassembles                         (execution/jax_jit.py)
GraphWithValues          graph + values; three value checks                        (valued.py)
```

Measured now (src / tests / processax): `.graph.graph` 8 / 4 / 1; `isinstance(x, ProvedGraph)`
unwraps 2 / 0 / 0; `ExecutableGraph(` 0 / 77 / 0; `CallableGraph(` 0 / 59 / 0; `JitGraph(`
0 / 12 / 6; `GraphWithValues(` 3 / 49 / 5; `violations(` 7 / 109 / 0.

### 6.2 Why

- **One graph type through every consumer.** No unwrapping, no forwarding, no `.graph.graph`.
- **The proof level leaks into the value.** `ExecutableGraph(g) != SteppedExecutableGraph(g)`
  though the hashes match, so one graph at two levels is two jit keys (*reviewer*).
- **Rules extensible by node kind.** Below `SteppedExecutableGraph` only per-node checks are
  allowed, so a rule about a node and its surroundings has nowhere to go.
- **The order is partly arbitrary.** driven → realised → callable is a real dependency;
  "is a jit key" depends only on realised.
- **A repository must be a graph**: several models producing one place, before selection.
  So storage cannot check one-producer-per-place at construction.

Not reasons: size (the mechanism costs about what the class shells save, *estimate*) and
runtime (a cold proof of a 185–290 node graph is 8–23 ms).

### 6.3 The design

**Storage.** One stored type, readable as data and nothing more:

```python
class NestingGraph(Frozen):
    definitions : OrderedMap[N, NodeDefinition[V]]
    within      : OrderedMap[N, N] = OrderedMap()
```

- Checked at construction: field types and the three nesting rules (they read only
  `within`, names and kinds).
- On it: `nodes`, `variables`, `owned_variables`, `read_variables`, `readers`, `enclosers`,
  `outermost_enclosers`, `enclosing`, `__getitem__`, `__contains__`, `subgraph`.
- Not on it: `owners`, the edges, anything derived from them.
- Gone: `AbstractGraph` as a stored layer, `ProvedGraph`, `GraphWithValues`.

**Rule groups.** Values, not classes. Two kinds, because only one survives restriction:

```python
class Rules(Frozen):      name: str; checks: tuple; implies: tuple = ()   # check(graph)
class NodeRules(Frozen):  name: str; checks: tuple; implies: tuple = ()   # check(name, node)

def unproved(graph, group): ...      # violations as data, implied groups first
def prove(graph, group) -> None: ... # raise the first, else remember on the graph
def proved(group, of=0): ...         # `prove`, shaped as a check for @checked
def carried(parent, child, also=()): ...   # copy what survives restriction
def view(requires, on=lambda graph: graph): ...   # class of queries -> function of the graph
```

- The record of what is proved is a memo in `__dict__`: outside equality, hash and pickle.
- `Valid` is also recorded on the `definitions` map, so `Nest` / `Unnest` do not re-run it.
- `subgraph` carries the `NodeRules` groups and `Valid`; `contract` carries
  `Valid` only (a boxed `FunctionalNode` has no implementation).

| Group | Kind | Rules | Implies |
|---|---|---|---|
| `Valid` | whole graph | one producer per place, exact reads | — |
| `ExecutableGraph` | whole graph | the seven executable rules | `Valid` |
| driven, realised, callable, jit-key | per node | 1, 1, 2, 2 | per-node groups only |
| `DrivenGraph`, `RealisedGraph`, `CallableGraph`, `JitGraph` | composite | none | `ExecutableGraph` + the per-node parts |

- A per-node group may imply only per-node groups; otherwise a proof carried to a subgraph
  asserts executability of something that is not (*reviewer*, demonstrated).
- The jit-key group implies realised, not callable.
- The two binding rules are written against raw `owners_of` / `readers_of`, never the view
  they guard.
- No compatibility wrappers under the old class names.

**Views.** A frozen object holding the graph, queries as cached properties. Getting the
view is the proof, so a query cannot be left unguarded.

| View | Requires | Holds |
|---|---|---|
| `dependencies(graph)` | `Valid` | `owners`, edges, `components`, `cycles`, `topological_order`, `ancestors`, `descendants`, `reach`, `boundary_inputs`, `boundary_outputs`, `feedback_variables`, `closing_readers` |
| `nesting(graph)` | `Valid` | `cycle_of`, `interior`, `contract` |
| `steps(graph)` | `ExecutableGraph` | `Run` and `Solve` steps |

`dependencies` is memoised on the `definitions` map. Views are the only route to `owners`
and the edges. Keep the set of step kinds open.

**Rules a node kind contributes.**

```python
class Sweep(ConditionalNode):
    in_graph = ((ExecutableGraph, check_interior_not_read_outside),)   # check(graph, name)
```

Proving a group asks its own checks plus what the kinds present contribute. A class-level
slot keeps the proof a function of the value; a global registry was rejected (import
order, stale memo). No contributions into a carried group unless per-node.

**Ops and plans.**

- Primitives build their result unchecked; `checks_graph` keeps every intent check
  (`NameTaken`, `NotBound`, `Stranded`, `AlreadyNested`, …).
- `GraphOp.apply(graph, keeps=Valid)` proves `keeps` of the input, asks the intent
  checks, applies, proves `keeps` of the result. A composed op proves `keeps` once.
- `Plan(base, ops, keeps=Valid)` proves `keeps` of its base and passes it on, also
  after unpickling. A repository plan keeps nothing until a selecting op resolves it.
- `keeps` is not a field of the graph (it would enter equality and the jit key). It is on
  `apply` and not only on `Plan` because 25% of op applications in the suite are bare
  (*reviewer*).

**Running.**

```python
@checked(proved(CallableGraph),
         check_every_input_given, check_nothing_owned_given, check_only_inputs_given)
def run(graph, values): ...
```

`Traced` holds `graph` and `values`. `ExecutableGraph.__repr__` becomes `outline(graph)`.
`violations(TowerClass, g)` in tests becomes `unproved(g, group)`. Drawings take
`blocks=True` instead of a proof instance.

| Today | After |
|---|---|
| `NestingGraph(AbstractGraph(OrderedMap(nodes)), OrderedMap())` | `NestingGraph(OrderedMap(nodes))` |
| `graph.graph.owners`, `.boundary_inputs`, `.components` | `dependencies(graph).owners`, … |
| `graph.cycle_of(p)`, `.interior(p)` | `nesting(graph).cycle_of(p)`, … |
| `ExecutableGraph(graph).steps` | `steps(graph)` |
| `run(GraphWithValues(CallableGraph(graph), values))` | `run(graph, values)` |
| `graph.nodes`, `graph.enclosers`, `graph[name]` | unchanged |

### 6.4 Order within the step

1. **Rule groups replace the tower.** The rules module, the groups, `steps(graph)`.
   Storage untouched and still eager, so `ExecutableGraph` implies nothing yet. Stands on
   its own; almost all of it is test call sites (counts in 6.1).
2. **Flat storage and views.** `Valid` becomes lazy, `keeps` on `apply` and `Plan`,
   the `in_graph` slot. Touches `.graph.owners`-style reads in src and processax.

### 6.5 What changed since the design was written

- **`run(graph, values)` was part of sub-step 1 in the design; section 4 lists it as its
  own step 4.** They overlap: once `proved(CallableGraph)` exists, `run` takes it for free.
  My reading is that step 4 shrinks to deleting `GraphWithValues` and moving `Traced` over.
  Not confirmed with the owner.
- **The driver change comes first**, so the driven group's rule reads `node.driver`, not
  `isinstance(DrivenNode)`, and the realised rule reads the `Implementation` attribute.
- **The choice table of a repository** was to be `Declared.node_tags`. Tags are going
  (section 8); a node attribute is the natural home now. Not discussed.
- The "operator node with two faces" follow-up in the old design is done (PR #7), under
  the names `owns_inside` / `reads_inside` / `reads_outside` / `owns_outside`.

- **Validity rules move to the layer that needs them** (agreed 2026-11-17; page "Rules Where
  They Belong", https://claude.ai/artifact/XfWBRUVUvoKQdvRq5sgjN7). `Attribute.fits` and
  `Attachment.violations_on` are rules from above that run inside a maths node's
  construction. They become node rules of the realised and driven groups (two rules each, not
  one), asked eagerly by `Implement`, `Assign` and `Undrive` through `checks_graph`.
  `misfits` / `AttributeDoesNotFit` leave maths, and `check_settable` keeps only
  `AttributeAlreadySet`. `Attachment` shrinks to `reads`, `owns` and `relabel`.
  Decision 1, owner to confirm in review: a misfit can exist until its group is proved.
  Decision 2: `Combine` refuses driven nodes (processax has no folds), so the driver-join
  code leaves maths. `Condition.join` stays.
- `in_graph` (6.3) goes on the condition class, read through `kind_of`: `ConditionalNode` is
  final now.

### 6.6 Rejected, so it is not reopened by accident

| Option | Why not |
|---|---|
| Keep the tower, fix it inside (forwarders, a memo) | Cannot hold a repository; class stays in equality; linear order stays |
| Flat storage that still checks everything eagerly | Rules the repository out of the type |
| Ops stop checking, lazy proof, `Plan.breaking` bisects | Errors arrive late and blame the wrong op; replaced by `keeps` |
| `keeps` only on `Plan` | Misses every bare `apply` |
| `@requires` on 25 separate query functions | Views pay once and cannot leave a query unguarded |
| A boolean `hereditary` flag on a group | A wrong flag silently copies a false proof |
| A global registry of contributed rules | Import order; stale memo |

### 6.7 Open

- Where the three value checks live besides `run` (working answer: `Traced` gets them
  through `run`; `valued()` returns values only).
- Whether `prefixes` on the place protocol becomes required (today optional via `getattr`).
- Older, unrelated: networkx in core vs a plain Tarjan; `OrderedMap.trusted` → `_trusted`.
- Deferred with a sweep kind: "does the operator end at a point" and the scope of an
  encloser in core. Only the old artifact (its section 7) has that discussion, with the
  operator catalogue, partial evaluation, switches and repositories.

## 7. Driver contracts: what was found, and the open design

A contract today is `accepts` + `takes` + `reports` (role classes). `data` / `reports` on
the node are the places for those roles, matched by position. A contract does two jobs:

- **Wiring** (which roles, shareable across nodes, compared on join). Sound.
- **Family identity** (ten classes in two tiers, boolean flags `steps` / `scaled` /
  `trajectory`). This is where the weirdness is: selection runs both ways, four
  overlapping matchers (`entry_for`, `Drive.covering`, `contract_for`, `chosen`), roles
  stated three to five times, `needs` cannot say "optional".

Why the split into a contract plus positional tuples exists (read from the code): the
contract is node-independent and the places are not; `AssignByRule` records a contract and
a place rule; a join needs "same contract"; the contract's class encodes legal
combinations (a trajectory needs times).

Direction discussed, **not decided**:

```python
Wiring(family=RootFinding, takes=(Start,), reports=(Steps,))     # shareable, no places
Driver(wiring, places={Start: (...), Steps: (...)})              # on the node
```

A non-reporting driver is `reports=()`. The flags and the level/final hierarchy go;
`ImplementDrivers` folds into `Drive`.

Proposed split: (a) wrapper out, field in, contract object unchanged; then (b) flatten
the contract. Questions for the owner before (b):

- Should "which family answers this node" be recorded in the backend-free graph?
- Is it deliberate that Newton cannot run a node assigned as fixed-point iteration?
- Is asking for a role (steps, scaling) per `drive` call, per node, or a surface default?

## 8. Step 3 mapped: tags to a convention

Found: `Declared` already is a set of side tables beside a plan (`given`, `place_tags`,
`node_tags`), with the policy "ops do not touch them; stale entries are tolerated".
`Static` is the only tag, read only by the jit registration; nothing reads `node_tags`;
processax uses no tags.

Agreed: replace the tag machinery with a convention in `pytree/consumers/declaration.py`
— floats, complex numbers and arrays (and array leaves of any pytree value) are traced;
ints, bools and everything else are static. No escape hatch for a static float. The rule
is JAX-specific and lives only in the JAX layer.

To do together with it (owner's decision): a place-keyed table on the graph for
per-variable static data, so the change is not only a removal. Its shape is undecided and
it has no reader yet once `Static` is gone.

Also to verify: that one place can hold a whole pytree (an `eqx.Module`) as its value;
add a test that differentiates through one.

## 9. Reviewer findings on attributes not acted on

- `implemented(node, fn)` replaces an existing implementation silently outside a plan.
- Refusing an implementation on a bare condition no longer hints "takes a driver first".

## 10. How to run things

- cottax, one run at a time, throttled:
  `cd ~/cottax && nice -n 19 taskset -c 0,1 ~/miniconda3/envs/fullcoupling/bin/python -m pytest -q -p no:cacheprovider --tb=short`
  (about 95 s).
- processax: `JAX_PLATFORMS=cpu`, same throttle, the `-S` + `site.addsitedir` recipe on
  its `.venv`; about 6 minutes.
