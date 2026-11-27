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
   - `FixedAndMoved` (relational) deleted: unreachable, `ReadsOwn` / `ReadOnBothSides`
     cover it. Theorem tests (step 1): deferred until memo seeding.
3. **`plan/` redesign: design agreed 2026-11-26, not implemented.** From two designers'
   debate and three blind reviews (the last a neutral one: today 5.5/10, this 8.0, ~8.5 with
   the fixes folded in below).

   *Today's problems it fixes (all reproduced by probes):* `refusals` raises raw `KeyError`s on
   unbound names (Rename, Combine, Implement, Cut, Assign, ...); `refusals` gives `[]` while
   `apply` raises (Replace / Nest on unbound names, FixedPointCut on a bound `at`,
   ImplementDrivers({}), Drive's NoneAccepts, ClosuresCollide, NotCombinable, AssignByRule's
   inner DrivenTwice); a 1500-op plan rebuilt cold hits RecursionError through `trunk`;
   `Monolithic.outside` resolves on a subgraph then re-applies (MDF on `upstream()`: 22
   applies); Contract builds three graphs; derived-first order needs the rewrites.py guard;
   ~9 framework concepts for op authors.

   *The design:*
   - **`Edit`**, a Frozen value, the only thing that builds a graph: `unnest`, `remove`,
     `redefine` (bound, in place), `insert` (free, appended), `nest`; applied in that order.
     A misfit (redefine/remove unbound, insert taken, a pair losing one end, unnest of an
     unnested node) raises `EditMisfit` (an `AssertionError`): a bug, never a refusal; every
     misfit a user can reach has a check that fires first. No `relabel` / `attributes`
     fields: `node.relabel(...)` / `dataclasses.replace` keep attributes. No lineage field
     yet (see §6).
   - **Two op kinds**, abstract on `Op`: `Leaf.edit(graph) -> Edit`; `Composite.steps(graph)`,
     a generator yielding a stage's ops and receiving the graph they leave (a plain list is
     one stage; Architecture's five stages are five yields). `steps` final on Architecture;
     its finals bind hooks only (Frozen's no-override rule).
   - **Field roles** `Bound[T]` / `Free[T]` (Annotated; shape read off `T`): the framework
     reports `NotBound` / `NameTaken`, so later checks may index those names.
   - **Tiers**, each asked once the one before holds: 0 `keeps.check(input)` (Valid by
     default) then the roles; 1 `checks_named` (facts of the named nodes); 2 `checks_graph`
     (the graph at large); 3 `Valid` then `keeps` of the result. Inside a tier the house
     order: root level first, a level once the levels above hold, siblings together. Gives
     `MDF(top=D1)` -> `[TopNotAProblem]`; the Determine guard and `check_*_bound` /
     `TopUnbound` go. Known cost: a derived check more fundamental than an inherited one in
     the same tier must move a tier earlier (e.g. Monolithic's `top_combinable` can be
     hidden by Architecture's `implicit_models`; accepted).
   - **Placement rule:** what the input graph decides is a check. Where a check and `steps`
     need the same decision (Drive's choices, AssignByRule's places, `closed`), both call one
     helper and the helper `refuse`s; the walk catches `refuse` in the tiers too, so the
     decision is written once. `refuse` mid-`steps` only for what an intermediate decides.
   - **One walk**, `attempt(op, graph, keeps=Valid) -> Applied(op, graph, stages) |
     Refused(violations, path, stages)`. Every step of a composite is walked the same way, so
     every intermediate graph is proved (probed: 407 tests in 16 files, no invalid
     intermediate, negligible cost); `keeps=None` deleted. `apply` raises the first refusal
     (the Frozen idiom); `refusals = attempt(...).violations`, empty exactly when `apply`
     succeeds; `Resolution` a typed view of the stages (`outside` flattens the analysis's
     own stages). No cheap mode (nothing needs one).
   - **What is a refusal:** a tier's violations, a step's `Refused`, `refuse(...)` (raises its
     first violation's own error with a marker only `refuse` sets, so a helper called
     outside a walk still raises a plain ValueError), and `built(cls, ...)` (constructs a
     step op from graph data via `violations(cls, ...)`, refusing with what it finds).
     Nothing else is caught: a nested `.apply` failing in a helper is a bug.
   - **Plan:** `base`, `ops`; `graphs` a `cached_property` built by a loop (no `trunk`, no
     recursion; not pickled, not in eq/hash). `__add__` primes the new plan's memo (as today)
     and also takes an `Applied` (hence `Applied.op`) so `attempt` then `+` walks once.
   - **Contract becomes a function**, `contracted(graph, blocks) -> Graph`: one Edit, boxes
     carry no attributes by definition (a coarsening for analysis, never in a plan); its own
     rules raise. `cuts_per_cycle_boxed` calls it; 16 test sites change from
     `Contract(b).apply(g)`. `graph.nodes` order, not `_position`.
   - **Monolithic.outside** yields one step: the analysis (a user-given one rebuilt with
     `dataclasses.replace`) with `apart=cycle_of(top)`; Coupled's `scope` and
     `check_implicit_models` read the graph without `apart` while its ops apply to the whole
     graph; an analysis whose `apart` covers everything yields nothing (as today's
     `if not off.nodes`). Probed sound (`cycle_of(top)` is a whole cycle). `apart`
     documented as Monolithic's. Only two tests exercise it: add some.
   - **Op mapping:** Insert `Free`; Delete `Bound`, Stranded named; Replace `Bound`, taken as
     given; Nest / Unnest `Bound`, AlreadyNested / NotNested named, a loop left to `Valid`;
     Rename / Cut `Bound`, NotRenamed / NotProduced / DoesNotRead named (Cut no longer a
     Rewire: `test_core_ops:147` drops `NotRenamed`); Set/RemoveAttribute, Implement,
     Assign, Undrive, SetOwnsInside (Determine) leaves; Combine one Edit (redefine `at` if
     bound, else insert); FixedPointCut `at: Free`; AssignByRule / Drive / ImplementDrivers /
     CloseFunctionalCycles composites; Architecture as above.
   - **Deleted:** Primitive, Rebind, Rebinding, Redefine + `edits_attributes`, Rewire,
     ComposedOp, StagedOp (Stage only as trace data), `per_level(derived_first=True)`,
     `keeps=None`, `Plan.trunk`, `check_*_bound` / `TopUnbound`, the order guards
     (relational/rewrites.py:26,33), Monolithic's re-apply, the Contract op, `Nest`'s own
     loop walk. Still open, settle in Cut's rewrite: `Cut.copy_of` is a callable inside a
     record meant to hash and pickle.
   - **Pin with a property test:** over every (op, graph) the suite builds, `attempt` never
     raises and `refusals` is empty exactly when `apply` succeeds; no shipped composite
     produces an invalid intermediate from a valid input (a step's tier-3 failure is
     reported as a refusal, so this is what keeps library bugs from posing as user errors).

   *Churn (the neutral review's count):* ~50 `refusals(` assertions in 13 test files (NotBound
   moves to tier 0, NotRenamed goes); 14 readers of `.composed` / `.staged` / `.resolution`;
   16 `Contract(...).apply`; tests calling `check_*` directly (e.g. test_core:298);
   `Plan.trunk` read by `consumers/visualization/page.py:237` (`prefixes`), test_cache:159,275,
   test_plan:103-113 -- rewrite on `graphs`. test_mdao_architectures:489 matches NotBound's
   wording.

   *Implementation order:* (a) `plan/` itself with the property test; (b) the ops outside
   `plan/` and their tests; (c) Monolithic.outside with `apart`. Uncommitted; targeted test
   files after each.
   - Settled in phase 1 (2026-11-26): `Edit`'s misfits are only what makes each field mean
     one thing (remove / redefine / unnest of something absent, insert of something present,
     nest of an inner still nested, a name in two of remove / redefine / insert). Graph rules
     are not restated there: a pair naming a removed node is kept and `Valid` refuses it
     (`Unbound`); an op removing both ends unnests them itself (`Delete` does). Keeping
     attributes is `relabel`'s contract (`renamed`, which restored them, is gone).
   - **Step 1 done (2026-11-26), the stripped core** (decided with the owner after two review
     rounds and a blind review of HEAD / the first phase-1 core / this one, scored 5 / 6 / 8).
     The premise: graphs change only through ops; an op is a named, checked change; a plan is a
     graph plus the ops that made it. What nothing outside `plan/` read is gone: `attempt`,
     `Applied` / `Refused` / `Stage`, the generator protocol, `refuse` / `built`, `__call__`.
     - Then cut to the minimal plan (owner: "an entire new convention for a non-existing
       problem"): no `Refusal` type, no `OpBug`, no `@final` (Frozen enforces abstract-or-final),
       no refusal path, no conftest watch or pin, no `refuse_quoted_roles`, no `Nested` /
       `Unnested` (`Nest` / `Unnest` have one small check each). The refusal vs bug distinction
       only matters to op authors; not a core concern.
     - A refusal is the first violation raised, carrying them all as `violations`: `frozen.first`
       now attaches every violation it found (as Frozen construction already did).
       `refusals(op, graph)` = `apply`, catching what carries `violations`.
     - `edit.py`: `Edit.applied` refuses (`first`) where it does not fit the graph, with the roles'
       violations and `NamedTwice`: safe on any graph, so Edits can be replayed.
     - `op.py` (was `algebra.py`): `GraphOp` -- `needs` (`AnyGraph`, a new empty rule group in
       `maths/rules.py`, by default; `Valid` for `Cut` and every legacy base), `checks_graph`,
       `apply(graph, keeps=Valid)` and `edits(graph, keeps=Valid)`, abstract `changed`. `Leaf.edit`,
       `Composite.ops`, `in_turn`. `keeps` is checked of the result only; steps keep `Valid`.
     - `plan.py`: ops only; `+` applies, a refused op never enters. `legacy.py` on this core.
     - Tests: op calls are `.apply(...)` (processax has none); new test_plan tests (edits as data,
       Edit refusing, `needs`, `keeps`); test_combine's join case now expects `[ReadsOwn]` (the
       join's refusal, as data). Same failures as before: test_incremental's stale `owners` (58);
       MDF xfail. Lines: plan/ 739 without legacy.py (HEAD 726), 878 with it.
     - Roles removed (owner + blind review): `roles.py` deleted. `Graph.__getitem__` refuses a
       missing name with `NotBound` (now in `maths/graph.py`); `Edit` refuses misfits incl. nest
       ends left unbound; `_walked` notes every refusal `refused by <op>`, innermost first.
       `.violations` = all of the first refusing step. No machinery whose only job is keeping
       op-level collection total. Phase 2: delete `architectures.py:262` (`first(refusals(...))`).
     - A graph need not be `Valid` between ops (owner: a model-repository graph cannot be).
       `apply`/`edits`/`refusals` take no `keeps`; `in_turn` keeps nothing. `Plan.keeps`
       (default `AnyGraph`) is checked after each op; assemblies ask `Valid`. `Edit` refuses
       only what it cannot do as stated; an unbound nest end is `Valid`'s `Unbound`. The
       `Nest`/`Unnest` checks went (the edit refuses the same). test_cx slot test xfailed
       beside the MDF one: same cause (Architecture's level checks run before the top's).
     - Public means checked: `apply`, `edits`. Author hooks private: `_changed`, `_edit`, `_ops`
       (`_with_edits` for `in_turn`). `refusals` moved to `tests/graphs.py` (no src user).
       `Leaf` -> `PrimitiveOp`; `primitive.py` -> `bindings.py`, then `bindings.py` + `ops.py` -> `primitive_ops.py`. `checks_graph` typed with `Self`.
       Later: `Composite` -> `ComposedOp` once legacy's `ComposedOp` is gone.
   - **Step 2 (phase 2), decided:** port the ops outside `plan/` (`composed` -> `ops`; Rebind /
     Redefine / Contract users -> `Leaf`s, `contracted`); `Combine` a `Leaf`; Architecture's `needs` its own rule group (a problem graph with its top: fixes
     MDF(top=D1), un-xfail test_top_names_a_bound_problem), stages as plain code over a small
     multi-stage helper, `Resolution` gone (its ~7 tests read the result and edits);
     `Monolithic.outside` replays the analysis's Edits through a checked `replayed()`, its
     precondition (a self-contained subgraph) stated;
     delete `legacy.py`. Then phase 3, `apart`.
   - Roadmap ideas from the blind review (none built): copying a node or subgraph over an index
     (collocation, multiple shooting, multi-point design); rebasing a plan onto an edited base;
     provenance (node -> op that made it); "what applies here?"; saving / loading a plan.
   - Open idea, no reader yet: `keeps` as anything with `check(graph)`, so a value holding
     the original graph works, e.g. `SameProblem(base)` refusing an `opt` nested in a
     residual that enclosed it in `base`. Not memoised (`remembered` keys are kinds); no
     separate `preserves` needed.
   - Declined (do not reopen): `keeps` threaded through `Plan`; a cheap `refusals` mode;
     lineage inferred from a relabel; catching any exception carrying `.refused`;
     a phase DAG for check order; `Edit.between(before, after)` (a cause rebuilt from its
     effect, and not always one Edit); composing Edits into one; `steps` yielding Edits;
     per-stage records and refusal traces as result objects (no reader).
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

**Place attributes: a side table on the base graph (decided), its shape open.** Owner-held
attributes (on the node owning the place) were considered and rejected: who owns a place is
itself edited -- determining an unknown (`Determine` / `SetOwnsInside`) moves ownership, as do
Cut and Combine -- so attributes would have to follow ownership changes, and free inputs have
no owner. A side table keeps a place's attributes independent of who owns it. Open for when the
first reader (units) lands:
- Shape: a `place_attributes` field on `Graph` (place -> attributes, like a node's), not a
  subclass; per-kind policy from the attribute kind, as for node attributes.
- Lineage: ops state it in their Edit (a new field then), never inferred from a relabel. A
  move (Rename, a relabel of every mention) and a copy (Cut: the readers get `copy`, the
  producer keeps `var`) are different and must be written separately (`moves` / `copies`),
  since renaming one node's read of `z` while another still reads it is a copy.
- Which kinds an op carries: e.g. a unit carried by every op that moves or copies a place,
  other kinds perhaps not; decided by the attribute kind, not by each op.
- Drop rule: what happens to a place's attributes when no node mentions it any more (Delete,
  Combine); stated per op or collected once when a plan ends -- not per Edit, or Delete +
  Insert in two steps would differ from one Replace.
- Conflicts: two places merged (Combine, a rename onto an existing place) with differing
  attributes -- refuse, or a kind-level rule.
- Stale entries: whether `Valid` (or a separate group) refuses an entry for a place no node
  mentions.

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
