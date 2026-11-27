# cottax hand-off — 2026-11-27 (maths core and plan/ done)

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

- Branch **`tower`**, committed by the owner up to `6bbd9de` (driver rework + plan rework,
  phase 1). Uncommitted: phase 2 of the plan rework (section 3.3), and the owner's own
  driver work (`conditional/driver.py`, `roles.py`, the `ode/` / `relational/` contracts,
  `consumers/execution/`). Leave `consumers_and_features.md`'s deletion alone.
  `core/plan/bindings.py` is a stale duplicate the owner's commit restored: nothing imports
  it; delete when the owner says.
- `main` has PRs #7 (condition node's two faces), #8 (node attributes), #9 (driver as a field).
- **processax** (`~/processax`) is the owner's: never edit it, only report breaks. Known
  breaks (owner to fix): `benchmark/structure.py` imports `Loop` (:9, now `Enclosed`) and
  uses `step.interior` as a graph (:24, :40-41); `Dependencies.of(..).owners` -> `.owner`
  (`process/study.py:112`, `process/graphs/kinds.py:109,118`,
  `preprocess/tests/test_coupled_reads.py:38,56,68`). Not yet checked against the plan
  rework (ops' hooks are now `_edit` / `_ops`; `ComposedOp`, `Redefine`, `Rebind` gone).
- Known failing: `test_incremental`'s stale `owners` (58 cases of two tests). Fix or xfail
  before a PR.

## 2. The maths core as it now is

One stored value, facts as views, rules as groups, edits as ops (`plan/`).

- **`Graph(definitions, within)`** (`graph.py`): checks nothing at construction. Literal facts
  true of any graph: `nodes`, `variables`, `owned_variables`, `read_variables`, `readers` and
  `owners` (place -> tuple of nodes), `boundary_inputs` / `boundary_outputs` (set
  differences). Projections (a graph to look at, not an edit): `subgraph(keep)` and
  `contracted(blocks, box)` (blocks: box name -> members; `box(members, reads, owns)` the
  caller's node), both `@checked`; `boxed` is contraction's shared logic.
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
- **Ops** (`plan/`, section 3.3): an `Edit` is the only builder of a graph; an op is a
  `PrimitiveOp` (`_edit`) or a `Composite` (`_ops`); a `Plan` is a base plus ops, asking
  `keeps` (default `AnyGraph`) of every graph it holds.
- Blind review (Fable, old maths vs new, unlabelled): old 5/10, new 8.5/10. It withdrew
  "the old type tower made invalid graphs unrepresentable" as a net virtue (that needs the
  wrapper tower, which does not compose). What a 9 needs is in section 3.

## 3. Next, in order

1. **Theorem tests** (small, from the review): a subgraph of a `Valid` graph is `Valid`; an
   interior of an `Executable` graph is `Executable`. Both are relied on (`Nesting._cycles`
   re-proves per level; `Schedule` proves each interior as its view's check) and would fail
   late, out of a cached property, if a future rule broke them.
2. **`frozen/` review: done 2026-11-25.** Levels asked as rule groups (`unchecked`,
   `_carried`, `skip=` gone); a check returns an iterable; pickle and copy rebuild through the
   constructor; `checked` reads its signature once and reads an iterator argument once into a
   tuple (so the checks and the body see the same values); `answer.py` registrations refuse
   with plain `TypeError`s.
3. **`plan/` rework: done 2026-11-27** (phase 1 committed in `6bbd9de`; phase 2 uncommitted).
   Premise: graphs change only through ops; an op is a named, checked change; a plan is a
   graph plus the ops that made it.
   - **`edit.py`**: `Edit(unnest, remove, redefine, insert, nest)`, applied in that order, the
     only builder of a graph. Refuses only what it cannot do as stated (`NamedTwice`, `NotBound`,
     `NameTaken`, `NotNested`, `AlreadyNested`). Replaying an Edit reproduces its effect;
     re-applying the op its intent.
   - **`op.py`**: `GraphOp` -- `needs` (a rule group, default `AnyGraph`; an op reading a view
     says `Valid`, e.g. `Cut`, `Architecture`), `checks` (op alone), `checks_graph` (root level
     first, the first refusing level only). Public `apply` / `edits`; private hooks `_changed`,
     `_edit`, `_ops`, `_with_edits`. Every refusal is noted `refused by <op>`, innermost first.
     Two kinds only (a test pins it): `PrimitiveOp._edit -> Edit`, `Composite._ops -> ops`
     (`in_turn`). An op decided in stages nests `Composite`s, each decided off the graph the
     last left, reading earlier results off it (by mint) or handed them as a field.
   - **`plan.py`**: `Plan(base, ops, keeps=AnyGraph)`; `keeps` asked of the base (`checks`)
     and after every op; `graphs` built by a loop; `+` applies, a refused op never enters.
     A graph need not be `Valid` between ops (a model repository never is); assemblies ask
     `keeps=Valid`.
   - **`primitive_ops.py`**: Insert, Delete (refuses stranding a pair: `Stranded`), Nest
     (refuses an unbound end), Unnest, Contract (`Graph.boxed` as an edit; a box at a new name
     is appended, as `contracted` places it), Replace, Rename, Cut, Set/RemoveAttribute.
     Refusals from intent sit on the op, not on `Edit`. Keeping attributes is `relabel`'s
     contract.
   - **Outside `plan/`**: AssignByRule, ImplementDrivers, Drive, FixedPointCut,
     CloseFunctionalCycles are `Composite`s; SetOwnsInside, Assign, Undrive, Combine are
     `PrimitiveOp`s (`assigned_by_rule` shared by AssignByRule and Drive). `Architecture` is a
     `Composite` over private stages `_Outside` -> `_Cutting` -> `_Placing(resolved)`;
     `cutting` / `placing` hold the decisions, which `resolution` walks too. Its design checks
     (`check_design_bound`, `check_design_is_problem`) run first. `legacy.py` deleted.
   - `refusals(op, graph)` lives in `tests/graphs.py` (no src user).
   - Blind reviews: vs the previous rework 8/10; three-way (delta-proof version / previous
     rework / now) 4.5 / 6 / 7, the rest held by choices: no public `refusals`, no stage
     records, no delta checks.
   - **Open:** `Resolution` gone and `Monolithic.outside` replaying the analysis's Edits (its
     precondition, a self-contained subgraph, stated); then phase 3, `apart` (Monolithic.outside
     as one step over `cycle_of(top)`; only two tests exercise it). Small: rename `Composite`
     -> `ComposedOp`; fold `in_turn` into `Composite`; `Plan` storing edits once something
     reads them; typos at `op.py:21,54` and `Edit`'s ordering docstring.
   - Roadmap ideas (none built): copying a node or subgraph over an index (collocation,
     multiple shooting, multi-point design); rebasing a plan; provenance (node -> op); delta
     checks per rule over an Edit's touched names (in `Plan`, a rule's optional delta, full
     check as fallback); `Edit.inverse(before)`; `keeps` as anything with `check(graph)`.
   - Declined (do not reopen): field roles (`Bound` / `Free`); tiers and `attempt` /
     `Applied` / `Refused`; a generator `StagedOp`; caching `apply` / `edits`; `Edit.between`
     and composing Edits (reopen only deliberately, e.g. for deltas or inverses); a cheap
     `refusals` mode; lineage inferred from a relabel; collecting `Contract`'s refusals
     before applying; enforcing a named `box`.
4. **The rest of core**, by the sheet (next: `conditional/`, the owner's), then `mints.py`,
   `realisation/`, `answers.py`, `refusal_kinds.py`.
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
  restate a type annotation do not belong in core (nor runtime type guards: a type checker
  catches those).
- Docstrings: one line of prose for helpers; `Args:` / `Returns:` / `Refuses:` for public
  functions with several arguments or refusals. New or touched docstrings only; no
  conversion pass inside a feature branch.
- Node checks stay on the node when a node breaking them is never a meaningful mid-plan
  state; relational and per-stage rules go to a group.
- A view holds only facts of the graph alone; facts true of any graph sit on `Graph`.
  No forwarding of view facts onto `Graph`; bind `deps = Dependencies.of(g)` locally.
- Proofs are not carried to subgraphs (each new value is proved when asked; cheap). Seeding
  the memo by a theorem is possible later, only after the theorem tests exist.
- A failing group runs its rules twice (accepted). networkx stays in `Dependencies`.
- `Rules.check`'s `graph if isinstance(graph, Graph) else graph.graph` stays: views'
  `checks` hand the check the view itself.
- Why views and rule groups, not a type tower (wrappers or subclasses that only add facts):
  nearly the same design; the choice is who converts. Typed, every caller proves first
  (`run(ExecutableGraph(g))`); with views the function that needs it asks (`Schedule.of(g)`
  inside, `@checked(Executable.check)` declaring it). To a user the graph is the same graph,
  shown to be executable, so `run(graph, values)`; a proven type is boilerplate at every call
  site, and ops return plain graphs anyway. Remembering is only what makes asking again cheap.
  A graph type checking at construction also cannot hold a repository or a graph mid-repair.
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
