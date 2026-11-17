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
- `WellFormed` is also recorded on the `definitions` map, so `Nest` / `Unnest` do not re-run it.
- `subgraph` carries the `NodeRules` groups and `WellFormed`; `contract` carries
  `WellFormed` only (a boxed `FunctionalNode` has no implementation).

| Group | Kind | Rules | Implies |
|---|---|---|---|
| `WellFormed` | whole graph | one producer per place, exact reads | — |
| `ExecutableGraph` | whole graph | the seven executable rules | `WellFormed` |
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
| `dependencies(graph)` | `WellFormed` | `owners`, edges, `components`, `cycles`, `topological_order`, `ancestors`, `descendants`, `reach`, `boundary_inputs`, `boundary_outputs`, `feedback_variables`, `closing_readers` |
| `nesting(graph)` | `WellFormed` | `cycle_of`, `interior`, `contract` |
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
- `GraphOp.apply(graph, keeps=WellFormed)` proves `keeps` of the input, asks the intent
  checks, applies, proves `keeps` of the result. A composed op proves `keeps` once.
- `Plan(base, ops, keeps=WellFormed)` proves `keeps` of its base and passes it on, also
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
2. **Flat storage and views.** `WellFormed` becomes lazy, `keeps` on `apply` and `Plan`,
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
