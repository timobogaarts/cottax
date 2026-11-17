# Design of experiments: plan (draft)

Branch `design-of-experiments`. This file is removed before the merge; what stays is the code,
its docstrings, a section in `CLAUDE.md` and the lines in `TODO.md`.

## What a DOE is

| | intent (the node) | exit test (the driver) |
|---|---|---|
| DOE | `h_i = F(x_i)` for `x_i = p(i)`, `i = 1..n` | `i = n` |

- `p` is the **plan**: a model upstream writing, per factor, the array `P = (p(1), ..., p(n))`,
  its leading axis `i`. `n` is that axis' length; a plan model takes it as `cx.Static(n)`.
- `x` is a **factor**: a place the model reads, owned by the DOE inside its cycle (one producer
  per place, so `p` cannot own it).
- `F` is the DOE's interior: every function and nested solve on its cycle.
- `h` is a **response**, read on the cycle; `H = (h_1, ..., h_n)` its **record**.

## The node

`DoeCondition(factors: tuple[Varied, ...], responses: tuple[Recorded, ...])`, the pairs as
`Dt` is ODE's: `Varied(factor, plan)`, `Recorded(response, record)`.

| face | holds |
|---|---|
| `owns_inside` | the factors |
| `reads_inside` | the responses |
| `reads_outside` | the plans |
| `owns_outside` | the records |

Core proves it as it is: a response no factor reaches is `Constant`, a plan computed on the
cycle `ReadInsideCycle`, a record read on it `OwnReadInsideCycle`. No `+`: a DOE combines with
nothing. `owns_inside` is derived from the pairs, so `SetOwnsInside` refuses it, as an ODE's.

## Flags

Each is a sentence in the docstring where it lives, labelled *Workaround*, *Pending* or *Work
in progress*, and a line in `TODO.md` (local, as `CLAUDE.md`), added by the step introducing
it.

- **F1, workaround: the records are the node's.** They sit on its `owns_outside`, so the run
  writes a condition's `owns_outside` and an implementation returns them after the unknowns.
  Pending the history rework, which may move them to the driver as reports.
- **F2, workaround: `Recorded` and `^record` stand alone.** A record is a `Trajectory` in
  concept (a place's value over a driver's evaluations, stacked on a leading axis); one class
  needs roles sized per any place (LB-1). Both, and the mint's name, are renamed or unified in
  the history rework.
- **F3, pending check: a DOE's interior is not an answer.** A model off the DOE's cycle may
  read a factor or anything owned on the cycle; after a run it holds the last point. To be
  refused by a check of the condition kind, which needs a hook in core. Pinned by a strict
  `xfail`.
- **F4, work in progress: a DOE is nested in nothing.** Documented, not enforced. A DOE nested
  in another problem proves and runs under core's rules today (robust design, probed), and is
  unsupported until F3.
- **F5: no DOE in the presets.** An architecture refuses a model holding one.

## Steps

One commit each, its tests beside it. **Core: no change.** **Surface: no change** (the part
lives in `doe`'s answer to declaration).

1. **Formulation** (`doe/`, as `ode/`). `doe_condition.py`: `Varied`, `Recorded`,
   `DoeCondition`, `is_doe`. `contracts.py`: the level `Sampler` (`accepts = is_doe`), its
   final `Sampling` (takes nothing, reports nothing), `CONTRACTS`. `consumers/__init__.py`,
   registering nothing yet. `TODO.md`: F1 to F4.
   Tests (`tests/test_doe.py`): a DOE over a toy model proves `ExecutableGraph`, its four faces
   as above; core's three refusals arrive by kind; the contract accepts a DOE and nothing else;
   a relabelling moves a factor, a plan, a response and a record.
2. **Execution.** `consumers/execution/run.py`: a solve writes the condition's `owns_outside`
   after its `owns_inside`; `interface.py`: `DriverImplementation.__call__` returns those
   values (F1; the relational and ODE conditions own nothing outside). `doe/consumers/execution.py`:
   `JaxDoe(batch_size=None)`, `jax.lax.map` over the points (`None` one at a time, `b` vmapped
   `b` at a time), the factors left at the last point (F3); plans of unequal length refused
   when traced (`PlansDiffer`).
   Tests: a toy DOE records `F(p(i))`, alike for `batch_size` `None`, 2 and `n`; a root find
   nested in it records its answer per point; `grad` through a jitted run with respect to the
   plan; `PlansDiffer` by kind; F3's `xfail`.
3. **Declaration.** `doe/mints.py`: `Points` (`^points.x`, where a factor's plan is read
   unless named), `Record` (`^record.h`, where a response's record is written unless named;
   F2). `doe/consumers/declaration.py`: a bodiless `Declaration` (as relational's `Problem`)
   and a part `DoeDeclaring` with one verb, `sample(over=..., record=...)`, each a name or a
   mapping, as ODE's `record=`: `over={'z_1': 'z_1s'}` reads the plan at `z_1s`, a value
   (`over={'z_1': array}`) is given at `^points.z_1`, a bare factor reads it there;
   `record={'f': 'fs'}` writes the record at `fs`, a bare response at `^record.f`. Slot-named
   only (no loose name, as `satisfy`). Joined in `assemblies/jax.py`: `ShippedSurface`,
   `CONTRACTS`, `cx.points`, `cx.record`, `JaxDoe`; `sample` added to `CONDITION_NODES` in
   `tests/test_layering.py`.
   Tests: the declared node equals the one written by hand; Sellar's MDA nested in a DOE over
   `z_1` (`fixed_point_cut`, `nest`, `drive(JaxDoe(), Newton())`) records what a loop of runs
   gives.
4. **Visualisation.** `doe/consumers/visualization.py`: a `Kind` in a hue of its own, the
   node's `NodeDrawing` (its responses and plans as slots). The mints keep the default,
   written around their spelling (`points(x)`, `record(h)`). Test: the kind and slots drawn,
   in every picture.
5. **Export.** `doe/consumers/json.py`: the `Spelling`, as ODE's. Test: a round trip.
6. **Architectures** (F5). `doe/consumers/architectures.py`: a `ModelReading` refusing a DOE
   as no implicit model, with no body, combining as nothing, so a preset meeting one says why
   (without it, the refusal asks for a module that does not exist). Test: `flat.mda()` on a
   model holding a DOE is refused by kind.
7. **Docs.** A DOE section in `CLAUDE.md`; the follow-ups in `TODO.md` (both local).

## Follow-ups, not on this branch

- `Vary`: a DOE's factors added by a later op, beside declaring them, once nesting or
  combining a DOE needs it.
- Per point convergence: a `Converged` report and a non-raising setting on the optimistix
  drivers; the DOE records it as a response.
- Plan models (a grid, a Latin hypercube) as input sugar.
- A DOE in the presets: MDF with a DOE top needs `check_top_combinable` to refuse only what
  it would combine.
- The condition-kind check (F3) and the history rework (F1, F2).
