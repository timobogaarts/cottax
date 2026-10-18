# Consumers and features

Why cottax is split the way it is, and how the split works.

## The problem

cottax grows along two axes. A **feature** is a new thing a graph can be made of: a
language of conditions (`relational`), a kind of place (`pytree`). A **consumer** is
something that reads a graph: running it, writing it to JSON, drawing it, declaring it. Every pair has code: how relational is drawn, how a pytree place
is written to JSON.

A directory tree keeps only one axis together. Cut by consumer alone, adding a language
means editing every consumer, and you have to know where each one hard-codes the old
language. That is what made time dependence impractical.

## The split

```
cottax/
  core/         what holds with nothing chosen: no place, no language, no way of running
  relational/   a feature: a language            consumers/  what it tells each consumer
  ode/          a feature: a language            consumers/  what it tells each consumer
  pytree/       a feature: a place               consumers/  what it tells each consumer
  consumers/    execution/  json/  pickle.py  visualization/  surface/  declaration/  architectures/
  assemblies/jax.py     the shipped assembly: the one module naming the features, a study's
                         import
```

Three rules place everything.

1. **A consumer's code knows no feature.** It is written against core and its own
   `interface.py`, and names no feature anywhere; what its `__init__` exports is for a
   study, and its refusal kinds are in `refusals.py`. A consumer uses another through
   `interface`, `interface_types` and `refusals`; another's `__init__` is a study's. A
   consumer in `USES_NONE` (declaration) imports no other. A feature's
   `consumers/__init__` imports its answers, so they register; the assembly imports it.
2. **An answer lives with the feature it is about.** How relational is drawn is
   `relational/consumers/visualization.py`. It has the name of the consumer it answers, and
   is one module: a file, or a package whose `__init__` exports what the file would. Support
   several answers share imports no consumer.
3. **Nothing imports the assembly.** A study does: `import cottax.assemblies.jax as cx`.
   It defines only the joined surface, `Scope`, the joins between consumers for the
   shipped pair (`drive`, `run`, `mdf` / `idf` / `sand` / `mda`) and the
   conveniences over exact names (`node`, `minted_node`, `cut`, `close`, `close_together`,
   `assign`, `implement`); the rest is imports and defaults. A surface declares and offers
   no op: it exists only because a place written as a string needs a root.

Why the answer sits with the feature and not with the consumer: when a consumer changes
what it asks, every stale answer is refused when its class is created, so the machine finds
them. When a language changes, nothing refuses a reader that forgot the new field; it
misreads silently. So the readers of one language are kept together, where a person will
see them.

## What a consumer asks

Core declares the minimum of a node (its reads and owns) and of a place (its segments).
Each consumer needs more, and says so in two files:

- `interface.py`: how a feature gets in, classes alone. Every class there extends `Answer` or
  `Choice` (`core/frozen/answer.py`), in a section each.
- `interface_types.py`: what its answer is handed, returns or builds on.

| consumer | registered (`Answer`) | chosen (`Choice`), and where |
|---|---|---|
| `execution` | `DataStructure`, of a place | `DriverImplementation`, per contract in `ImplementDrivers` |
| `json` | `Spelling`, of a node; `PlaceCodec`, of a place; `LabelCodec`, of a node name | |
| `visualization` | `NodeDrawing`, of a condition node; `PlaceDrawing`, of a place | `Formatter`, a drawing's `formatter=` |
| `declaration` | `DerivedNames`, of a place | `Names` and a vocabulary (a level under `Surface`), in the surface built in the assembly or by `Scope` |
| `architectures` | `ModelReading`, of a condition node | `Closing`, an architecture's second argument |

What a vocabulary's kinds extend, `Declaration` and its `Wrapper`, is in `interface_types.py`.

An `interface.py` holds what a consumer asks about a node or a place. A consumer's own
options (a drawing's `Style`, JSON's `implementations=`) say nothing of either: they are
its study API, not a `Choice`.

Each is optional. A node that answers none of them is still a valid node: the graph is
built, rewritten and proved. The drawing shows it grey; JSON refuses it and says which
class has no answer.

## Registered, or chosen

- **An `Answer`: one right answer per class.** How a node is drawn or spelled has one
  answer, so the feature writes a final and registers it, `@answers(*classes)`, when its
  module is imported. `answer(interface, obj)` finds it along the class's ancestry, most
  specific first; a graph being over one place class and one label class,
  `answer_for_places(interface, graph)` / `answer_for_labels` asks it of one place / node
  name. A second answer for one class is refused, and so is a final leaving a
  method abstract. With none, the consumer decides: a default (the drawing's grey problem,
  a place's `str`) or a refusal naming the class. `answered()` is the table of what is
  answered. `answer_for_class(interface, cls)` is the same with the class in hand (a
  surface finds `DerivedNames` by the place class its `Names` writes). Importing a feature's
  `consumers` registers its answers (the assembly does).
- **A `Choice`: several fit.** Which solver answers a problem, how a name is written, which
  vocabulary a study declares in: none is registered (`@answers` refuses one), and the study
  names one in the slot its interface class states in a `Handed in as:` block, an example of
  the call. Each final on offer is marked
  `@choice(interface)` where it is defined: the mark registers nothing, so a choice is found
  by searching, as an answer is by `@answers`. A registered answer is never handed in: a
  consumer finds it.

## What an addition takes

**A language**: its node and predicates in its own package; then, one consumer at a
time, a module under its `consumers/` implementing that consumer's `interface.py`. No
consumer is edited. `ode/` is the second: it took one move in a consumer, `Paired` (a body
and a condition node beside it) from relational's declarations into declaration's
`interface_types`. It also took
the architectures out of relational into a consumer of their own: what they ask of a
problem is three questions (`ModelReading`): whether it is an implicit model and what its
body is are each language's own picture (an ODE's body is its whole cycle, since a closed cut
there changes its answer), and only what concerns several problems is the architecture's. No
assembly holds it: a study joins `Basic`, `Relational` and `Ode` itself and drives from both
features' `CONTRACTS`.

**A place**: the same, for the "of a place" half.

**A consumer**: a package under `consumers/` with its `interface.py`; then an answer in
each feature that should support it.

**A solver, a declaration kind**: one class, where the language keeps them.

## What keeps it true

`tests/test_layering.py` reads the import graph and pins the rules above, and that
`import cottax` loads nothing. Imports are not the whole story: a consumer could read an
attribute it never asked for. What catches that is running each consumer on a toy node and
a toy place that have nothing but core's interface and their answers: `tests/test_answers.py`
takes one toy language and one toy place through all of them.

## What it costs, and what it does not cover

- More files, and registration happens on import: an answer exists only once its module
  is imported.
- A fact about one variable (a unit, a shape) or one node is neither a node nor a place. It
  is carried as a tag beside the plan, not through these interfaces.
- A feature that changes core itself (staging, storage) is not helped by any of this:
  every consumer has to learn it.
