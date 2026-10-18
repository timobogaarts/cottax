# cottax

Python package for computational graph generation using JAX, enabling graph manipulation and multidisciplinary optimization.

**Documentation, theory and examples: https://timobogaarts.github.io/cottax/**

## Installation

cottax needs Python 3.12 or newer.

```bash
pip install cottax                  # the core: jax and networkx
pip install "cottax[solvers]"       # + optimistix and scipy, needed by `cottax.assemblies.jax`
pip install "cottax[solvers,ode,viz]"   # + diffrax for ODEs, and the drawing dependencies
```

From source, with everything needed for the tests and the docs:

```bash
git clone https://github.com/timobogaarts/cottax.git
cd cottax
pip install -e ".[dev]"
JAX_PLATFORMS=cpu pytest
```

To get started, see the [Sellar example](https://timobogaarts.github.io/cottax/examples/implicit_sellar.html).

## Why the Frozen base?

This package is designed to ensure that the graphs it builds are *always* valid, and the operations on them preserve that validity. 
These graphs are therefore immutable: any change to a graph must prove that the new graph is actually valid too.
Furthermore, they are additive checks: every additional level (e.g. executable/runnable) only *adds* checks, never removes them.
This only holds if no subclass accidentally overrides checks or change what a check reads. Python is rather permissive here, so to make that enforced 
rather than convention, no implemented method may be overridden in a subclass. 
You may extend such a class with additional methods (abstract or not), or add additional rules and checks though.

The `Frozen` base class implements this pattern; it is additionally used for the nodes, since they carry the same proof obligation.

