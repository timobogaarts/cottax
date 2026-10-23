# cottax

Python package for computational dependency graph declaration, manipulation, and execution.


**Documentation, theory and examples: https://timobogaarts.github.io/cottax/**

Theory and application discussed at https://arxiv.org/abs/2610.22687

## Installation

cottax needs Python 3.12 or newer.

```bash
pip install cottax                  # the core: jax and networkx
pip install "cottax[solvers]"       # + optimistix and scipy, needed by `cottax.assemblies.jax`
pip install "cottax[solvers,ode,viz]"   # + diffrax for ODEs, and the drawing dependencies
```
