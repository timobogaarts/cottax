"""Registry of example notebooks rendered into the docs.

Single source of truth, consumed by two places:
- ``build_docs.sh`` (run as a script here) to know which notebooks to copy
  from their ``*_i.ipynb`` sources into the docs tree, and where.
- ``conf.py`` to exclude ``ci=False`` notebooks from myst-nb execution (they
  keep whatever outputs are committed in the notebook instead).

Add a notebook here when adding one. Outputs are committed only for ``ci=False``
notebooks -- they publish exactly what was last run locally, and
``.gitattributes`` exempts them from the nbstripout filter.
"""

NOTEBOOKS = {
    # docs-tree path (relative to docs/source): {"source": source notebook, "ci": bool}
    "theory/theory_1.ipynb": {"source": "docs/source/theory/theory_1_i.ipynb", "ci": True},
    "theory/theory_2.ipynb": {"source": "docs/source/theory/theory_2_i.ipynb", "ci": True},
    "theory/theory_3.ipynb": {"source": "docs/source/theory/theory_3_i.ipynb", "ci": True},
    "theory/theory_4.ipynb": {"source": "docs/source/theory/theory_4_i.ipynb", "ci": True},
    "theory/theory_5.ipynb": {"source": "docs/source/theory/theory_5_i.ipynb", "ci": True},
    "theory/theory_6.ipynb": {"source": "docs/source/theory/theory_6_i.ipynb", "ci": True},
    "examples/implicit_sellar.ipynb": {"source": "docs/source/examples/implicit_sellar_i.ipynb", "ci": True},
    "examples/coupled_ode.ipynb": {"source": "docs/source/examples/coupled_ode_i.ipynb", "ci": True},
    "examples/non_jax.ipynb": {"source": "docs/source/examples/non_jax_i.ipynb", "ci": True},
}

NON_CI_NOTEBOOKS = [docs_path for docs_path, meta in NOTEBOOKS.items() if not meta["ci"]]

if __name__ == "__main__":
    # Emitted for build_docs.sh: one "<source>\t<docs_path>" pair per line.
    for _docs_path, _meta in NOTEBOOKS.items():
        print(f"{_meta['source']}\t{_docs_path}")
