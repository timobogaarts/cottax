#!/usr/bin/env bash
set -euo pipefail

# ./build_docs.sh              -- execute CI-runnable notebooks fresh (matches CI)
# ./build_docs.sh --no-execute -- skip execution, render whatever outputs are
#                                 already sitting in the docs copies (fast
#                                 local iteration on prose only)
if [ "${1:-}" = "--no-execute" ]; then
  export COTTAX_DOCS_NO_EXECUTE=1
fi

cd "$(dirname "$0")"

# Each notebook's source is a `*_i.ipynb` under docs/source; it is copied beside
# itself without the suffix, where myst_nb renders it as a page. The copies are
# generated, hence gitignored -- edit the `_i` source.
# The list of notebooks and their destinations lives in
# docs/source/_notebook_manifest.py (single source of truth, also used by
# conf.py) -- add a notebook there, not here.
copy_notebook() {  # <source notebook> <destination inside docs/source>
  mkdir -p "$(dirname "docs/source/$2")"
  cp "$1" "docs/source/$2"
}

while IFS=$'\t' read -r src dest; do
  copy_notebook "$src" "$dest"
done < <(python docs/source/_notebook_manifest.py)

cp assets/cottax-logo.svg docs/source/_static/cottax-logo.svg
cp assets/cottax-logo-dark.svg docs/source/_static/cottax-logo-dark.svg
cp assets/cottax-icon.png docs/source/_static/cottax-icon.png

python -m sphinx.ext.apidoc \
  -o docs/source/api \
  src/cottax \
  --force \
  --module-first \
  --separate\
  --no-toc \
  -t docs/source/_templates \
  -d 2

# depth of 2 for module documentation


python -m sphinx -b html docs/source docs/build/html
