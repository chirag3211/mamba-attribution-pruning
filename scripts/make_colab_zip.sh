#!/usr/bin/env bash
# Build mamba_grid.zip for Colab: the scripts the notebooks expect, in a folder called "mamba_grid".
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/mamba_grid && mkdir -p build/mamba_grid
cp src/*.py tests/*.py build/mamba_grid/
(cd build && rm -f ../mamba_grid.zip && zip -qr ../mamba_grid.zip mamba_grid)
rm -rf build
echo "wrote mamba_grid.zip - upload it to /content on Colab, together with one notebook from colab/"
