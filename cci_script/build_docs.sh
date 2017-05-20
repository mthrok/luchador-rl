#!/bin/bash
set -eu

rm -rf docs/source/API/*.rst
sphinx-apidoc -e -o docs/source/API luchador_rl --force
make -C docs html
