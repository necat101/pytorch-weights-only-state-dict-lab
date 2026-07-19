#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (directory containing this script)
cd "$(dirname "${BASH_SOURCE[0]}")"

# Find python
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: python / python3 not found in PATH" >&2
    exit 1
fi

echo "== pytorch-weights-only-state-dict-lab =="
echo "Python: $($PYTHON --version 2>&1)"
if $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    :
else
    echo "PyTorch: not installed (torch-dependent cases will framework_skip — this is normal)"
fi
echo

echo "--- running lab ---"
$PYTHON run_lab.py
echo

echo "--- running tests ---"
$PYTHON -m unittest -v
echo

echo "Done. See observations.json / observations.csv / RESULTS.md"
