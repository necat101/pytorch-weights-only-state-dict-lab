#!/usr/bin/env zsh
set -euo pipefail

# cd to repo root (directory containing this script)
cd "${0:A:h}"

# Find python
if (( $+commands[python3] )); then
    PYTHON=python3
elif (( $+commands[python] )); then
    PYTHON=python
else
    print -u2 "Error: python / python3 not found in PATH"
    exit 1
fi

print "== pytorch-weights-only-state-dict-lab =="
print "Python: $($PYTHON --version 2>&1)"
if $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    :
else
    print "PyTorch: not installed (torch-dependent cases will framework_skip — this is normal)"
fi
print

print "--- running lab ---"
$PYTHON run_lab.py
print

print "--- running tests ---"
$PYTHON -m unittest -v
print

print "Done. See observations.json / observations.csv / RESULTS.md"
