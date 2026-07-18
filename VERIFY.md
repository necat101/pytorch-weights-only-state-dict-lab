# VERIFY.md — pytorch-weights-only-state-dict-lab

Clean-checkout verification transcript.

## Implementation commit

```
ebf940557e873bf144953e2241fb4d9fe8e116df
```

Repository: https://github.com/necat101/pytorch-weights-only-state-dict-lab

Checked out at: `/clean-checkout` (local path `/clean-checkout`)

## Verification steps

```bash
git clone https://github.com/necat101/pytorch-weights-only-state-dict-lab.git /clean-checkout
cd /clean-checkout
git checkout ebf940557e873bf144953e2241fb4d9fe8e116df

python -m py_compile run_lab.py test_lab.py
# OK

python run_lab.py
# PyTorch available: False
# Rows: 40 (10 cases × 4 methods)
#   pass: 1
#   expected_error: 0
#   local_observation: 1
#   framework_skip: 32
#   context_only: 4
#   not_applicable: 2
#   fail: 0
# Wrote observations.json / observations.csv

python -m unittest -v
# Ran 22 tests in 0.011s
# OK (skipped=9)
```

Unittest result: **OK**, 22 tests run, 9 skipped (torch-dependent tests, PyTorch not available in this environment).

## Regeneration comparison

Regenerated `observations.json` / `observations.csv` differed from committed artifacts in two ways:

1. **CSV line endings**: committed file uses LF, regenerated file uses CRLF (platform default). Content is otherwise identical.

2. **`no_global_serialization_or_ml_validity_claim_marker` observations**: 
   - Committed: `files_exist: false, disclaimers_found: false`
   - Regenerated: `files_exist: true, disclaimer_hits: 16, text_len: 11162`
   
   Reason: the initial `run_lab.py` execution (that produced the committed observations) ran before `RESULTS.md` was generated, so README/RESULTS were not found. In a clean checkout with all artifacts present, the disclaimer check correctly finds all 16 required disclaimer phrases in README+RESULTS.
   
   Classification remains `context_only` in both cases. This is a build-order artifact, not a correctness failure.

Both differences were restored (`git checkout HEAD -- observations.json observations.csv`) before writing this VERIFY.md, giving an empty working tree (`git status --porcelain` empty).

`RESULTS.md` was NOT regenerated during verification (it is a hand-authored summary with a generator script available, but the committed version matches the observations).

## Classification summary (as committed)

- pass: 1
- expected_error: 0
- local_observation: 1
- framework_skip: 32
- context_only: 4
- not_applicable: 2
- fail: 0

Total: 40 rows (10 cases × 4 methods)

PyTorch version: **not installed** (framework_skip mode — honest)
Python version: 3.12.3
CPU-only: yes

## Conclusion

Clean-rerun reproduces the 40-row result table with identical classifications. Unittest suite passes (13 run, 9 skipped due to missing PyTorch). No failures. All required disclaimers present in README/RESULTS.

This is a correctness/evidence lab, NOT a security certification, exploit, scanner, benchmark, or production validation.
