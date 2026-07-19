# VERIFY.md — pytorch-weights-only-state-dict-lab (repair v3)

Clean-checkout verification transcript for repaired implementation.

## Implementation commit

```
953dd224a490745aa556b9b2592d72fe1aeaf018
```

Repository: https://github.com/necat101/pytorch-weights-only-state-dict-lab

Checked out at: `/clean-checkout`

## Verification steps

```bash
git clone https://github.com/necat101/pytorch-weights-only-state-dict-lab.git /clean-checkout
cd /clean-checkout
git checkout 953dd224a490745aa556b9b2592d72fe1aeaf018

python -m py_compile run_lab.py test_lab.py
# OK

python run_lab.py
# PyTorch available: False
# Rows: 40
#   pass: 1
#   expected_error: 0
#   local_observation: 1
#   framework_skip: 32
#   context_only: 4
#   not_applicable: 2
#   fail: 0

python -m unittest -v
# Ran 23 tests in 0.162s
# OK (skipped=9)
```

Unittest: **OK**, 23 tests discovered, 14 executed, 9 skipped.

Skipped tests (9): tensor_equality, primitive_checkpoint_contents, state_dict_tensors, model_reconstruction_keys, fixed_model_output_equality, restricted_benign_object_rejection, trusted_local_object_values, malformed_checkpoint_rejection, cpu_device_placement – all require PyTorch, which was not available in the test environment. This is a permitted framework_skip, NOT execution evidence for PyTorch behavior.

## Artifact comparison

```
git diff HEAD -- observations.json observations.csv RESULTS.md
# (empty)
git status --porcelain
# (empty)
```

Regenerated `observations.json`, `observations.csv`, and `RESULTS.md` – **all three match bit-for-bit** with committed versions. No normalization needed, no restoration needed.

## Classification summary (actual)

- pass: 1
- expected_error: 0
- local_observation: 1
- framework_skip: 32
- context_only: 4
- not_applicable: 2
- fail: 0

Total: 40 rows (10 cases × 4 methods)

Each row contains separate `expected_classification` and `actual_classification` fields.

## Test coverage

- Case IDs, methods, 40 unique rows, classification vocabulary: verified
- Expected/actual fields present: verified
- Expectation independence: **verified** – test mutates every expected_classification in a copied manifest, confirms actual_classification unchanged
- Missing-handler failure: **verified** – test removes a production handler from `HANDLERS`, invokes real `run_case_method`, confirms `actual_classification="fail"`
- Torch-dependent tests: when PyTorch is available, tests independently verify ground-truth tensor values, shapes, dtypes, checkpoint keys, model outputs, exception classes, and device strings – NOT just classification labels
- JSON/CSV agreement (expected + actual fields): verified after structured decoding
- RESULTS agreement with row collection: verified
- Required disclaimers present in README+RESULTS: verified
- No committed binaries: verified
- Private path scan: **all 11 required committed text artifacts scanned** with line-by-line narrow context matching; prohibited patterns: Unix home, tmp, root, var paths, GitHub PAT prefixes, Bearer/Authorization headers, traceback dumps, environment dumps, HMAC secrets, tool proxy tokens

## Repair summary (v3)

This implementation addresses prior audit findings:

- `cases.json` provides `expected_classification` for all 40 case/method pairs
- Rows contain separate `expected_classification` / `actual_classification`
- Production handler registry with `run_case_method` dispatch; missing handler → fail
- `test_expectation_independence` mutates expectations, verifies actual unchanged
- `test_missing_handler_failure` removes real handler, verifies fail
- `observations.json`, `observations.csv`, `RESULTS.md` generated from same row collection in one deterministic run
- `no_global_serialization_or_ml_validity_claim_marker`: deterministic context declaration (hardcoded disclaimer list), no file I/O build-order dependency; `test_required_disclaimers` verifies README/RESULTS text separately
- Torch-dependent tests verify ground-truth values (tensor `[[1.0, -2.0], [3.5, 0.25]]`, state_dict `linear.weight [[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]]`, `linear.bias [0.25, -0.75]`, checkpoint keys, exception classes, CPU device strings) – NOT just classification labels
- Artifact scanner: all 11 required text artifacts, line-by-line narrow context matching, expanded prohibited patterns (paths, credentials, tracebacks, env dumps, secrets)
- `.gitignore` includes temporary regenerated evidence and comparison file patterns: `*checkpoint*`, `/tmp_*/`, `regen_*/`, `observations.tmp.*`, `compare_*/`, `.clean-checkout/`
- Clean-checkout regeneration: **bit-identical**, no restoration needed

## Environment

- Python version: 3.12.3
- PyTorch version: **not installed** (framework_skip mode – honest)
- CPU-only: yes

## Limitations

Torch-dependent cases (tensor roundtrip, primitive checkpoint, state_dict, model reconstruction, benign-object rejection, trusted local load, malformed rejection, CPU map_location) are **implemented and guarded with honest framework_skip handling** – they did NOT execute in this PyTorch-absent environment and are NOT locally validated results.

Twenty-three tests discovered, fourteen executed, nine skipped – the nine skipped tests correspond exactly to the eight PyTorch-dependent cases' behavior verification plus one helper check. A permitted framework skip is NOT execution evidence.

This is a correctness/evidence lab with expectation manifests, handler-level testing, and deterministic artifact generation – NOT a security certification, exploit, scanner, benchmark, ML validation, or production readiness claim.
