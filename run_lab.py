#!/usr/bin/env python3
"""pytorch-weights-only-state-dict-lab — correctness lab

10 cases × 4 methods = 40 rows
Classifications: pass, expected_error, local_observation, framework_skip, context_only, not_applicable, fail
"""
import sys, json, io, csv, platform, inspect

CASE_IDS = [
    "torch_environment_marker",
    "plain_tensor_weights_only_marker",
    "primitive_checkpoint_weights_only_marker",
    "state_dict_roundtrip_marker",
    "model_reconstruction_marker",
    "benign_custom_object_rejection_marker",
    "trusted_custom_object_explicit_load_marker",
    "malformed_checkpoint_rejection_marker",
    "cpu_map_location_marker",
    "no_global_serialization_or_ml_validity_claim_marker",
]
METHODS = [
    "inspect_environment",
    "exercise_serialization",
    "verify_roundtrip",
    "security_context_observation",
]

# --- torch availability ---
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
    TORCH_VERSION = getattr(torch, "__version__", "unknown")
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    TORCH_VERSION = None
    torch = None  # type: ignore

# Benign custom object (top-level, no __reduce__)
class BenignRecord:
    def __init__(self):
        self.label = "local-only"
        self.count = 3

# TinyLinear for state_dict tests
if TORCH_AVAILABLE:
    class TinyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2, bias=True)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]], dtype=torch.float32))
                self.linear.bias.copy_(torch.tensor([0.25, -0.75], dtype=torch.float32))

def sanitize_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:180]}"

def row(case_id, method, classification, observation=None):
    return {
        "case_id": case_id,
        "method": method,
        "classification": classification,
        "observation": observation,
    }

rows = []

# ---------------------------------------------------------------------
# Case 1: torch_environment_marker
# ---------------------------------------------------------------------
case = "torch_environment_marker"

# inspect_environment
try:
    env = {
        "python_version": platform.python_version(),
        "torch_available": TORCH_AVAILABLE,
        "torch_version": TORCH_VERSION if TORCH_AVAILABLE else None,
        "torch_import_error": None if TORCH_AVAILABLE else TORCH_IMPORT_ERROR,
    }
    if TORCH_AVAILABLE:
        env.update({
            "cpu_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "default_dtype": str(torch.get_default_dtype()),
            "torch_save_exists": hasattr(torch, "save"),
            "torch_load_exists": hasattr(torch, "load"),
            "load_state_dict_exists": hasattr(torch.nn.Module, "load_state_dict"),
        })
        # check weights_only in torch.load signature
        try:
            sig = inspect.signature(torch.load)
            env["torch_load_weights_only_arg"] = "weights_only" in sig.parameters
        except Exception:
            env["torch_load_weights_only_arg"] = None
        # default device (safely)
        try:
            env["default_device"] = str(torch.tensor([0]).device)
        except Exception:
            env["default_device"] = None
    rows.append(row(case, "inspect_environment", "pass", env))
except Exception as e:
    rows.append(row(case, "inspect_environment", "fail", sanitize_exc(e)))

# exercise_serialization
rows.append(row(case, "exercise_serialization", "not_applicable", {"note": "environment marker, no serialization exercise"}))

# verify_roundtrip
rows.append(row(case, "verify_roundtrip", "not_applicable", {"note": "environment marker"}))

# security_context_observation
try:
    obs = {"torch_available": TORCH_AVAILABLE}
    if not TORCH_AVAILABLE:
        obs["note"] = "PyTorch not installed in this environment; torch-dependent cases will be framework_skip."
    rows.append(row(case, "security_context_observation", "local_observation", obs))
except Exception as e:
    rows.append(row(case, "security_context_observation", "fail", sanitize_exc(e)))

# ---------------------------------------------------------------------
# Helper for framework_skip rows
# ---------------------------------------------------------------------
def framework_skip_case(case_id, reason="PyTorch not available in this environment"):
    for m in METHODS:
        rows.append(row(case_id, m, "framework_skip", {"reason": reason, "torch_available": TORCH_AVAILABLE}))

# ---------------------------------------------------------------------
# Cases 2-9: torch-dependent
# ---------------------------------------------------------------------
torch_dependent_cases = [
    "plain_tensor_weights_only_marker",
    "primitive_checkpoint_weights_only_marker",
    "state_dict_roundtrip_marker",
    "model_reconstruction_marker",
    "benign_custom_object_rejection_marker",
    "trusted_custom_object_explicit_load_marker",
    "malformed_checkpoint_rejection_marker",
    "cpu_map_location_marker",
]

if not TORCH_AVAILABLE:
    for cid in torch_dependent_cases:
        framework_skip_case(cid)
else:
    # -------------------------------------------------
    # Case 2: plain_tensor_weights_only_marker
    # -------------------------------------------------
    case = "plain_tensor_weights_only_marker"
    # inspect_environment
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION, "cpu_only": True}))
    # exercise_serialization
    try:
        tensor = torch.tensor([[1.0, -2.0], [3.5, 0.25]], dtype=torch.float32)
        buf = io.BytesIO()
        torch.save(tensor, buf)
        byte_len = buf.tell()
        buf.seek(0)
        obs = {
            "source_dtype": str(tensor.dtype),
            "source_shape": list(tensor.shape),
            "source_device": str(tensor.device),
            "source_values": tensor.tolist(),
            "serialized_byte_length": byte_len,
        }
        rows.append(row(case, "exercise_serialization", "pass", obs))
        # verify_roundtrip
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        v_obs = {
            "loaded_dtype": str(loaded.dtype),
            "loaded_shape": list(loaded.shape),
            "loaded_device": str(loaded.device),
            "loaded_values": loaded.tolist(),
            "torch_equal": bool(torch.equal(tensor, loaded)),
        }
        cls = "pass" if torch.equal(tensor, loaded) else "fail"
        rows.append(row(case, "verify_roundtrip", cls, v_obs))
        # security_context_observation
        rows.append(row(case, "security_context_observation", "local_observation", {
            "weights_only": True,
            "map_location": "cpu",
            "note": "locally generated tensor, weights_only=True explicitly passed"
        }))
    except Exception as e:
        # fill remaining
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 3: primitive_checkpoint_weights_only_marker
    # -------------------------------------------------
    case = "primitive_checkpoint_weights_only_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        weights = torch.tensor([1.25, -0.5], dtype=torch.float32)
        ckpt = {"step": 7, "name": "tiny-checkpoint", "weights": weights, "shape": [2]}
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        rows.append(row(case, "exercise_serialization", "pass", {
            "keys": sorted(list(ckpt.keys())),
            "primitive_step": ckpt["step"],
            "primitive_name": ckpt["name"],
        }))
        v_obs = {
            "loaded_keys": sorted(list(loaded.keys())),
            "step_equal": loaded.get("step") == 7,
            "name_equal": loaded.get("name") == "tiny-checkpoint",
            "weights_equal": bool(torch.equal(weights, loaded.get("weights", torch.tensor([])))),
            "shape_equal": loaded.get("shape") == [2],
        }
        all_ok = all([v_obs["step_equal"], v_obs["name_equal"], v_obs["weights_equal"], v_obs["shape_equal"]])
        rows.append(row(case, "verify_roundtrip", "pass" if all_ok else "fail", v_obs))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "weights_only": True,
            "note": "locally generated dict with primitive types + tensor; do not generalize to every python object or checkpoint schema"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 4: state_dict_roundtrip_marker
    # -------------------------------------------------
    case = "state_dict_roundtrip_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        model = TinyLinear()
        sd = model.state_dict()
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        loaded_sd = torch.load(buf, map_location="cpu", weights_only=True)
        rows.append(row(case, "exercise_serialization", "pass", {
            "state_dict_keys": sorted(list(sd.keys())),
        }))
        # verify every tensor
        tensor_results = {}
        all_equal = True
        for k in sd.keys():
            src = sd[k]
            dst = loaded_sd.get(k)
            eq = dst is not None and torch.equal(src, dst)
            all_equal = all_equal and eq
            tensor_results[k] = {
                "shape": list(src.shape),
                "dtype": str(src.dtype),
                "device": str(src.device),
                "values": src.tolist(),
                "equal": eq,
            }
        rows.append(row(case, "verify_roundtrip", "pass" if all_equal else "fail", tensor_results))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "weights_only": True,
            "note": "state_dict contains parameters only, not architecture"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 5: model_reconstruction_marker
    # -------------------------------------------------
    case = "model_reconstruction_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        # source model
        model_src = TinyLinear()
        sd = model_src.state_dict()
        # fresh model
        model_dst = TinyLinear()
        # zero it to prove load works
        with torch.no_grad():
            for p in model_dst.parameters():
                p.zero_()
        # load
        missing, unexpected = model_dst.load_state_dict(sd, strict=True)
        # Actually load_state_dict returns a namedtuple in newer torch, or missing/unexpected keys
        # Handle both
        if hasattr(missing, "missing_keys"):
            missing_keys = missing.missing_keys
            unexpected_keys = missing.unexpected_keys
        else:
            missing_keys = list(missing) if isinstance(missing, (list, tuple)) else []
            unexpected_keys = list(unexpected) if isinstance(unexpected, (list, tuple)) else []
        rows.append(row(case, "exercise_serialization", "pass", {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }))
        # inference
        x = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float32)
        with torch.inference_mode():
            out_src = model_src(x)
            out_dst = model_dst(x)
        equal = torch.equal(out_src, out_dst)
        rows.append(row(case, "verify_roundtrip", "pass" if equal else "fail", {
            "input": x.tolist(),
            "output_src": out_src.tolist(),
            "output_dst": out_dst.tolist(),
            "equal": equal,
        }))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "note": "same architecture and parameters produced the same fixed output; state_dict does not reconstruct architecture automatically; model is not trained or useful"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 6: benign_custom_object_rejection_marker
    # -------------------------------------------------
    case = "benign_custom_object_rejection_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        obj = BenignRecord()
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        rows.append(row(case, "exercise_serialization", "pass", {
            "type": "BenignRecord",
            "label": obj.label,
            "count": obj.count,
        }))
        # try weights_only load
        try:
            loaded = torch.load(buf, map_location="cpu", weights_only=True)
            # If we get here, rejection did NOT happen
            rows.append(row(case, "verify_roundtrip", "fail", {
                "note": "weights_only=True load unexpectedly succeeded for custom object",
                "loaded_type": type(loaded).__name__
            }))
        except Exception as load_e:
            rows.append(row(case, "verify_roundtrip", "expected_error", {
                "exception_class": type(load_e).__name__,
                "exception_summary": sanitize_exc(load_e),
                "rejected": True,
            }))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "note": "restricted load rejection is expected; exception messages may vary by pytorch version"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 7: trusted_custom_object_explicit_load_marker
    # -------------------------------------------------
    case = "trusted_custom_object_explicit_load_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        obj = BenignRecord()
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        rows.append(row(case, "exercise_serialization", "pass", {
            "type": "BenignRecord",
            "saved_label": obj.label,
            "saved_count": obj.count,
        }))
        loaded = torch.load(buf, map_location="cpu", weights_only=False)
        ok = (getattr(loaded, "label", None) == "local-only" and getattr(loaded, "count", None) == 3)
        rows.append(row(case, "verify_roundtrip", "pass" if ok else "fail", {
            "loaded_type": type(loaded).__name__,
            "loaded_label": getattr(loaded, "label", None),
            "loaded_count": getattr(loaded, "count", None),
            "match": ok,
        }))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "weights_only": False,
            "note": "trusted local buffer only; NOT proof that arbitrary pickle loading is safe"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 8: malformed_checkpoint_rejection_marker
    # -------------------------------------------------
    case = "malformed_checkpoint_rejection_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        buf = io.BytesIO(b"not a pytorch checkpoint")
        rows.append(row(case, "exercise_serialization", "pass", {"malformed_bytes_len": 24}))
        try:
            loaded = torch.load(buf, map_location="cpu", weights_only=True)
            rows.append(row(case, "verify_roundtrip", "fail", {"note": "malformed input unexpectedly loaded"}))
        except Exception as load_e:
            rows.append(row(case, "verify_roundtrip", "expected_error", {
                "exception_class": type(load_e).__name__,
                "exception_summary": sanitize_exc(load_e),
                "rejected": True,
            }))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "note": "one malformed-input rejection is not a general parser-security guarantee"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

    # -------------------------------------------------
    # Case 9: cpu_map_location_marker
    # -------------------------------------------------
    case = "cpu_map_location_marker"
    rows.append(row(case, "inspect_environment", "pass", {"torch_version": TORCH_VERSION}))
    try:
        tensor = torch.tensor([[1.0, -2.0], [3.5, 0.25]], dtype=torch.float32)
        buf = io.BytesIO()
        torch.save(tensor, buf)
        buf.seek(0)
        rows.append(row(case, "exercise_serialization", "pass", {
            "source_device": str(tensor.device)
        }))
        loaded = torch.load(buf, map_location="cpu", weights_only=True)
        ok = (str(loaded.device) == "cpu" and torch.equal(tensor, loaded))
        rows.append(row(case, "verify_roundtrip", "pass" if ok else "fail", {
            "loaded_device": str(loaded.device),
            "equal": bool(torch.equal(tensor, loaded)),
        }))
        rows.append(row(case, "security_context_observation", "local_observation", {
            "note": "does not test accelerator checkpoint migration"
        }))
    except Exception as e:
        for m in ["exercise_serialization", "verify_roundtrip", "security_context_observation"]:
            if not any(r["case_id"] == case and r["method"] == m for r in rows):
                rows.append(row(case, m, "fail", sanitize_exc(e)))

# ---------------------------------------------------------------------
# Case 10: no_global_serialization_or_ml_validity_claim_marker
# ---------------------------------------------------------------------
case = "no_global_serialization_or_ml_validity_claim_marker"

def check_disclaimers():
    """Check README and RESULTS for required disclaimers."""
    try:
        with open("README.md") as f:
            readme = f.read().lower()
        with open("RESULTS.md") as f:
            results = f.read().lower()
    except FileNotFoundError:
        # During first run, files may not exist yet – allow context_only anyway
        return {"files_exist": False, "disclaimers_found": False}
    text = readme + "\n" + results
    # Required negative claims that must be disclaimed
    required_phrases = [
        "does not prove that every pytorch checkpoint is safe",
        "weights_only=true",
        "restricted loading",
        "complete sandbox",
        "weights_only=false",
        "untrusted",
        "zip",
        "pickle",
        "state_dict",
        "architecture",
        "model correctness",
        "model quality",
        "provenance",
        "safetensors",
        "onnx",
        "production",
    ]
    found = sum(1 for p in required_phrases if p in text)
    return {"files_exist": True, "disclaimer_hits": found, "text_len": len(text)}

# All four methods -> context_only
for m in METHODS:
    obs = check_disclaimers()
    obs["method"] = m
    rows.append(row(case, m, "context_only", obs))

# ---------------------------------------------------------------------
# Validate row completeness
# ---------------------------------------------------------------------
expected_pairs = {(c, m) for c in CASE_IDS for m in METHODS}
actual_pairs = {(r["case_id"], r["method"]) for r in rows}
missing = expected_pairs - actual_pairs
extra = actual_pairs - expected_pairs
if missing:
    print(f"ERROR missing pairs: {missing}", file=sys.stderr)
    sys.exit(1)
if extra:
    print(f"ERROR extra pairs: {extra}", file=sys.stderr)
    sys.exit(1)
if len(rows) != 40:
    print(f"ERROR row count {len(rows)} != 40", file=sys.stderr)
    sys.exit(1)

# Write observations.json
with open("observations.json", "w") as f:
    json.dump(rows, f, indent=2)

# Write observations.csv
with open("observations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case_id", "method", "classification", "observation"])
    w.writeheader()
    for r in rows:
        w.writerow({
            "case_id": r["case_id"],
            "method": r["method"],
            "classification": r["classification"],
            "observation": json.dumps(r["observation"], separators=(",", ":"), ensure_ascii=False) if r["observation"] is not None else "",
        })

# Print summary
from collections import Counter
counts = Counter(r["classification"] for r in rows)
print(f"PyTorch available: {TORCH_AVAILABLE}")
if TORCH_AVAILABLE:
    print(f"PyTorch version: {TORCH_VERSION}")
print(f"Rows: {len(rows)} (10 cases × 4 methods)")
for k in ["pass","expected_error","local_observation","framework_skip","context_only","not_applicable","fail"]:
    print(f"  {k}: {counts.get(k,0)}")
print("Wrote observations.json / observations.csv")
