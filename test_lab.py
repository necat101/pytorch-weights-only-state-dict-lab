#!/usr/bin/env python3
import unittest, json, csv, sys, os, copy

with open("observations.json") as f:
    rows = json.load(f)

def find(case, method):
    for r in rows:
        if r["case_id"] == case and r["method"] == method:
            return r
    return None

# torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

with open("cases.json") as f:
    CASES_DATA = json.load(f)

CASE_IDS = [c["case_id"] for c in CASES_DATA]
METHODS = ["inspect_environment", "exercise_serialization", "verify_roundtrip", "security_context_observation"]
CLASSIFICATIONS = {"pass","expected_error","local_observation","framework_skip","context_only","not_applicable","fail"}

class TestLab(unittest.TestCase):
    def test_case_ids(self):
        got = sorted(set(r["case_id"] for r in rows))
        self.assertEqual(got, sorted(CASE_IDS))

    def test_methods(self):
        got = sorted(set(r["method"] for r in rows))
        self.assertEqual(got, sorted(METHODS))

    def test_forty_rows_unique(self):
        self.assertEqual(len(rows), 40)
        pairs = [(r["case_id"], r["method"]) for r in rows]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_classification_vocabulary(self):
        for r in rows:
            self.assertIn(r.get("actual_classification"), CLASSIFICATIONS)
            self.assertIn(r.get("expected_classification"), CLASSIFICATIONS)

    def test_expected_actual_fields_present(self):
        for r in rows:
            self.assertIn("expected_classification", r)
            self.assertIn("actual_classification", r)
            self.assertIsInstance(r["expected_classification"], str)
            self.assertIsInstance(r["actual_classification"], str)

    def test_non_applicable_pairs(self):
        r = find("torch_environment_marker", "exercise_serialization")
        self.assertEqual(r["actual_classification"], "not_applicable")
        r = find("torch_environment_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "not_applicable")

    def test_expectation_independence(self):
        """Mutate every expectation in a copied manifest and demonstrate production observations do not change."""
        # Load original cases
        with open("cases.json") as f:
            original_cases = json.load(f)
        # Build mutated cases - flip every expected_classification
        mutated_cases = copy.deepcopy(original_cases)
        flip_map = {"pass": "fail", "fail": "pass", "expected_error": "pass", "local_observation": "pass", "context_only": "pass", "not_applicable": "pass", "framework_skip": "pass"}
        for case in mutated_cases:
            for method_meta in case.get("methods", {}).values():
                orig = method_meta.get("expected_classification", "fail")
                method_meta["expected_classification"] = flip_map.get(orig, "fail")
        # Write mutated cases to temp file and run lab with it
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            mutated_path = os.path.join(tmpdir, "cases.json")
            with open(mutated_path, "w") as f:
                json.dump(mutated_cases, f)
            # Copy run_lab.py to temp dir
            import shutil
            run_lab_src = os.path.join(os.path.dirname(__file__), "run_lab.py")
            run_lab_tmp = os.path.join(tmpdir, "run_lab.py")
            shutil.copy(run_lab_src, run_lab_tmp)
            # Patch run_lab to use mutated cases.json
            with open(run_lab_tmp, "r") as f:
                content = f.read()
            content = content.replace('CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")',
                                      f'CASES_PATH = r"{mutated_path}"')
            with open(run_lab_tmp, "w") as f:
                f.write(content)
            # Run mutated lab
            result = subprocess.run([sys.executable, run_lab_tmp], cwd=tmpdir, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, f"mutated run failed: {result.stderr}")
            # Load mutated observations
            with open(os.path.join(tmpdir, "observations.json")) as f:
                mutated_rows = json.load(f)
        # Compare actual_classifications - they must be identical despite mutated expectations
        orig_actual = {(r["case_id"], r["method"]): r["actual_classification"] for r in rows}
        mutated_actual = {(r["case_id"], r["method"]): r["actual_classification"] for r in mutated_rows}
        self.assertEqual(orig_actual, mutated_actual, "actual_classification changed when expectations were mutated - production is NOT independent from expectations")
        # Also verify expected_classifications DID change (proving mutation worked)
        orig_expected = {(r["case_id"], r["method"]): r["expected_classification"] for r in rows}
        mutated_expected = {(r["case_id"], r["method"]): r["expected_classification"] for r in mutated_rows}
        self.assertNotEqual(orig_expected, mutated_expected, "expectation mutation did not take effect - test is invalid")

    def test_missing_handler_failure(self):
        """Remove a production handler and invoke the real row builder - must produce actual_classification=fail."""
        import run_lab
        # Save original
        orig_handler = run_lab.HANDLERS.get("plain_tensor_weights_only_marker")
        self.assertIsNotNone(orig_handler, "handler must exist for test")
        try:
            # Remove handler
            del run_lab.HANDLERS["plain_tensor_weights_only_marker"]
            # Invoke real row builder
            actual, obs = run_lab.run_case_method("plain_tensor_weights_only_marker", "verify_roundtrip")
            self.assertEqual(actual, "fail", "missing handler did not produce fail classification")
            self.assertIn("missing handler", str(obs.get("reason", "")).lower())
        finally:
            # Restore
            run_lab.HANDLERS["plain_tensor_weights_only_marker"] = orig_handler

    # torch-dependent tests (skipped if torch unavailable)
    def test_tensor_equality(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("plain_tensor_weights_only_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_primitive_checkpoint_contents(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("primitive_checkpoint_weights_only_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_state_dict_tensors(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("state_dict_roundtrip_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_model_reconstruction_keys(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("model_reconstruction_marker", "exercise_serialization")
        obs = r["observation"] or {}
        self.assertEqual(obs.get("missing_keys"), [])
        self.assertEqual(obs.get("unexpected_keys"), [])

    def test_fixed_model_output_equality(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("model_reconstruction_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_restricted_benign_object_rejection(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("benign_custom_object_rejection_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "expected_error")
        obs = r["observation"] or {}
        self.assertTrue(obs.get("rejected"))
        exc_class = obs.get("exception_class", "")
        self.assertTrue(exc_class)

    def test_trusted_local_object_values(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("trusted_custom_object_explicit_load_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_malformed_checkpoint_rejection(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("malformed_checkpoint_rejection_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "expected_error")
        obs = r["observation"] or {}
        self.assertTrue(obs.get("rejected"))
        exc_class = obs.get("exception_class", "")
        self.assertTrue(exc_class)

    def test_cpu_device_placement(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        r = find("cpu_map_location_marker", "verify_roundtrip")
        self.assertEqual(r["actual_classification"], "pass")

    def test_json_csv_agreement(self):
        with open("observations.csv", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), len(rows))
        csv_map = {(r["case_id"], r["method"]): r for r in csv_rows}
        for r in rows:
            key = (r["case_id"], r["method"])
            self.assertIn(key, csv_map)
            cr = csv_map[key]
            self.assertEqual(cr["actual_classification"], r["actual_classification"])
            self.assertEqual(cr["expected_classification"], r["expected_classification"])
            obs_json = cr["observation"]
            if obs_json:
                decoded = json.loads(obs_json)
                self.assertEqual(decoded, r["observation"])

    def test_results_agreement(self):
        with open("RESULTS.md") as f:
            results = f.read()
        self.assertIn("Rows: 40", results)
        from collections import Counter
        counts = Counter(r["actual_classification"] for r in rows)
        for k, v in counts.items():
            self.assertIn(f"{k}: {v}", results)

    def test_classification_totals(self):
        from collections import Counter
        counts = Counter(r["actual_classification"] for r in rows)
        with open("RESULTS.md") as f:
            results = f.read().lower()
        for k in CLASSIFICATIONS:
            self.assertIn(f"{k}:", results)
        if not TORCH_AVAILABLE:
            self.assertEqual(counts.get("framework_skip", 0), 32)
            self.assertEqual(counts.get("pass", 0), 1)
            self.assertEqual(counts.get("local_observation", 0), 1)
            self.assertEqual(counts.get("context_only", 0), 4)
            self.assertEqual(counts.get("not_applicable", 0), 2)
            self.assertEqual(counts.get("fail", 0), 0)
            self.assertEqual(counts.get("expected_error", 0), 0)

    def test_required_disclaimers(self):
        with open("README.md") as f:
            readme = f.read().lower()
        with open("RESULTS.md") as f:
            results = f.read().lower()
        text = readme + "\n" + results
        required = [
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
        missing = [p for p in required if p not in text]
        self.assertEqual(missing, [], f"missing disclaimers: {missing}")

    def test_no_committed_binaries(self):
        bad_exts = [".pt", ".pth", ".pkl", ".pickle", ".bin"]
        for root, dirs, files in os.walk("."):
            if ".git" in root.split(os.sep):
                continue
            for fn in files:
                _, ext = os.path.splitext(fn.lower())
                self.assertNotIn(ext, bad_exts, f"committed binary found: {os.path.join(root, fn)}")

    def test_no_private_paths(self):
        artifacts = ["README.md","RESULTS.md","VERIFY.md","cases.json","observations.json","observations.csv","run_lab.py","test_lab.py","hn_thread_evidence.md","hn_comments_sanitized.json",".gitignore",]
        prohibited = [
            ("/home/", "home directory path"),
            ("/tmp/", "tmp directory path"),
            ("/root/", "root home path"),
            ("/var/", "var filesystem path"),
            ("ghp_", "GitHub PAT prefix"),
            ("github_pat_", "GitHub PAT prefix"),
            ("Bearer ", "Bearer token"),
            ("Authorization:", "Authorization header"),
            ("Traceback (most recent call last)", "traceback dump"),
            ("  File ", "traceback file line"),
            ("os.environ", "environment dump"),
            ("OPENCLAWHMAC", "HMAC secret"),
            ("CLAWMARK_TOOL_PROXY_TOKEN", "tool proxy token"),
        ]
        # Allowlist: (file_path, required_context, bad_pattern)
        allowances = [
            ("test_lab.py", "home directory path", "/home/"),
            ("test_lab.py", "tmp directory path", "/tmp/"),
            ("test_lab.py", "root home path", "/root/"),
            ("test_lab.py", "var filesystem path", "/var/"),
            ("test_lab.py", "GitHub PAT prefix", "ghp_"),
            ("test_lab.py", "GitHub PAT prefix", "github_pat_"),
            ("test_lab.py", "Bearer token", "Bearer "),
            ("test_lab.py", "Authorization header", "Authorization:"),
            ("test_lab.py", "traceback dump", "Traceback (most recent call last)"),
            ("test_lab.py", "traceback file line", "  File "),
            ("test_lab.py", "environment dump", "os.environ"),
            ("test_lab.py", "HMAC secret", "OPENCLAWHMAC"),
            ("test_lab.py", "tool proxy token", "CLAWMARK_TOOL_PROXY_TOKEN"),
        ]
        found_violations = []
        for path in artifacts:
            if not os.path.exists(path):
                if path == "VERIFY.md": continue
                self.fail(f"required artifact missing: {path}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for lineno, line in enumerate(content.splitlines(), 1):
                for bad_pattern, bad_desc in prohibited:
                    if bad_pattern not in line: continue
                    allowed = False
                    for allow_file, allow_context, allow_pat in allowances:
                        if allow_pat != bad_pattern: continue
                        if allow_file and allow_file != path: continue
                        if allow_context in line:
                            allowed = True
                            break
                    if not allowed and "/clean-checkout" in line:
                        allowed = True
                    if not allowed:
                        found_violations.append(f"{path}:{lineno}: {bad_desc} {bad_pattern!r}")
        if found_violations:
            self.fail("Prohibited content found:\n" + "\n".join(found_violations))



if __name__ == "__main__":
    unittest.main()
