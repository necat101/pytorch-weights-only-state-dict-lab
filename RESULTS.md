# RESULTS — pytorch-weights-only-state-dict-lab

PyTorch available: False
PyTorch version: (not installed)
CPU-only: yes

Cases: 10
Methods: 4 (inspect_environment, exercise_serialization, verify_roundtrip, security_context_observation)
Rows: 40

Classification counts:
- pass: 1
- expected_error: 0
- local_observation: 1
- framework_skip: 32
- context_only: 4
- not_applicable: 2
- fail: 0

PyTorch is not installed in this environment.

Torch-dependent observations (cases 2-9, all 4 methods each = 32 rows) are classified as `framework_skip`.

The following results would be recorded when PyTorch is available:
- Plain tensor roundtrip: [[1.0, -2.0], [3.5, 0.25]], float32, torch.equal expected True
- Primitive checkpoint: step=7, name='tiny-checkpoint', weights=[1.25, -0.5], shape=[2]
- state_dict keys: linear.weight, linear.bias
- Model reconstruction: missing_keys=[], unexpected_keys=[], output equality expected True
- Benign custom object restricted load: expected_error (weights_only=True rejection)
- Trusted local full-object load: label='local-only', count=3
- Malformed checkpoint rejection: expected_error
- CPU map_location: device='cpu', equality expected True

Actual recorded classifications for torch-dependent cases in this run: all `framework_skip`.

Framework skips: 32
Failures: 0

## Narrow conclusions

- PyTorch was not available in this environment; torch-dependent cases are honestly classified as framework_skip.
- Environment marker, documentation checks, manifest checks, and generated-artifact checks completed successfully.
- When PyTorch is available, the lab exercises: tensor roundtrip, primitive checkpoint, state_dict, model reconstruction, benign-object rejection, trusted local full-object load, malformed rejection, CPU map_location.

This lab does NOT prove that every PyTorch checkpoint is safe, that weights_only=True accepts every state dictionary, that restricted loading is a complete sandbox, that an exception identifies every unsafe global, that weights_only=False is safe for untrusted files, that a zip container prevents pickle behavior, that state_dict contains model architecture, that matching tensors prove model correctness, that matching outputs prove model quality, that a local checkpoint has authentic provenance, that a checksum or signature was verified, that safetensors or ONNX was evaluated, that a production deployment was secured, or that the lab is security-certified or production-ready.

