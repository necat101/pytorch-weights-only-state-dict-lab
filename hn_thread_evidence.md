# HN thread evidence — "Exploiting machine learning Pickle files"

Hacker News item 26489150 — https://news.ycombinator.com/item?id=26489150
Linked article: https://blog.trailofbits.com/2021/03/15/never-a-dill-moment-exploiting-machine-learning-pickle-files/

Thread was fetched via the Hacker News API before the README discussion was prepared. See `hn_comments_sanitized.json` for full public comment text with item id, author, parent id, timestamp, and type.

## Comment summaries

**roddux (26505183)** — highlighted that the Trail of Bits article author also created **fickling** (https://github.com/trailofbits/fickling), a tool for examining pickle files. Quoting the article: "[Fickling] can help you reverse engineer, test, and even create malicious pickle files."

**tyingq (26502986)** — was surprised that pickle was used for machine-learning models, noting a common impression that pickle is slow. Wondered whether support for complex Python objects explained its popularity in ML: "Maybe it's used here because it's Python aware, and doesn't have trouble saving complex data structures?"

**kvathupo (26504295, reply to tyingq)** — distinguished research code from deployment engineering. Described deep-learning research code as often having larger software engineering problems than just pickling, underscoring a distinction between computer science and software engineering. Noted that sleep-deprived grad students produce borderline unreadable paper code because they don't care about deployment, and expressed hope that enterprise pipeline engineers take such risks into consideration.

**nonameiguess (26503086, reply to tyingq)** — emphasized that Python's own documentation warns that pickle is unsafe (in a big red box). Also noted pickle doesn't support NumPy natively (though NumPy has its own persistence modules), and that they would have expected something like HDF5 for ML model storage.

**liuliu (26506471, reply to nonameiguess)** — separated checkpointing from export, explaining that dynamic PyTorch models make persistence more complicated. Recalled discussions with Soumith about HDF5 early in the AI renaissance (~2014), noting Torch (Lua) maintainers were aware of better formats. Stated that moving from Caffe/TensorFlow to dynamic PyTorch models made it harder to persist "both the executable objects and the weights" efficiently and safely. Argued that "export" and "checkpointing" should be two different things: an exported model should be safe to deploy on platforms like Azure ML, while a checkpointing model should be treated like code. Suggested ONNX fills the export role.

**hprotagonist (26503226, reply to nonameiguess)** — noted that a zip container does not by itself remove the underlying serialization problem. Pointed out that PyTorch serialized models can include Python code for JIT scripts, making it non-obvious how to store Python code safely. Noted that torch moved to a zipfile implementation as of PyTorch 1.6 (https://pytorch.org/docs/stable/generated/torch.save.html). Also noted MATLAB's .mat files have been HDF5 since R2006b.

**ogrisel (26504149, reply to tyingq)** — explained that pickle performance depends strongly on whether an object contains many small Python objects or a few large numerical arrays. `pickle.dump`/`load` is slow with many small nested objects (e.g., dicts with millions of small str/int values). With a few large sub-objects (multi-MB/GB NumPy arrays for ML model parameters), pickle can be very fast, IO-bottlenecked.

**wodenokoto (26503845, reply to tyingq)** — said pickle is commonly used internally where developers believe the code is trusted: "I'm surprised pickled models are used for sharing with 3rd parties. But internally in projects I see it used all the time. It's easy and it works and you trust internal code."

**ori_b (26504444, reply to wodenokoto)** — warned that internal code has a habit of becoming external code.

**krallistic (26503807, reply to tyingq)** — argued that loading performance is often less important than the security risk. For most use-cases, model loading cost is low compared to training cost or thousands of inference calls. "BUT the security problems still remain and weigh much higher."

**wendythehacker (26506435, top-level)** — raised authenticity and integrity checking. Noted that ML frameworks (including newer ones) don't have built-in authenticity/integrity checking when loading model and architecture. Developers must build their own solutions like hash/signature checking — very few do.

Additional thread context: lunixbochs reported a Defcon CTF 2019 exploit via a pickled TensorFlow model; mrguyorama/a-dub/craigacp discussed XGBoost's JSON checkpoint format being experimental, slow, and subject to float precision loss, with a separate binary C++ format.

## What the thread does NOT prove

Per repository scope constraints, the HN thread does NOT prove that:

- every PyTorch checkpoint is malicious,
- every pickle file contains executable behavior,
- a zip-based checkpoint is automatically safe,
- `state_dict` alone describes a complete model architecture,
- JSON is always an appropriate tensor format, or
- one local serialization test validates a production model pipeline.

This repository is a small correctness/evidence lab about `torch.save`/`torch.load` with explicit `weights_only`, `state_dict`/`load_state_dict`, and narrow tensor equality — not an exploit, scanner, benchmark, or security certification.
