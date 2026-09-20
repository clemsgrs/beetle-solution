# Attempt 04: frozen Mascaret

Attempt 04 keeps Attempt 01's lightweight two-block decoder, ROI population,
patient folds, uniform sampling, losses, optimizer, schedule, effective batch 64,
and no augmentation. The encoder is `wearewaiv/mascaret`, pinned at
`e95e7ea15e039e78d74def101415e19d9a67ba80` by slide2vec 5.8.2. Its recommended
encoding precision is FP32; the cache remains FP16. The decoder input projection
changes from 1,280 to 1,536 channels. The 37×37 feature grid is unchanged.

The run directory is `/maindisk/clement/beetle-attempt-04-20260912`. It contains:

- `cache/mascaret_e95e7ea15e039e78d74def101415e19d9a67ba80_dense_fp16/`: new features.
- `venv/`: repository-pinned Soma, slide2vec, hs2p and direct dependencies,
  inheriting the existing CUDA/PyTorch installation.
- `evidence/preparation.json`: model weight hash, package identities, exact ROI
  replay hashes, and resolved configuration hash.
- `evidence/smoke.json`: numerical extraction and batch-64 decoder preflight.
- `active-config.yaml`: resolved training/extraction settings.
- `status.json`: worker phase; consult the guard status first if work stops.
- `jobs/run-*/guard-status.json`: authoritative lifecycle and stop reason.
- `jobs/run-*/run.log` and `gpu_guard.log`: worker and GPU safety logs.
- `runs/`, `scores/`, `provenance/`: training, held-out test scores, and comparison.

`python -m beetle.attempt_04 prepare` verifies the original slide/split hashes,
replays the exact 124,697 ROI coordinates into the new cache, checks the replayed
ROI files byte-for-byte, and records the encoder weight hash.

`launch.py` in the run directory starts a detached resume loop (`beetle/gpu_resume.py`).
Each launch runs a supervisor (`beetle/gpu_job.py`) in a fresh `jobs/run-NN`
directory. The worker and all GPU work run in one process group, with one
persistent CUDA client. Before work, the supervisor requires an empty GPU process
query, validates sole ownership after CUDA initialization, and arms the run
checkout’s `beetle/gpu_guard.py` with a one-second interval. Both monitors query
only the physical GPU UUID selected by `CUDA_VISIBLE_DEVICES`; an index or
multiple devices are refused. The supervisor separately checks for new GPU
processes and query/guard failures. It terminates our process group on detection
and escalates after five seconds if needed.

The resume loop is given every GPU UUID on the machine and pins each launch to one
of them through `CUDA_VISIBLE_DEVICES`. It first launches on a GPU with no process.
It relaunches only after a yield to another GPU process, and only on a GPU that has
shown no process for 10 unbroken minutes, preferring the one idle longest. It polls
all GPUs while a launch runs, so a GPU that stayed free throughout is used as soon as
our job yields; otherwise the loop waits until some GPU qualifies. It does not relaunch
after any other stop, or after six consecutive yields within 20 minutes of launch;
those wait for a person. Its state is in `jobs/autoresume-status.json` and
`jobs/autoresume.log`. Touch `jobs/autoresume.stop` to prevent further
relaunches; send SIGTERM to the loop's PID to also stop the current launch.
Relaunches reuse Soma run `attempt-04-mascaret-20260912`, so folds with
`metrics.json` are skipped and an interrupted fold restarts from epoch 0.

The worker proceeds through smoke testing, full feature extraction, five-fold
training, held-out test scoring, and comparison against Attempt 01. Verdicts use
the research journal's paired test-fold rule and patient bootstrap. The comparison
is written to `provenance/comparison.json`; add the result and verdict to the
research journal once the experiment concludes.
