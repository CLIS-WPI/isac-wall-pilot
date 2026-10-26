# AGENTS.md — Standing rules for this repository

You are implementing a physics research pilot. The full specification is
`docs/BRIEF.md`. It is the single source of truth. Read the sections named in
each task before writing code. Never edit `docs/BRIEF.md`.

Work in phases (see the task message). Finish ONE phase, run its checks,
commit, then STOP and report. Never start the next phase on your own.

## 1. Environment facts (already verified on this server, do not re-discover)

- Host GPU: NVIDIA H100 NVL. It has NO RT cores; ray tracing runs on CUDA via OptiX.
- Sionna 2.2.0 is **PyTorch-based**. There is no TensorFlow anywhere. If you
  write `import tensorflow` or `tf.`, you are using outdated knowledge. Stop.
- Mitsuba variant on GPU: `cuda_ad_mono_polarized`.
- `RCSSolver` is imported from `sionna.rt.rcs`, not from `sionna.rt`.
- Dr.Jit/Mitsuba needs `libatomic1` (apt). Without it: `NameError: _drjit_ext`.
- `NVIDIA_DRIVER_CAPABILITIES` must include `graphics`, or OptiX is missing and
  RT silently falls to CPU.
- The host shell may print a `libtinfo.so.6: no version information` warning
  from a conda env. It is harmless. Run docker commands with conda deactivated.

## 2. Sionna API rule (most important rule)

Sionna 2.2.0 was released after your training data. Your memory of its API is
probably wrong (older versions used TensorFlow and different signatures).

Before using ANY Sionna / Mitsuba / Dr.Jit function or class:
1. Inspect it inside the container: `inspect.signature`, `help()`, and read the
   installed source under `/opt/venv/lib/python3.12/site-packages/sionna/`.
2. Record the real signature, argument meanings, return shapes/dtypes, and
   default values (especially `normalize_delays`, `normalize`, `out_type`) in
   `docs/API_NOTES.md`, with a tiny snippet that you actually executed and
   its actual printed output.
3. Only then use it in package code.

Never guess an argument name. If something does not exist, say so and
search the installed source; do not invent a wrapper that pretends it exists.

## 3. Execution rules

- All Python execution happens inside Docker on the GPU. Use
  `scripts/drun.sh` (see Phase 0). Never `pip install` on the host.
- Scientific code must hard-fail if Mitsuba is not on a `cuda_*` variant or if
  tensors are not on CUDA. Never add a CPU/LLVM fallback path.
- Do not change host GPU settings, MIG mode, drivers, or other users' containers.
  Do not stop the existing `sionna` Jupyter container.
  Never run `docker system prune`, `docker rm -f` on containers you did not
  create, `sudo`, `nvidia-smi -r`, or anything that kills other processes.
- Keep heavy tensors on the GPU. No per-subcarrier Python loops.
- Use complex64/float32 for the main pipeline; complex128/float64 for analytical checks.

## 4. Scientific honesty rules (violating these makes the whole project worthless)

- Never write a number into a report, README, docstring, figure, or log that was
  not produced by code that actually ran in this repo. No "expected" results.
- The analytical values in BRIEF §5.3 are for verification tests only.
- No placeholder figures, no fabricated arrays, no `TODO` inside computation
  branches, no `pass` in a function that should compute something.
- Never weaken, skip, delete, or loosen the tolerance of a test to make it pass.
  If a test fails, keep it failing and report the measured values and your diagnosis.
- Never tune geometry, power, thresholds, or gates to make the attack and
  target separate. A negative or inconclusive result is a valid outcome.
- Never use ground truth (true positions, true wall offsets, true path delays,
  hypothesis labels, evaluator seeds) inside planner/receiver/decision code.
  Truth may only flow into `experiments.py` (generation) and `metrics.py` (scoring).
- Thresholds are chosen on calibration geometries only and frozen before the
  held-out run. The held-out run requires explicit human approval.
- Allowed final decision labels: SUPPORTED, UNRESOLVED, ACQUISITION_FAILED,
  NOT_OBSERVABLE (path status). Never output "ATTACK DETECTED".

## 5. Scope

Do NOT implement: LLM agents, multi-agent logic, RL, Bayesian belief states,
OAI, USRP, schedulers, throughput results, a manuscript, or "differentiable
ray tracing" claims. Only what BRIEF.md specifies.

## 6. Deviations and ambiguity

- If the brief is ambiguous, choose the most conservative reading, write it in
  `docs/DEVIATIONS.md` (what, why, which BRIEF section), and mention it in your report.
- If a pinned version/tag is unavailable, record the exact error in
  `docs/DEVIATIONS.md` and choose the nearest explicitly pinned alternative.

## 7. Git

- Commit at the end of each phase: `phaseN: <summary>`. Do not rewrite history.
- `results/` is git-ignored (except small JSON/MD summaries if a phase says so).

## 8. Phase report format (end every phase with exactly this)

```
PHASE N REPORT
1. Files created/changed: (list)
2. Commands run + exact final output lines (copy, do not paraphrase)
3. Tests: X passed / Y failed / Z errors (list each failing test with measured values)
4. API facts learned (new entries in docs/API_NOTES.md)
5. Deviations from BRIEF (entries in docs/DEVIATIONS.md)
6. Anything I am unsure about
7. STOPPED. Waiting for approval to start phase N+1.
```
