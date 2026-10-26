# Deviations from BRIEF

Each entry: what changed, why, and which BRIEF section it affects.

## D1 — `libatomic1` added to the apt package list

- **What:** Dockerfile apt-get list is BRIEF §3.2 plus `libatomic1`.
- **Why:** Dr.Jit/Mitsuba fails to import without it, raising
  `NameError: _drjit_ext` (the underlying error is
  `ImportError: libatomic.so.1: cannot open shared object file`). Verified on
  this server with Sionna 2.2.0 / Mitsuba 3.9.1 / Dr.Jit 1.5.0.
- **BRIEF section:** §3.2 Baseline Dockerfile.
