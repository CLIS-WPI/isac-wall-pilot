# isac-wall-pilot

Code and pre-registered evaluation for **"Physical Sensing Verification for
Cognitive ISAC against UAV Spoofing via Digital Twins"** (S. B. Hashemi Natanzi,
B. Tang, WPI).

A monostatic 5G/6G base station confirms a claimed target echo by predicting,
from a digital twin, the *forced* wall echo any single-antenna relay would also
have to produce, then checking the measurement for it. The pipeline is GPU-only,
built on Sionna 2.2.0 (PyTorch) ray tracing. Every threshold and decision rule
was committed to `docs/ANALYSIS_PLAN.md` **before** the held-out run; the
headline verdict is reported honestly as INCONCLUSIVE.

Paper: https://arxiv.org/abs/2611.08658

## Requirements

- NVIDIA GPU (developed on an H100 NVL) with the NVIDIA Container Toolkit.
- Docker. All Python runs inside the container — never `pip install` on the host.

## Build

```bash
docker build -t isac-wall-pilot:2.2.0 .
scripts/drun.sh python -m isac_verify preflight --require-h100 --output results/preflight
```

On a shared server, pick the allocated GPU by UUID:
`ISAC_GPU=device=GPU-xxxx scripts/drun.sh python -m isac_verify preflight ...`

## Reproduce the evaluation

The pipeline runs in the frozen order below (see `docs/ANALYSIS_PLAN.md` for the
exact protocol and `AGENTS.md` for the standing rules):

```bash
scripts/drun.sh python -m isac_verify validate   --config configs/pilot.yaml   # physics checks
scripts/drun.sh python -m isac_verify run        --config configs/pilot.yaml   # ray-traced trials
scripts/drun.sh python -m isac_verify calibrate  --config configs/pilot.yaml   # thresholds (calibration only)
scripts/drun.sh python -m isac_verify heldout    --config configs/pilot.yaml   # frozen held-out run
scripts/drun.sh python -m isac_verify stress     --config configs/pilot.yaml   # standoff / robustness sweeps
scripts/drun.sh python -m isac_verify report                                   # summaries
```

Paper figures and tables are regenerated (CPU) from the committed summaries in
`docs/results/`:

```bash
scripts/drun.sh python scripts/paper_figures_tables.py
scripts/drun.sh python scripts/paper_wp_tables.py
```

## Layout

- `src/isac_verify/` — pipeline (planner, receiver, decision, experiments, metrics).
- `scripts/` — runners and paper figure/table generators.
- `configs/pilot.yaml` — scenario, RF profiles, calibration/held-out split.
- `docs/ANALYSIS_PLAN.md` — pre-registration (frozen; dated amendments only).
- `docs/DEVIATIONS.md` — every deviation, with reason.
- `docs/results/` — committed summaries behind the paper's tables and figures.
- `tests/` — verification tests (require CUDA).

## Citation

```bibtex
@misc{hashemi2026isac,
  title        = {Physical Sensing Verification for Cognitive ISAC against
                  UAV Spoofing via Digital Twins},
  author       = {Hashemi Natanzi, Seyed Bagher and Tang, Bo},
  howpublished = {arXiv:2611.08658},
  year         = {2026},
  url          = {https://arxiv.org/abs/2611.08658}
}
```
