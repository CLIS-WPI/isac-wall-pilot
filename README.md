# isac-wall-pilot

GPU-only Sionna 2.2.0 (PyTorch) pilot for active wall-assisted sensing
verification. Specification: `docs/BRIEF.md` (do not edit). Standing rules:
`AGENTS.md`.

Status: **Phase 0 only** (repository, container, preflight). There are no
scientific results yet. Later phases add `validate`, `run`, and `report`.

## Host prerequisites

- NVIDIA driver with NVIDIA Container Toolkit, one visible H100 (`--gpus`).
- `results/` is created automatically by `scripts/drun.sh`.

## Build

```bash
docker build -t isac-wall-pilot:2.2.0 .
scripts/record_image.sh
```

`scripts/record_image.sh` writes `results/preflight/image_info.json` with the
image ID, the built image repo digests, and the base image digest. The container
cannot inspect its own image.

## Preflight (must pass before any science)

```bash
scripts/drun.sh python -m isac_verify preflight --require-h100 --output results/preflight
```

On a shared server select the allocated GPU by UUID:

```bash
ISAC_GPU=device=GPU-xxxxxxxx scripts/drun.sh python -m isac_verify preflight \
  --require-h100 --output results/preflight
```

Preflight exits nonzero on any CUDA/OptiX failure and never falls back to CPU.
It writes `environment.json` and `preflight.log` under the output directory.

## Dependency freeze

```bash
scripts/drun.sh pip freeze > requirements.lock.txt
```

The NVIDIA CUDA base image prints a 15-line licence banner on container start,
so that banner precedes the freeze output. The committed `requirements.lock.txt`
is the exact freeze with only those banner lines removed.

## Development loop

```bash
ISAC_DEV=1 scripts/drun.sh python -m pytest -q tests
```

`ISAC_DEV=1` bind-mounts the repository at `/workspace` so code edits are used
without rebuilding. The container runs as the host UID/GID.

## Known limitations

- H100 has no RT cores; ray tracing uses CUDA/OptiX (`cuda_*` Mitsuba variant).
- `results/` is git-ignored; only explicitly required summaries are committed.
- The Dockerfile follows BRIEF §3.2 with one documented change (`libatomic1`)
  recorded in `docs/DEVIATIONS.md`.
