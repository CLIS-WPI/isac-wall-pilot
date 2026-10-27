# Deviations from BRIEF

Each entry: what changed, why, and which BRIEF section it affects.

## D1 — `libatomic1` added to the apt package list

- **What:** Dockerfile apt-get list is BRIEF §3.2 plus `libatomic1`.
- **Why:** Dr.Jit/Mitsuba fails to import without it, raising
  `NameError: _drjit_ext` (the underlying error is
  `ImportError: libatomic.so.1: cannot open shared object file`). Verified on
  this server with Sionna 2.2.0 / Mitsuba 3.9.1 / Dr.Jit 1.5.0.
- **BRIEF section:** §3.2 Baseline Dockerfile.

## D2 — Wall material is the Sionna finite-thickness slab with default thickness

- **Type:** interpretation.
- **What:** The wall uses Sionna's `RadioMaterial` finite-thickness
  single-layer slab model. The thickness is the API default
  `0.10000000149011612 m` as realized by Dr.Jit, recorded in
  `configs/pilot.yaml` as `wall.thickness_m`; it is a fixed controlled
  parameter and is not swept. Slab coefficients are evaluated at 3.5 GHz for
  incidence angles `0, 26.565, 45, 63.435` degrees measured from the inward
  wall normal and written to `results/phase2/material_reference.json`.
- **Why:** Phase 1 showed that `PathSolver` reflection coefficients match
  `itu_coefficients_single_layer_slab`, not the simplified single-interface
  Fresnel coefficients. The default thickness is part of the executed material
  object, so making it explicit and fixed avoids an unreported wall-loss
  degree of freedom.
- **BRIEF section:** §5.1 nominal physical scene; §7.1 actual Sionna
  components; §9.2 twin-guided wall prediction.
- **Phase 1 evidence:** `RadioMaterial` signature has `thickness=0.1`;
  creating the pilot material and printing `thickness` gave
  `0.10000000149011612`. At 3.5 GHz,
  `complex_relative_permittivity` gave `eta=[5-0.102715j]`. Executed slab
  checks matched RT:
  `r_TE=-0.2030512689+0.1115346686j` vs slab
  `-0.2030507326+0.1115333885j` (abs error `1.387875e-6`) at the 45-degree
  wall case, and `r_TE=-0.3660115766-0.2306077063j` vs slab
  `-0.3660115600-0.2306076884j` (abs error `2.441216e-8`) at normal
  incidence. The executed function returns `(r_te, r_tm, t_te, t_tm)`.

## D3 — Solver flags and seed handling are explicit for every scientific run

- **Type:** interpretation.
- **What:** Every scientific `PathSolver` and `RCSSolver` construction uses
  `refraction=False`, `diffraction=False`, `diffuse_reflection=False`,
  `deterministic=True`, and an explicit seed from `configs/pilot.yaml`; no
  scientific run passes `seed=None`. These keys are explicit in
  `solver.path_solver` and, where supported, `solver.rcs_solver`.
- **Why:** BRIEF §7.1 requires deterministic construction where supported and
  explicit seeds; the primary model is specular-first-order unless a separate
  labelled diagnostic says otherwise.
- **BRIEF section:** §7.1 actual Sionna components; §13.1 staged runs;
  §15 required validation tests.
- **Phase 1 evidence:** `PathSolver` is constructed as
  `PathSolver(deterministic=True)`; `deterministic` is not a `__call__`
  argument. `PathSolver.__call__` accepts `refraction`, `diffraction`, and
  `diffuse_reflection`; `RCSSolver.__call__` accepts `refraction` and
  `seed` but has no `diffraction` or `diffuse_reflection` argument. Measured
  deterministic solves with fixed seeds were exact in Phase 1. The unsupported
  RCSSolver keys are therefore recorded as unavailable flags rather than passed
  as invalid arguments; `RCSSolver` still uses `deterministic=True`,
  `refraction=False`, and an explicit seed.

## D4 — Primary additive model separates target paths from background and relay paths

- **Type:** interpretation.
- **What:** The primary additive model computes background clutter and the
  gNB-relay/relay-gNB links with `PathSolver` on a blank scene containing no
  target cube. Target-interacting paths come only from `RCSSolver` with the
  target present. Occlusion is a separate labelled check, not part of the
  primary additive `H_target + H_background + y_relay` construction.
- **Why:** BRIEF §7.1 assigns `RCSSolver` to target-interacting sensing paths
  and `PathSolver` to background and relay links; mixing a target scatterer
  into the background solve would double count or alter the background.
- **BRIEF section:** §7.1 actual Sionna components; §7.2 received-signal
  model.
- **Phase 1 evidence:** `RCSSolver` returned only target-interacting paths,
  and the target scattering event counts toward `max_depth`. A
  `ConstantRCSSensingTarget` at the LoS midpoint blocked ordinary
  `PathSolver` (`valid` changed from `1` to `0`), demonstrating that the two
  solves must not share the same target-present scene for the primary model.

## D5 — Transmit steering is `v(phi)=conj(a_TX(phi))`

- **Type:** interpretation.
- **What:** The planner uses
  `v(phi) = conj(a_TX(phi))`, with
  `a_TX(phi)_m = exp(+j 2*pi/lambda * p_m . k(phi)) / sqrt(M)`, and reports
  physical directional gain as `|v(phi)^H w|^2` exactly as in BRIEF §9.3.
  Code uses this order for beam construction and the RT adapter; it does not
  take an extra conjugate of the Sionna channel when evaluating `h @ w`.
- **Why:** Phase 1 measured that Sionna's synthetic-array TX response follows
  the `+j` exponential convention and that matched transmit weights are
  `conj(a_TX)`. This keeps the BRIEF formulas literal and avoids a sign
  inversion.
- **BRIEF section:** §9.2 twin-guided wall prediction; §9.3 wall beam and
  finite-aperture leakage; §15 required validation tests.
- **Phase 1 evidence:** `steering_vectors` returned unit-norm
  `a_m=exp(+j*2*pi/lambda*d_m.r_hat)/sqrt(M)` and synthetic-array RT phases
  matched with TX error `5.761645e-6`. Phase 2 adapter validation with
  `synthetic_array=True` and `h @ w` measured peaks: requested
  `-45 -> -45`, `-20 -> -20`, `0 -> 0`, `26.565 -> 27`, `45 -> 45`
  degrees on the forward visible arc. With `synthetic_array=False`, tested
  `Paths.a` phases were zero for this configuration, so that mode is not used
  for beam validation.

## D6 — The gNB TX-to-gNB RX direct self path is excluded by ideal isolation

- **Type:** interpretation.
- **What:** No background or relay computation solves the direct monostatic
  gNB TX-to-gNB RX path. Relay links are solved separately as gNB-to-relay and
  relay-to-gNB. The target is solved with `RCSSolver`. The assumed direct
  leakage is cancelled by ideal TX/RX isolation; all other enabled paths are
  retained.
- **Why:** The nominal Experiment A scene has co-located gNB phase centres.
  A direct self path at zero range is not a physical background path for the
  sensing model and would contaminate the matched negative control. BRIEF
  §11.1 removes walls and background scatterers for Experiment A.
- **BRIEF section:** §7.1 actual Sionna components; §7.2 received-signal model;
  §11 Experiment A.
- **Phase 3 evidence:** `background.self_link_policy=excluded` in
  `configs/pilot.yaml`; `channels.py` performs no gNB self solve. The executed
  Validation run solved only `gNB->relay`, `relay->gNB`, and the target path;
  the resulting target and matched relay responses agreed to
  `8.310603e-6` maximum relative error over 70 beams and every RX element.

## D7 — Matched relay gain and delay are calibrated from actual Sionna paths

- **Type:** interpretation.
- **What:** The evaluator calibrates the fixed relay gain at the reference
  direct beam from the actual `RCSSolver` target coefficient and the actual
  `PathSolver` downlink and uplink coefficients. It records the analytical
  nominal delay, the measured path delay, their difference, the complex gain,
  and the relay average transmit power. A gain whose transmit power exceeds
  the 10 dBm cap is marked infeasible and is never rescaled.
- **Why:** BRIEF §7.4 requires evaluator-side calibration against the actual
  direct-path candidate return and forbids hiding a cap violation by scaling
  the target or relay. The actual Sionna coefficients differ slightly from the
  closed-form free-space values because of the finite target model and
  array/solver conventions.
- **BRIEF section:** §7.4 attacker capabilities and constraints; §11.1
  physical assumptions; §11.2 required checks.
- **Phase 3 evidence:** Executed nominal values:
  measured `delta=200.1384586947097 ns`, analytical
  `delta=200.1384571188912 ns`, difference `1.575818e-6 ns`,
  `g=-10.298308372497559-0.9971696734428406j`, `|g|=10.3464729591`,
  relay average transmit power `8.841942352730346e-7 W` against the
  `0.01 W` cap (feasible). The corresponding target and matched-relay CFRs
  agreed to `8.310603e-6` maximum relative error over the 70 beam actions and every RX element. The target and relay RX steering vectors differed by `1.201e-07` relative L2 difference.

## D8 — Wall-thickness sensitivity is a labelled follow-up stress test

- **Type:** labelled follow-up stress test, not the BRIEF baseline.
- **What:** The baseline wall remains the Sionna default finite-thickness slab
  at `0.10000000149011612 m` from D2. A separate diagnostic sweeps
  `stress.wall_thickness_m` over `[0.05, 0.10, 0.20] m` and evaluates
  `r_TE` and `r_TM` at `63.435` degrees from the inward normal for the
  target-wall path and `45` degrees for the relay-wall path. It does not
  change the nominal scene, beam design, gate, or any Experiment B result.
- **Why:** The wall thickness is a fixed controlled parameter in the baseline,
  but a finite slab can show non-monotonic reflection behaviour from internal
  interference. Recording the sensitivity separately prevents an unreported
  material degree of freedom from being mistaken for the baseline.
- **BRIEF section:** §5.1 nominal physical scene; §7.1 actual Sionna
  components; §8.2 material-error stress separation.
- **Phase 4 evidence:** The executed reference in
  `results/phase4/geometry_reference.json` and `phase4_summary.json` gives:

| thickness (m) | angle (deg) | \|r_TE\| | \|r_TM\| |
| ---: | ---: | ---: | ---: |
| 0.05 | 63.435 | 0.862327 | 0.075650 |
| 0.05 | 45.000 | 0.758609 | 0.435581 |
| 0.10 | 63.435 | 0.713500 | 0.047439 |
| 0.10 | 45.000 | 0.231671 | 0.100727 |
| 0.20 | 63.435 | 0.786926 | 0.063325 |
| 0.20 | 45.000 | 0.361320 | 0.166744 |

  The baseline row is `0.10 m`; it is not used to tune any decision threshold.

## D9 — Per-link D9 depth uses 1% power convergence, with a relay double-bounce requirement

- **Type:** interpretation.
- **What:** Phase 5 uses `target=3`, `background=2`, `downlink=2`, `uplink=2`.
  The declared rule is the first depth whose next-depth total path power
  changes by less than `1%`. For the relay links, depth is raised to at least
  `2` because the threat model must retain the first-order wall paths and the
  two wall-wall double-bounce paths, not only the direct path.
- **Why:** The strict per-link rule gives relay depth `1` (`downlink`
  `d1->d2 = 0.0033771`, `uplink = 0.0033771`), but a depth-1 relay would
  discard the wall-return paths needed for the two-action wall test. At depth
  `2`, the raw-path `d2->d3` change is `0.0005584` for both relay links, still
  below `1%`. Target depth `3` has raw `d3->d4 = 0.0014610`; background depth
  `2` has raw `d2->d3 = 0.0037175`. Both are below `1%`.
- **BRIEF section:** §7.1 actual Sionna components; §7.2 received-signal
  model; §15 required validation tests.
- **Phase 5 evidence:** `results/phase4/geometry_reference.json` and
  `results/phase5/phase5_summary.json` contain the executed depth tables and
  `selected_depths`; `depth_power_decision` raises if the configured depths
  differ from the computed depths. The P1 duplicate audit found no duplicate
  paths at depths 3 or 4, so no D10 deduplication entry is needed.
- **Phase 5 C2 correction:** raw path power is not the power received under
  the actual beams. At the configured depths, the beam-weighted checks give:
  - wall-beam relay input `d1=7.809439e-13 W`, `d2=9.584364e-13 W`,
    `d3=9.632587e-13 W`, `d4=9.656921e-13 W`; `d1->d2=22.727944%`,
    `d2->d3=0.503141%`, `d3->d4=0.252617%`;
  - direct-beam relay input `d2->d3=2.142e-06`, `d3->d4=1.491e-07`;
  - target wall-gate coherent power is `2.723302e-13 W` at depths `2`, `3`,
    and `4` under the wall beam, and `1.048250e-14 W` at all three depths
    under the direct beam;
  - background direct-gate and wall-gate coherent powers are zero at depths
    `1` through `4` for both beams.
  Starting from the configured depths and raising only when a next-depth
  beam-weighted change reaches `1%`, the selected depths remain
  `target=3`, `background=2`, `downlink=2`, `uplink=2`.
- **C1 correction:** the gate-consistent relay pairs use the measured direct
  uplink at AoA `0 deg`, not the wall uplink at `±45 deg`. The selected
  wall-downlink group contains both first-order wall paths at
  `141.519251 ns`; with the direct uplink at `100.069229 ns`, the measured
  required added delay is `182.311567 ns` (analytic
  `423.900055 - 141.519251 - 100.069229 = 182.311575 ns`), a shift of
  `-17.826892 ns`. The gate-coherent required gain change is `47.279065 dB`;
  the required relay TX power using its total input is `5.483458e-06 W`,
  below the `0.01 W` cap. The previous `140.861545 ns` value used a wall
  uplink and did not lie at the verifier gate AoA.

The selected background depth `2` also excludes the depth-4 four-bounce
wall-wall monostatic clutter at `400.2769173894194 ns`; that path appears only
at depth 4 and is delay-coincident with the direct target return. It is
recorded as a diagnostic, not used in the selected Phase 5 background.

## Phase 6 pre-declaration fixes

These changes were made before any Phase 6 trial data were inspected. They are
declared processing choices, not post-hoc tuning.

### P6-1 — Blocked-target test is non-vacuous

- **Type:** verification design.
- **What:** The blocked-target test now checks measured quantities, not only
  the null-threshold `UNRESOLVED` label. The declared noise-consistency
  criterion is `E_W - N_W <= 3 N_W`, a one-sided three-sigma interval around
  the recorded-calibration noise mean. It also requires
  `S_blocked < 0.01 S_unblocked` for the same seeds. A test-only threshold
  fixture (`supported_min=0.5`, `rejected_max=0.05`, `noise_sigmas=3`) is used
  only inside `tests/test_phase6.py`; it is never written to
  `configs/pilot.yaml`. With that fixture the blocked case must return
  `UNRESOLVED` with reason `wall_energy_consistent_with_noise`, and the
  unblocked target must return `SUPPORTED`. No attack label exists in the
  decision function.
- **Why:** With `thresholds: null`, every nominal status is `UNRESOLVED`, so a
  status-only blocked test would pass even if the receiver saw a wall return.
- **BRIEF section:** §10.3 evidence score; §10.4 decision rule; §15 tests.
- **Evidence:** The criterion and test-only thresholds are in
  `tests/test_phase6.py`; `classify_evidence` is the shared decision mapping.

### P6-2 — Guard interval is disjoint and retains its noise-only purpose

- **Type:** processing declaration.
- **What:** The guard interval is `[550, 650] ns`, disjoint from the
  `[300, 500] ns` delay search and from the direct/wall gates. Its sole purpose
  is a recorded noise sanity check; evidence noise still comes from the
  recorded noise calibration. `ProcessingChoices` rejects any guard that
  overlaps the search window, and Phase 6 checks that both frozen gates are
  disjoint from the guard. The guard median must lie within
  `[0.5, 2.0]` times the calibration estimate.
- **Why:** The previous `[250, 350] ns` guard overlapped the search window,
  so it could not be an independent noise check and could contaminate
  acquisition.
- **BRIEF section:** §10.3 evidence score; §10.4 decision rule.
- **Evidence:** `configs/pilot.yaml`; `receiver.guard_noise_power`;
  `phase6._guard_disjoint_from_gates`.

### P6-3 — Candidate uncertainty comes from the measured rule

- **Type:** processing declaration.
- **What:** Candidate range uncertainty is
  `sigma_range = max(0.5 c sigma_tau, 0.05 m)`, where `sigma_tau` is the
  measured candidate delay uncertainty from
  `psf_half_width_over_sqrt_2_snr`. The fixed `1.2 m` (later `0.4 m`) value is
  removed. The floor is declared explicitly in
  `phase6.diagnostic.candidate_range_uncertainty_floor_m: 0.05`. Phase 6
  records and prints the measured standard deviation, the floor, and the used
  value; both the robust-beam ensemble and the wall gate use the used value.
- **Why:** The robust ensemble and gates must propagate the measured candidate
  uncertainty, not an arbitrary fixed range error.
- **BRIEF section:** §10.2 gates; §10.2 measured point-spread width and
  candidate-estimation uncertainty.
- **Evidence:** `phase6.candidate_uncertainty` in
  `results/phase6/phase6_summary.json`.

### P6-4 — Hann delay and angle windows reduce clutter leakage

- **Type:** processing declaration.
- **What:** Delay processing uses a non-periodic Hann window over the 2048
  subcarriers, normalized to unit RMS. Angular processing uses a non-periodic
  Hann window over the 16 receive elements, normalized to unit L2 norm. Both
  are declared in `phase6.processing.delay_window: hann` and
  `aoa_window: hann`. The noise calibration, delay-Doppler transform, angular
  scan, guard check, and point-spread measurement all use these same windows.
  The window choice is rectangular-or-Hann; Hann is selected because the
  background wall clutter is about `32 dB` stronger than the target wall
  return.
- **Why:** A rectangular window leaks strong low-delay clutter into the wall
  gate. Hann suppresses that leakage at the cost of a wider measured PSF; the
  PSF is remeasured with the declared windows and drives the gate expansion.
- **BRIEF section:** §10.1 declared window and scaling; §10.2 measured
  point-spread width.
- **Evidence:** `phase6.processing` in
  `results/phase6/phase6_summary.json`; `receiver._delay_window` and
  `receiver._aoa_window`.

### P6-5 — Zero-padded PSF measurement and lag-axis verification

- **Type:** processing declaration and verification.
- **What:** The delay PSF peak and Hann width are measured from a finely
  zero-padded inverse transform (`1 << 16` bins) of the effective channel
  after the declared subcarrier window and receive steering. The angular PSF
  is measured on the declared 0.001 deg grid with interpolation. Candidate
  delay is refined with the same zero-padded profile after the angular
  estimate. The lag-to-delay mapping is
  `delay = lag / (fft_size * subcarrier_spacing_hz)`, which is `8.138020833 ns`
  at the pilot profile. A noiseless V1 check at
  `300.0, 400.276914, 423.900055, 441.726946, 500.0 ns` verifies acquisition,
  PSF peak, direct-gate center, and twin prediction; the measured maximum
  module error is below `0.02 ns` against the declared `0.1 ns` tolerance.
- **Why:** Sparse-grid interpolation underestimated the Hann delay width
  (`9.79 ns` versus `11.71 ns`) and a committed API note incorrectly stated a
  `4.883 ns` lag spacing. The wider measured width moves the direct and wall
  gates into a `1.27 ns` delay overlap while their centers remain `23.6 ns`
  apart; replay wall energy remains at the noise floor.
- **BRIEF section:** §10.2 measured point-spread width; §10.3 evidence score;
  §15 tests.
- **Evidence:** `src/isac_verify/axis.py`; `receiver._zero_padded_delay_profile`;
  `tests/test_phase6.py::test_delay_axis_agrees_across_receiver_modules`;
  `results/phase6/phase6_summary.json`.
