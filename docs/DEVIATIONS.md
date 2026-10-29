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
  Hann window over the 16 receive elements, normalized to unit L2 norm in the
  original declaration; the implemented normalization is unit RMS, corrected
  in P7-3. Both are declared in
  `phase6.processing.delay_window: hann` and
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

## Phase 7 pre-run corrections

### P7-1 — Map-free two-beam control uses one seeded extra look

- **Type:** experiment declaration.
- **What:** For map-free two-beam control, the transmitter uses the shared
  direct action (the normal 64-symbol probe) plus exactly one extra 64-symbol
  wall/scan action aimed at one angle chosen from the codebook
  `{-60, -50, ..., -10, +10, ..., +60}` degrees by a fixed seeded
  `random.Random(beam_selection_seed).choice(...)` rule. This is two beams in
  total, but only one extra action. Coarse-to-fine remains `4 x 8 + 2 x 16 =
  64` extra symbols after the shared direct action. Every method therefore
  spends `64` extra wall/scan symbols, and mapped/extra-action rows record
  `128` symbols total including the shared direct probe.
- **Why:** The original implementation sampled two directions and split the
  `64` extra symbols as `32 + 32`, which changed both the action count and the
  per-angle integration time relative to the mapped methods. The smoke stage
  needs an equal-extra-symbol control.
- **BRIEF section:** §12.3 map-free two-beam control; §13.1 smoke stage.
- **Evidence:** `phase7._mapfree_look_plan`;
  `phase7._scan_looks`; `beam.map_free.total_extra_symbols`.

### P7-2 — `S` and `wall_excess_snr` are separate, availability-masked columns

- **Type:** statistics and output schema correction.
- **What:** `S` is populated only for mapped methods with a twin reference.
  `wall_excess_snr = (E_W - N_W) / N_W` is computed the same predeclared way
  for every method that spends an extra wall/scan action, mapped or map-free.
  The direct-only method has no wall action. Each column has an explicit
  boolean audit flag, `S_available` and `wall_excess_snr_available`.
  Not-applicable entries are `NaN`, never `0.0`. Integrity checks require each
  score column to be finite wherever its flag is true and `NaN` wherever it is
  false. Aggregates and figures use only available rows and print the
  available counts.
- **Why:** The earlier plan used a single `score` column and `0.0` for
  direct-only, which mixed two different metrics and turned a missing
  wall-probe action into an apparent measured zero. This correction preserves
  a common wall-excess metric without inventing values.
- **BRIEF section:** §13.1 smoke outputs; §13.2 metrics; §14 reporting.
- **Evidence:** `metrics.integrity_checks`; `metrics.summarize_trials`;
  `report._score_figure`; `trials.parquet` columns `S`, `S_available`,
  `wall_excess_snr`, `wall_excess_snr_available`.

### P7-3 — Receiver noise normalization corrected to unit-RMS window and mean calibration

- **Type:** validity fix.
- **What:** The angular Hann window is normalized to unit RMS, not unit L2.
  `noise_calibration` returns the mean per-cell noise over
  `phase6.processing.noise_calibration_records: 64` records, not a median over
  four records. Phase 7 writes `noise_normalization.csv` with 128 independent
  noise-only records per method/gate variant, reporting mean `E/N̂`, its 95%
  interval, and the cells used; the implementation-validity check requires
  every interval to contain one.
- **Why:** The receive steering vector has unit norm. A unit-L2 angular window
  multiplied by that steering vector gives a sum of squares near `1/M`
  (`M=16`), so the calibrated no-target energy was about `1/16` too low. A
  four-record median added a further low bias. The pre-fix smoke therefore
  reported noise-only `E/N` near `0.1`, not `1`.
- **BRIEF section:** §10.1 declared window and scaling; §13.1 smoke outputs;
  §14 reporting.
- **Evidence:** Pre-fix direct `E/N=0.1001`, wall `0.0940`. Post-fix gate
  means span `0.9624–1.0202`; all ten 95% intervals contain one; maximum
  absolute deviation from one is `0.0376`. Phase 6 now reports
  `noise_calibration=1.884609e-17 W/cell from 64 records`; V2 measured
  `delay full=11.709431 ns, half=5.854716 ns; AoA full=11.004057 deg`;
  acquisition `peak/noise=41.865 dB`.
  Evidence: `results/phase7/noise_normalization.csv`;
  `results/phase7/report/report.md`; `results/phase6/phase6_summary.json`.

## Phase 7 validity fixes after review

### P7-4 — Independent action randomness with common random numbers

- **Type:** validity fix.
- **What:** Direct-action probe, receiver-noise, and relay-noise seeds are now
  one declared stream per record and are shared across methods, sigma settings,
  and worlds. The mapped wall probe and each map-free look use separate
  declared seed streams; coarse-to-fine predeclares four coarse and two
  refinement look streams per record.
- **Before:** Every method and every map-free look reused the same
  `probe_seed`, `receiver_noise_seed`, and `relay_noise_seed`, so different
  actions received identical symbol and noise realizations.
- **After:** `seed_manifest.json` declares `direct`, per-method
  `wall_action`, and per-look `mapfree` streams. `integrity_checks` verifies
  direct-seed agreement across methods within each record.
- **BRIEF section:** §6.1 waveform and seed declaration; §13.1 staged runs;
  §14.3 output files.
- **Evidence:** `src/isac_verify/phase7.py:129` (`_seed_manifest`),
  `src/isac_verify/phase7.py:651` (`_scan_looks`),
  `src/isac_verify/phase7.py:1153` (`run_smoke`), and
  `src/isac_verify/metrics.py:156` (`integrity_checks`).

### P7-5 — Per-record acquisition and per-trial planning

- **Type:** validity fix.
- **What:** Each record now estimates its candidate, point spread, direct gate,
  and direct energy from that record's own direct observation. Every mapped
  trial builds its plan, twin prediction, beam, measured-direction null, gates,
  and `r_DT` from that candidate and its own twin realization. Acquisition
  failures produce `ACQUISITION_FAILED` rows for every method and are counted.
- **Before:** One shared base acquisition per world was reused for all sixteen
  records, so candidate, point spread, and gates were constant within a world;
  the null depth was effectively constant at about `-69.9 dB`.
- **After:** Candidate delay has 16 distinct values per method/world/sigma cell,
  and per-trial null depth spans roughly `-102.9` to `-62.9 dB` across the
  saved smoke rows.
- **BRIEF section:** §8.2 imperfect-map construction; §9.1 direct observation;
  §9.2 twin-guided wall prediction; §13.1 staged runs.
- **Evidence:** `src/isac_verify/phase7.py:431` (`_acquire_record`),
  `src/isac_verify/phase7.py:534` (`_plan_for`), and the per-record loop in
  `src/isac_verify/phase7.py:1153`.

### P7-6 — Independent twin wall errors with shared standardized draws

- **Type:** validity fix.
- **What:** The two walls now receive independent standard-normal draws
  `z+` and `z-`. For each twin realization the same draws are reused across
  sigma, with `epsilon+ = sigma*z+` and `epsilon- = sigma*z-`. The declared
  corridor validity rule is checked for each trial's twin map.
- **Before:** One rigid delta was applied to both walls, so their errors were
  perfectly correlated and the two-sided corridor never changed width.
- **After:** `twin_realizations.csv` reports `z`, realized `epsilon`, and both
  wall offsets for all twelve sigma/realization combinations; all combinations
  pass corridor validity.
- **BRIEF section:** §8.2 imperfect-map construction; §9.2 twin-guided wall
  prediction.
- **Evidence:** `src/isac_verify/phase7.py:196` (`_twin_offsets`),
  `src/isac_verify/phase7.py:534` (`_plan_for`), and
  `results/phase7/twin_realizations.csv`.

### P7-7 — Remove `REJECTED` and use the three allowed evidence labels

- **Type:** validity fix.
- **What:** `classify_evidence` now returns only `SUPPORTED`, `UNRESOLVED`, or
  `ACQUISITION_FAILED`. Scores below the support threshold return
  `UNRESOLVED` with reason `below_support_threshold`; direct-only successful
  rows return `UNRESOLVED` with reason `no_wall_probe`.
- **Before:** Scores below `rejected_max` returned `REJECTED`, a label outside
  the declared decision set.
- **After:** The full smoke run has 576 `UNRESOLVED` rows and no `REJECTED`
  rows; `integrity_checks` enforces the allowed status set.
- **BRIEF section:** §10.3 evidence score; §10.4 allowed decisions; §13.2
  equal-budget comparisons.
- **Evidence:** `src/isac_verify/verification.py:180`
  (`classify_evidence`), `src/isac_verify/metrics.py:156`
  (`integrity_checks`), and `tests/test_phase7.py`.

### P7-8 — Restrict `NOT_OBSERVABLE` to twin planning outcomes

- **Type:** validity fix.
- **What:** `NOT_OBSERVABLE` is now produced only when a twin-based planner
  finds no usable wall path. Direct-only rows use
  `path_availability="NOT_ATTEMPTED"` and reason `no_wall_probe`. Both map-free
  methods use `path_availability="NOT_APPLICABLE_NO_TWIN"` on every row.
  Metrics and the status-rate figure now report acquisition failure,
  `NOT_OBSERVABLE`, `NOT_ATTEMPTED`, and `NOT_APPLICABLE_NO_TWIN` separately.
- **Before:** Direct-only rows were `NOT_OBSERVABLE`, and map-free availability
  was set from the measured excess SNR, mixing missing-map semantics with a
  data-dependent outcome.
- **After:** The smoke run has 96/96 direct-only `NOT_ATTEMPTED` rows,
  192/192 map-free `NOT_APPLICABLE_NO_TWIN` rows, and no measured-data
  `NOT_OBSERVABLE` rows.
- **BRIEF section:** §10.3 evidence score; §12.2 imperfect-map evaluation;
  §14.1 core metrics.
- **Evidence:** `src/isac_verify/phase7.py:1153` (`run_smoke`),
  `src/isac_verify/metrics.py:34` (`summarize_trials`),
  `src/isac_verify/report.py:356` (`_status_figure`).

### P7-9 — Restore Experiment A Figure 2 and keep Experiment B separate

- **Type:** validity fix.
- **What:** Figure 2 is now the Experiment A effective-response difference for
  all 70 direct, dither, and random beams, saved as
  `experiment_a_beams.csv`. The paired target-versus-replay wall-energy scatter
  is retained as `02b_experiment_b_paired_wall_energy`.
- **Before:** Figure 2 was titled Experiment A but plotted Experiment B paired
  wall energies.
- **After:** `02_experiment_a` plots the c128 and Sionna relative effective-
  response differences across every beam index, grouped by beam family.
- **BRIEF section:** §11 Experiment A; §14.4 figures.
- **Evidence:** `src/isac_verify/phase7.py:1120`
  (`_write_experiment_a_beams`), `src/isac_verify/report.py:126`
  (`_experiment_a_figure`), and `src/isac_verify/report.py:175`
  (`_experiment_b_figure`).

### P7-10 — Compare mean guard power with mean calibration

- **Type:** validity fix.
- **What:** `guard_noise_power` now returns the mean per-cell guard power, and
  the Phase 6 guard ratio compares that mean with the mean calibration. The
  expected noise-only ratio is one. Bounds are declared in
  `configs/pilot.yaml` and repeated in the Phase 7 report.
- **Before:** The Phase 6 diagnostic compared a single-realization median
  guard power with the mean calibration; for exponential cell power the
  expected median/mean ratio is `ln 2 ≈ 0.693`, not one.
- **After:** Phase 6 reports guard mean `1.692430e-17 W/cell`, ratio `0.898027`,
  within the declared `[0.5, 2.0]` bounds. The report states the statistic and
  bounds.
- **BRIEF section:** §10.1 effective-channel processing; §13.1 staged runs.
- **Evidence:** `src/isac_verify/receiver.py:792` (`guard_noise_power`),
  `src/isac_verify/phase6.py:784`, and
  `src/isac_verify/report.py` (guard-statistic section).

### P7-11 — Add Phase 8 runtime and memory projection without running it

- **Type:** validity fix.
- **What:** The report projects the declared Phase 8A grid
  `12 x 5 x 32 x 2 x 6 = 23040` trials and the `24`-geometry held-out grid
  `46080` trials from measured smoke throughput. It also reports process-local
  peak reserved memory, device total, and headroom.
- **Before:** The report contained smoke timing but no Phase 8 wall-clock or
  held-out projection and no memory-headroom statement.
- **After:** From 576 smoke rows in `247.997 s`, the warm-median projection is
  `2.411 h` for Phase 8A and `4.813 h` for the held-out grid; the conservative
  end-to-end projection is `2.756 h` for Phase 8A. Neither grid was executed.
- **BRIEF section:** §13.2 equal-budget comparisons; §14.2 runtime and H100
  metrics.
- **Evidence:** `src/isac_verify/report.py:874` (projection section);
  `results/phase7/timing.json`; `results/phase7/report/report.md`.

## Phase 7 validity-fix verification

- `ISAC_DEV=1 scripts/drun.sh python -m isac_verify run --config
  configs/pilot.yaml --stage smoke --output results/phase7`:
  `576/576` rows, 36/36 cells, integrity true, `SMOKE_COMPLETE`.
- `ISAC_DEV=1 scripts/drun.sh python -m isac_verify validate --config
  configs/pilot.yaml --require-h100 --output results/validate`:
  `validation status: PASS`, `phase 6 status: PASS`.
- `ISAC_DEV=1 scripts/drun.sh python -m pytest -q tests`:
  `44 passed`.

## Phase 8A pre-calibration notes

### P8-1 — Twin reference detectability uses the twin wall reference

- **Type:** validity fix (applied before any calibration trial).
- **What:** `evidence_score` now marks the twin reference detectable only when
  the twin wall energy `E_W,twin` exceeds the 10 dB floor over `N_W_hat` and the
  twin direct energy exceeds the same floor over `N_D_hat`.
- **Before:** `reference_detectable` tested only `E_D,twin > floor * N_D_hat`,
  which duplicates the direct-reliability check and never tested the predicted
  wall reference path that `r_DT` is built from.
- **After:** a weak predicted wall path now returns UNRESOLVED instead of
  letting `S = (E_W - N_W)/(E_D - N_D)/r_DT` divide by a near-zero `r_DT`.
- **BRIEF section:** §10.3 ("if ... r_DT predicts an undetectable reference
  path, return unresolved status").
- **Evidence:** `src/isac_verify/verification.py` (`evidence_score`); the full
  Phase 7 smoke rerun after this change is numerically identical in all 56
  pre-existing trial columns (no smoke trial was affected).

### P8-2 — Per-record trial function shared by smoke and calibration

- **Type:** implementation refactor, no behavior change.
- **What:** the per-record body of `run_smoke` is now
  `phase7.evaluate_record`; the calibration run calls the same function with
  per-geometry configuration, seeds, and extra columns (per-`c` evidence,
  coverage flags, noise-only floor excess).
- **Evidence:** a full Phase 7 smoke rerun matches the approved run in all 56
  pre-existing trial columns (`rtol=1e-5`).

### P8-3 — Reading of rules 3 and 6 where the plan is silent

- **Type:** interpretation (most conservative reading, BRIEF §10.2/§10.3).
- **What:** (a) The floor `k` is computed for the three mapped methods, the
  only methods whose decision uses it; the map-free methods use the common rule
  (rule 5), which has no floor. (b) The twin interval inflated by `c` is the
  delay ensemble interval; the AoA gate has no twin-ensemble component, so
  `c` leaves it unchanged. (c) The `>= 99%` coverage requirement is evaluated
  on the primary method (`imperfect_robust`) pooled over sigma and geometries;
  per-method x sigma coverage is reported for all mapped methods. `c` is one
  global value, so it also widens the delay gate of the other mapped methods
  where their twin interval is non-degenerate (the narrow planner uses
  `sigma_plan = 0`, so `c` cannot change its interval).
- **Why:** the plan states one frozen `c` and a pooled requirement; the narrow
  planner's degenerate interval cannot be repaired by inflation, so tying `c`
  to it would make the rule unsatisfiable by construction.
- **Evidence:** `docs/ANALYSIS_PLAN.md` operational definitions;
  `src/isac_verify/calibration.py`.

## Phase 8A validity fix after the first calibration run

### P8-4 — Delay refinement used an angular-blind profile

- **Type:** validity fix (receiver bug). It was found after the first Phase 8A
  calibration run, using calibration geometries only. No analysis-plan rule
  changed (see `docs/ANALYSIS_PLAN.md`, Amendment 1).
- **Where:** `src/isac_verify/receiver.py`, `_zero_padded_delay_profile`
  (before the fix: lines 981-1002 at commit `4e44d3b`; introduced in
  `866c494`). It is used by `estimate_candidate` (candidate delay) and by
  `measure_point_spread` (PSF delay width).
- **Mechanism:** `h_eff` has shape `[symbols, rx, freq]`. The old code
  steered with `vector.conj()[:, None] * h_eff`, then `.sum(dim=0)` summed over
  **symbols**, not RX elements. After the zero-padded IFFT,
  `.abs().square().sum(dim=0)` added the RX elements **incoherently**. Steering
  only rotated each element's phase, so the refined delay profile was
  omnidirectional, without the AoA window. Its argmax was also taken over the
  whole delay axis. The candidate AoA came from the coarse angle-delay map,
  but the candidate delay came from this omnidirectional profile.
- **Effect:** at the target delay the reciprocal-wall pair (TX at 0 deg with
  full beam gain, arriving at +/-theta) adds about `2|a_rec|^2 / |a_dir|^2` of
  the direct return in that profile. For `2h/D` below about 0.47 this ratio
  exceeds 1, and the candidate delay jumped to the wall-path delay.
  - The jump happened in 5 of 12 calibration geometries (`cal_05`-`cal_08`,
    `cal_10`) in every target-world record, and in 1 of 32 records of `cal_04`.
    It never happened in the replay world.
  - It caused the gate-coverage failure (rule 6: 0.72-0.81 for c = 1-2) and
    the `imperfect_robust` `s_hi` = 103.
- **Evidence:** `results/phase8a_diag/` (`probe*.py`, `probe*_report.json`,
  `*_stdout.log`).
  - `cal_05` record 0 reproduces the run exactly: candidate 454.8450 ns,
    `E_D` 1.5090e-12; true direct delay 438.101 ns, true wall delay 454.854 ns.
  - In the coarse map at AoA 0 the direct return is 34.54 dB above the wall
    bin (`cal_00`: 17.61 dB).
  - In the old dense profile the wall is 0.376 dB above the direct return
    (`cal_00`: -0.742 dB). The predicted `2|a_rec|^2/|a_dir|^2` is +0.443 dB
    (`cal_00`: -0.868 dB).
  - Solver amplitudes match the analytic `lambda sqrt(sigma)/((4 pi)^1.5 D^2)`
    and `lambda sqrt(sigma)|Gamma_TE|/((4 pi)^1.5 L1 D)` to 4 digits.
  - The replay world (`cal_05` record 0) acquires at 438.111 ns.
- **Fix:**
  1. `_zero_padded_delay_profile` now follows the coarse-map signal path: Hann
     delay window, coherent sum over symbols (the zero-Doppler bin), Hann AoA
     window, coherent RX combining with the steering vector at the candidate
     AoA, Sionna's `1/(N S)` scaling, a zero-padded IFFT (pad 65536, 32x the
     coarse lag grid), then `|.|^2`.
  2. The dense argmax is restricted to +/-1 coarse lag bin around the
     coarse-map peak (`_dense_peak_index`). This is declared here and applies
     to both the candidate delay and the PSF peak.
  3. Named-shape assertions (`_require_shape`) are added wherever an axis is
     reduced.
- **Reduction audit (`receiver.py`):**

  | Function | Reduction |
  |---|---|
  | `cfr_from_paths` | `matrix @ weights` sums TX elements (beamforming) |
  | `make_observation` | none (`h_eff = y * conj(symbols)` per element) |
  | `delay_doppler_map` | Sionna transform over freq (to lag) and symbols (to Doppler) on `[rx, symbols, freq]`; output asserted `[rx, doppler, lag]` |
  | `angular_delay_map` | sums RX elements of the zero-Doppler `[rx, lag]` map with AoA window and steering; output asserted `[lag, aoa]` |
  | `noise_calibration` | mean over all `[rx, lag]` zero-Doppler cells of per-element noise maps |
  | `guard_noise_power` | mean over all `[rx, lag]` cells of the guard lags |
  | `gate_energy` | sums `[lag, aoa]` cells inside the gate |
  | `estimate_candidate` / `_peak_index` | argmax over `[lag, aoa]` |
  | dense profile | sums symbols, then RX (fixed) |
  | `measure_point_spread` | width walk on the `[aoa]` row and on the dense delay profile |

  The per-element noise means equal the combined-map noise per cell, because
  the steering vector has unit norm and the AoA window has unit RMS.

  Two findings are recorded but not changed:
  - `docs/API_NOTES.md` says the AoA window is unit-L2. The code uses unit RMS,
    and the noise calibration is consistent with the code.
  - The lower edge of `delay_search_ns` (300 ns) is never applied. The maps
    and the argmax cover lag 0 to `l_max`, and only `l_max` (500 ns) is
    enforced. Background clutter (87-225 ns, AoA +/-90 deg) stays below the
    detection floor in the probe, so this did not affect any calibration
    trial.
- **Tests** (`tests/test_receiver_refinement.py`; all three fail at `4e44d3b`
  and pass after the fix):
  - (a) The dense profile on the coarse grid equals the coarse map at that AoA.
    Measured before the fix: max relative error 0.604. After the fix:
    2.0e-7 at 0 deg and 1.9e-7 at 30 deg (`results/phase8a_diag/test_a_values.json`).
  - (b) A noiseless `2h/D = 0.40` geometry gives a candidate within 0.1 ns of
    the true direct delay in the target and replay worlds. Before the fix: the
    target candidate was 450.330 ns against a true 433.633 ns.
  - (c) With a 0-deg path and a 3x stronger 30-deg path, the dense profile at
    0 deg peaks at the 0-deg delay. Before the fix: it peaked at 439.96 ns, on
    the 30-deg path.
- **PSF** (`results/phase8a_diag/psf_compare.json`): the direct-path PSF
  delay width is 11.709316 ns new and 11.709316 ns old (nominal geometry);
  theory is 11.709580 ns. The PSF is measured on the single twin direct path,
  where the old profile was already correct, so the gate widths are unchanged.
- **Re-validation after the fix:** full pytest passes (64 passed). Phase 6
  `validate` passes (`results/validate_p8fix`), with E_D, E_W and S equal to
  the old run to about 1e-6 relative and no status changes. The Phase 7 smoke
  (`results/phase7_p8fix`, 576 rows, `SMOKE_COMPLETE`) matches `results/phase7`
  trial by trial: 0 of 576 statuses changed (all `UNRESOLVED`, as before),
  acquisition stays 1.0 in all 36 cells, the largest candidate-delay change is
  0.048 ns (median 0.019 ns), and the largest S_median change is in
  imperfect_narrow replay at sigma 1.0 (7.7e-5 to 5.8e-5). The nominal
  geometry has `2h/D` above the capture region, so it was not affected.
- **First calibration run:** moved to `results/phase8a_invalid_v1/`, kept and
  never used. Calibration is re-run from scratch with the same seeds
  (`docs/ANALYSIS_PLAN.md` Amendment 1).

### P8-5 — Declared delay search window enforced in acquisition

- **Type:** validity fix. It closes the two audit findings recorded in P8-4
  and is applied before the calibration re-run. No analysis-plan rule and no
  declared value changed: `phase6.processing.delay_search_ns` stays
  `[300.0, 500.0]`.
- **What was wrong:** `delay_search_ns` only set the map's last lag,
  `ceil(500 ns * W)` = lag 62 = 504.56 ns at 8.138 ns spacing. The coarse
  argmax covered lag 0 to 504.56 ns, so neither edge of the declared window
  was applied as declared: the lower edge was ignored, and the upper edge
  admitted one lag past 500 ns. (P8-4 says "only `l_max` (500 ns) is
  enforced"; the measured last lag is 504.56 ns.)
- **Fix:** `estimate_candidate` takes the coarse argmax only over lags with
  delay in [300, 500] ns, both edges inclusive (`_search_lag_range`). The
  dense refinement window (+/-1 coarse bin, P8-4) is further clipped to the
  same [300, 500] ns. The maps and their noise calibration are unchanged; log-parabolic
  interpolation still reads the neighbouring cells. Every acquisition path
  (Phase 6, Phase 7, Phase 8A, delay-axis check) calls `estimate_candidate`,
  and it applies the same rule in the target and replay worlds.
  `measure_point_spread` is not acquisition and is unchanged.
- **Tests** (`tests/test_receiver_refinement.py::test_candidate_respects_declared_delay_search_window`,
  coarse-only and dense refinement; both cases fail at `eb73949`):
  - A 0-deg return at 200 ns, 3x the direct amplitude, plus the direct return
    at 400 ns: the candidate is at 400 ns. Before the fix: 199.918 ns.
  - A 0-deg return on the last map lag (504.56 ns), 1.5x the direct
    amplitude, plus the direct return at 400 ns: the candidate is at 400 ns.
    Before the fix: 504.557 ns. Its measured leakage one lag inside is
    -6.0 dB (`results/phase8a_diag/window_probe.py`), so it does not win the
    in-window lag.
  - A return at 515 ns already passed at `eb73949`: its main lobe peaks outside
    the map, so it is not a useful regression case and was not kept.
- **Documentation:** `docs/API_NOTES.md` now says the AoA window is unit-RMS
  (documentation only; the code was already unit-RMS, per P7-3).
- **Re-validation:** full pytest passes (66 passed). Phase 6 `validate`
  passes (`results/validate_p8fix2`), and every numeric field of
  `validation_summary.json` (13578) and `phase6_summary.json` (472) equals the
  P8-4 run to within 1e-9 relative. Only the output paths differ. The nominal
  direct return (400.28 ns) and wall return (423.90 ns) both lie inside
  [300, 500] ns, so the smoke was not re-run.
- **Run record:** `calibrate` now samples NVML's process list on device 0
  every 30 s during the run and writes the per-PID summary to
  `timing.json` as `gpu_processes_observed` (host PIDs, including the run's own
  process). It does not affect any computed value.

### P8-6 — Calibration geometries run in parallel processes on one GPU

- **Type:** execution change only. No computation, seed, rule or declared
  value changed.
- **What:** `calibrate --workers N` runs the 12 calibration slots in N worker
  processes on the same GPU, each owning a fixed subset of slots
  (`slot k` goes to worker `k mod N`). The main process runs preflight, writes the
  resolved config and seed manifest, samples NVML for the whole run, waits for all
  workers, and only then runs the unchanged analysis on the 12 shards. Each
  worker calls the same per-slot function (`_run_slots`) as the sequential
  path. Every worker recomputes the noise calibration from the declared seeds,
  and the main process refuses to analyze unless all workers' values agree
  exactly. A failed worker stops the run before analysis.
- **Why:** a single process kept GPU 1 busy only about 35% of the time,
  because each record launches many small kernels from Python. The calibration
  slots are independent, so running them side by side shortens the run without
  touching the computation.
- **Equivalence evidence** (`results/phase8a_diag/eq_seq`, `eq_par`;
  2 geometries x 2 records, run sequentially and with 2 workers): 240 rows in
  both. All 104 non-timing columns are identical (`discovery` compared as text,
  because its dicts contain NaN), and `thresholds.json` decision thresholds
  are identical. Only `trial_runtime_s` differs.
- **Run record:** the Phase 8A calibration uses `--workers 12` on GPU 1.
  `timing.json` records `workers`, `worker_slots`, per-worker wall time and
  PyTorch peaks, the agreed noise calibration, and the NVML process record for
  the whole device.

### P8-7 — Tag `pilot-frozen` moved from `712b789` to `0350377`

- **Type:** repository record only. No commit was rewritten.
- **What:** `pilot-frozen` was first created on `712b789 phase8a: frozen
  thresholds`, which froze the calibration v5 thresholds as written in rules 4-5
  (pooled over sigma). ANALYSIS_PLAN Amendment 2 then re-derived `s_lo` and
  `t_m` per sigma cell from the same calibration data. It was committed as
  `0350377 phase8a: frozen thresholds (amendment 2)`, and the lightweight tag
  was moved there with `git tag -f pilot-frozen 0350377`, with the user's
  approval. `712b789` stays in the branch history. The repository has no
  remote, so the old tag target was never published.
- **Why:** the plan names `pilot-frozen` as the frozen operating point, and
  the frozen thresholds are now the Amendment 2 values.
- **Held-out:** no held-out geometry was generated, solved or run before or
  during the move.
- **Stale artifact:** `results/phase8a/thresholds.json`, written by
  `calibrate` with the rule-4/5 values from before the amendment, was renamed
  to `thresholds_pooled_preA2.json`. The frozen values are only in
  `configs/pilot.yaml` `phase8.frozen`.

### P8-8 — Held-out harness added after the freeze (no computation change)

- **Type:** execution harness only. No receiver, planner, trial, seed, gate or
  threshold code changed.
- **What:** the frozen code (`0350377`) could not run the held-out split.
  `phase8a.py` hard-coded `SPLIT = "calibration"`, and no held-out command
  existed (the plan scoped held-out to Part B). Nothing read `phase8.frozen`:
  in-pipeline decisions read `phase6.evidence.thresholds` (null, so every trial
  row is UNRESOLVED / `thresholds_null`), and calibration decisions were applied
  afterwards by `calibration.classify_trials` with freshly fitted values. The
  harness now:
  - threads a `split` argument through the existing per-slot path
    (`run_geometry`, `_run_slots`, workers, seed manifest, geometry ids `ho_XX`);
  - unlocks the held-out stream only via `isac_verify heldout --approved`
    (the stream stays refused by default);
  - applies decisions in `analyze_heldout` from `phase8.frozen` only
    (`frozen_decision`; nothing is fitted on held-out data);
  - records the host git commit in `environment.json` (`source_commit`) and
    refuses to run held-out from an unknown or dirty tree (`scripts/drun.sh`
    passes `ISAC_GIT_COMMIT` / `ISAC_GIT_DIRTY`).
- **Equivalence evidence** (`results/phase8a_diag/eq_harness`, GPU 1):
  cal_00 x 2 records run on the harness code. All 104 non-timing columns of
  the 120 rows are identical, compared as text, to the same records of the
  calibration v5 shard produced by the frozen code. `analyze_heldout` was dry-run on
  a relabelled copy of those calibration rows
  (`results/phase8a_diag/analyzer_dryrun_calibration_relabelled`); no held-out
  geometry was drawn.
- **Tags:** `pilot-frozen` stays on `0350377`. The held-out run executes from a
  worktree at tag `pilot-heldout` (this commit).

### P8-9 — Stress harness (ANALYSIS_PLAN A3.3 / A5), inactive by default

- **Type:** execution harness only. No receiver, planner, trial, gate or
  threshold code changed. Stresses use `phase8.frozen` unchanged.
- **What:**
  - `isac_verify stress --level L --approved` writes `<output>/stress_config.yaml`
    (`stress.stress_config`) and runs it through the unchanged held-out
    harness (`run_calibration(split="heldout")`).
  - The override replaces `phase8.grid` with `phase8.stress.grid` (A3.3:
    sigma 0.5, 2 twin realizations, 8 records, methods narrow / robust /
    coarse_to_fine). For a bandwidth level it also sets
    `phase6.processing.profile` and the `bandwidth_profile` row label to the
    level.
  - The stress grid itself lives in new keys under `phase8.stress`; no
    existing key changed.
  - `run_geometry` now takes its method list from `phase8.grid.methods`
    (ordered as `METHODS`), and maps each sigma to its declared noise-only
    seed block via `phase8.stress_runtime.declared_sigma_map_m`. With the
    baseline config both reduce to the previous values.
  - `analyze_heldout` counts expected rows with `phase8.grid.methods`.
- **Scope:** only the A5.3 block 1 levels (`bw246`, `bw61`) are implemented.
  Later blocks add their levels before they run, and repeat the equivalence
  check below.
- **Twin index:** with 2 realizations a stress record `r` uses twin `r mod 2`;
  the main run used `r mod 4` (A5 operational note).
- **Equivalence evidence** (`results/phase8a_diag/eq_stress`, GPU 1): cal_00 x
  2 records on the stress-harness code, baseline config. All 104 non-timing
  columns of the 120 rows are identical (`compare_shards.py`) to the
  calibration v5 shard.

### P8-10 — Stress harness block 2: relay standoff sweep (ANALYSIS_PLAN A5.2)

- **Type:** execution harness only; inactive by default.
- **What:**
  - Levels are `standoff_{03,06,10,15,20,30}m` (`phase8.stress.standoff`).
  - `_run_slots` keeps the accepted held-out geometry, as drawn with its
    original relay, so `R_T` and `d` are the main run's. It then sets
    `R_A = R_T - Delta_R` and re-runs `prepare_geometry`, which re-solves the
    scene and re-matches the relay by the same rule.
  - An infeasible pair writes an empty shard with the rejection in its
    metadata; it is not redrawn.
  - Profile `baseline`; methods exact_oracle, imperfect_robust,
    mapfree_coarse_to_fine; the A3.3 grid otherwise.
- **Smoke test** (cal_00, 1 record, calibration split): at 3 m the relay sits
  at 64.76 m with 20.014 ns added delay (floor 20 ns). At 20 m it sits at
  47.76 m with 133.4 ns. The infeasible branch was not exercised.
- **Equivalence evidence** (`results/phase8a_diag/eq_stress2`): cal_00 x 2
  records on the baseline config. All 104 non-timing columns of the 120 rows
  are identical to the calibration v5 shard. 71 tests passed.

### P8-11 — Stress harness block 3: array calibration error D11 (ANALYSIS_PLAN A7.1)

- **Type:** execution harness only; inactive by default.
- **What:**
  - Level `array_error_d11` reads the existing `phase8.stress.array_calibration`
    values (0.5 dB, 5 deg, 4 realizations) plus the new key
    `phase8.stress.array_calibration_seed` (20261028).
  - `run_geometry` builds 4 perturbed copies of the true link matrices
    (`stress.perturb_links`: target and background `diag(e_RX) M diag(e_TX)`,
    downlink `M diag(e_TX)`, uplink `diag(e_RX) M`). Record `r` uses copy
    `r mod 4`, recorded in the extra column `array_error_realization`.
  - The planner, twin maps, PSF, receiver steering, noise calibration and the
    matched-relay calibration keep the ideal arrays.
  - A3.3 grid, baseline profile.
- **Smoke test** (cal_00, 1 record, calibration split): 6 rows, realization 0.
  E_D changed from 1.2617e-11 to 1.2497e-11 W (target). The mapped beams
  moved from +25.5 to -25.6 deg, so the planner chose the other wall side.
- **Equivalence evidence** (`results/phase8a_diag/eq_stress3`): cal_00 x 2
  records on the baseline config. All 104 non-timing columns of the 120 rows
  are identical to the calibration v5 shard. 72 tests passed.

### P8-12 — Stress harness blocks 4 and 5: probe power, wall thickness D8, target RCS (ANALYSIS_PLAN A8)

- **Type:** execution harness only; inactive by default.
- **What:**
  - Levels:
    - `probe_power_{m20,m10,0,p20}dbm` (new key `phase8.stress.probe_power_dbm`);
    - `wall_thickness_{50,200}mm` (existing key `phase8.stress.wall_thickness_m`);
    - `rcs_{p6,m6}db` (new key `phase8.stress.rcs_reflected_illumination_db`).
  - Probe power: `_run_slots` builds the direct beam at the level and
    overrides `waveform.nominal_total_power_dbm` in each geometry config
    after acceptance and relay calibration at 10 dBm (`stress.with_probe_power`).
  - Thickness: the stress config sets `phase8.geometry.wall_thickness_m`
    (read by `draw_geometry` -> `geometry_config`) and `wall.thickness_m`,
    so truth and twin both use it.
  - RCS: `stress.scale_reflected_illumination` scales the true target links
    whose interaction list has SPECULAR before SENSING. On cal_00 that is 8 of
    13 paths: `outgoing_wall_*` x2, plus 6 multi-bounce paths.
  - The A3.3 relay gain, material and blocker levels are absent (NOT RUN,
    Amendment 8).
- **Smoke tests** (cal_00, 1 record, calibration split):
  - −20 dBm: acquisition failed on record 0 in both worlds.
  - 50 mm: robust target S 0.888 -> 0.936.
  - RCS +6 dB: robust target S 0.888 -> 3.495 (3.9x); replay rows unchanged.
- **Equivalence evidence** (`results/phase8a_diag/eq_stress4`): cal_00 x 2
  records on the baseline config. All 104 non-timing columns of the 120 rows
  are identical to the calibration v5 shard. 73 tests passed.

### P8-13 — Candidate outside the twin's free space: UNRESOLVED / NOT_OBSERVABLE row instead of crash (ANALYSIS_PLAN A9)

- **Type:** planner behaviour change, only on records that previously raised an
  uncaught `SceneError`.
- **What:** `phase7._acquire_record` catches the direct-action PSF twin's
  `SceneError` and flags the record; `evaluate_record` then gives every method
  of that record the A9.1 row. `_plan_for` raises `TwinFreeSpaceError` for the
  wall-plan twin and only that method gets the A9.1 row. Row: status
  UNRESOLVED, `status_reason` `candidate_outside_twin_free_space`,
  `path_availability` NOT_OBSERVABLE, energies NaN.
- **Why:** levels −20 dBm (12 of 24 geometries) and −10 dBm (1 of 24, slot 20)
  crashed with `SceneError` for false candidates outside the corridor. Both
  partial outputs are kept as `FAILED_*` evidence and never merged.
- **Evidence:** test `test_candidate_beyond_wall_is_unresolved_not_observable`;
  74 tests passed; `results/phase8a_diag/eq_stress5`: cal_00 x 2 records,
  rows=120 compared_columns=104 differing=[] against the calibration v5 shard.

### P8-14 — C1 passive gate observed the opposite wall's reciprocal path (receiver AoA axis is mirrored)

- **Type:** defect in a post-freeze script (`scripts/wpc_passive.py`, `passive_gate`), found post hoc by D3.
  No package code, frozen decision or committed result is changed.
- **What:** `receiver.angular_delay_map` combines `sum_n a_n(theta) w_n h_n`, so a path arriving at link angle
  phi peaks at map angle −phi. D3 (`scripts/wpd_mechanism.py`) asserts this on measured noiseless maps: e.g.
  ho_10 uplink `wall_clutter_positive_wall` at link AoA +33.97° peaks at map −34.0°; all 8 checked paths
  mirror within the 0.25° grid. A11.1 centres the passive AoA window at the twin wall path's departure angle
  (`plan.predicted_aod_deg`, link convention). In map coordinates that window looks at the OTHER wall's
  reciprocal path: on ho_10, 94.9% of the relay's passive-gate energy is LoS-down × positive-wall-up while
  the plan's wall side is negative.
- **Effect:** the passive score is internally consistent (twin ratio r_pass, calibration and evaluation all
  use the same mirrored gate, and the corridor walls are symmetric in delay), but it is not the "same wall
  side as the active plan" that A11 describes. The C1 numbers stand as the result of the gate as built; their
  sensitivity to the mirror was NOT measured (it needs new GPU runs).
- **Frozen pipeline:** unaffected, because every candidate lies within ±0.11° of 0° (measured candidate AoA in
  every held-out and calibration row; true azimuth 0) and its AoA gates are centred on that measured AoA,
  where the mirror is the identity. Fact recorded in `docs/API_NOTES.md` (Phase 8C, mirrored angle map).
- **Decision (user, 2026-10-29):** no C1 re-run; C1 is internally consistent.

### P8-15 — Receiver angle-map sign fixed (ANALYSIS_PLAN A12.1; post-freeze code change)

- **Type:** post-freeze correction of package code (`src/isac_verify/receiver.py`), committed after
  Amendment 12 (da29b88) and before any A12 run. Frozen thresholds, config and earlier results unchanged.
- **What:** `angular_delay_map` and `_zero_padded_delay_profile` combined `sum_n a_n(theta) w_n h_n`
  (`steering_vector(theta).conj()`); they now combine `sum_n conj(a_n(theta)) w_n h_n`, matched to the RT
  arrival convention `h ~ a_RX(phi)` (API_NOTES). A path at link angle phi now peaks at map angle +phi.
- **Evidence (D3, P8-14):** on the pre-fix code, 8 of 8 ray-traced uplink paths peaked at −phi (e.g.
  ho_10 `wall_clutter_positive_wall` link +33.97° → map −34.0°).
- **Tests (A12.2, `tests/test_angle_convention.py`):** pre-fix: 3 failed — single paths +30° → −30.0° and
  −20° → +20.0°; asymmetric corridor (walls +12/−18 m, candidate 50 m at +10°): acquired azimuth −12.21°;
  with that sign check disabled, the planned (negative) side held 0.312 of its outgoing path in the wall
  gate and 8.6e-7 of its reciprocal path in the reciprocal gate, while the opposite side's held 0.588 and
  0.374. Post-fix: 3 passed — planned (negative) side 0.982 (wall gate) and 0.967 (reciprocal gate); opposite
  side 5.8e-5 and 5.8e-11. The test points the direct beam at the candidate bearing (+10°) and uses 50 m so
  the direct return lies inside `delay_search_ns` (a 40 m first draft acquired an in-window wall return).
- **Test fixture corrected:** `tests/test_receiver_refinement.py::_plane_wave` synthesized `conj(a(phi))`
  (the mirrored convention), so after the fix its "30°" path appeared at −30° and
  `test_dense_profile_matches_coarse_map_on_grid` compared a sidelobe column (1.71e-5 > 1e-5). The fixture
  now uses `a(phi)`, as its docstring states; tolerances unchanged; the test passes.
- **Still failing:** `tests/test_phase6.py::test_blocked_genuine_wall_path_matches_noise`. Phase 6 places
  the absorber on the POSITIVE wall path (`wall_paths.blocker_spec`). The corridor is symmetric (±15 m);
  the measured candidate azimuth bias flips sign with the fix (+0.008657° → −0.008657°, candidate y
  +0.0091 m → −0.0091 m), which flips the twin's strongest outgoing-wall path from positive (aod +26.557°)
  to negative (−26.557°). The probe now illuminates the unblocked wall: blocked E_W = 8.50e-12 W vs
  N_W = 5.03e-15 W (pre-fix 1.30e-14 W), score 1.068 (pre-fix 0.0010). Kept failing; not changed.
  The planner's wall side in symmetric corridors is decided by this sub-0.01° tie-break (see A12.3).
