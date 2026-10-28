# Phase 8 analysis plan (pre-registration)

Status: committed as `phase8a: analysis plan` before any calibration trial was
generated. Thresholds are chosen on the 12 calibration geometries only, frozen
into `configs/pilot.yaml` (`phase8.frozen`), and tagged `pilot-frozen`. The
24 held-out geometries are not generated, solved, or run until explicit human
approval. BRIEF references: §10.2, §10.3, §13.1–§13.3, §16.

## Rules

1. **Geometry sampling (BRIEF §13.1).** Candidate distance `U[50, 70] m`,
   relay distance `U[20, 40] m` collinear with the candidate, corridor
   half-width `U[12, 18] m`, wall thickness `0.10 m`. Separate declared seed
   streams for calibration (12 geometries) and held-out (24 geometries). A draw
   is rejected only if the matched fixed relay is infeasible (latency floor,
   causality, or 10 dBm cap); rejected draws are logged and replaced from the
   same stream. Wall-path observability is NOT a rejection criterion.
2. **Trial grid.** `sigma_map in {0, 0.25, 0.5, 1.0, 2.0}` m, 4 twin
   realizations per sigma (`eps+- = sigma * z+-`, P7-6), 32 records per world,
   all 6 methods, 10 dBm, baseline bandwidth, matched fixed relay with physical
   relay noise.
3. **Absolute wall-evidence floor `k` (per method).** Smallest `k` such that
   the noise-only false exceedance rate of `(E_W - N_W_hat) > k * N_W_hat` is
   `<= 1%`, from `>= 1000` noise-only records through the actual pipeline. No
   labels used.
4. **Primary decision (mapped methods).** SUPPORTED iff direct reliable, twin
   reference detectable, floor `k` passed, and `s_lo <= S <= s_hi`. Per method,
   pooled over all sigma and calibration geometries: `s_lo` = smallest value
   with replay-world SUPPORTED rate `<= 5%`; `s_hi` = 99.5th percentile of
   target-world `S` among floor-passing rows. Otherwise UNRESOLVED (or
   ACQUISITION_FAILED). One threshold set per method, not per sigma.
5. **Common rule for ALL wall-probing methods (mapped and map-free).**
   SUPPORTED_common iff `wall_excess_snr >= t_m`, with `t_m` = replay-world
   95th percentile on calibration, pooled over sigma. Used to compare methods
   at equal replay support.
6. **Gate coverage (BRIEF §10.2).** On calibration target-world trials, the
   fraction where the true target wall-path delay and AoA fall inside the wall
   gate, per method x sigma. Required `>= 99%` pooled. If not met, apply the
   smallest inflation factor `c in {1.0, 1.25, 1.5, 2.0}` of the twin interval
   that meets it, frozen; then recompute rules 3-5 with `c` applied.
7. **Methods.** Primary method: `imperfect_robust`. Secondary:
   `imperfect_narrow`. `exact_oracle` is a diagnostic only.
8. **Held-out decision criteria (BRIEF §16)**, evaluated with geometry-level
   bootstrap (resample geometries, 10,000 draws, 95% CI):
   - replay SUPPORTED rate: point estimate `<= 5%` and upper CI `<= 10%`;
   - genuine-target SUPPORTED rate at `sigma <= 0.5` reported with CI (useful =
     point estimate `>= 50%`);
   - "advantage" = primary target SUPPORTED rate minus map-free
     coarse-to-fine SUPPORTED_common rate (equal energy/symbols/time), at
     `sigma = 0.5` and `sigma = 1.0`; advantage holds if the CI lower bound
     `> 0` at either sigma;
   - robustness: report the same quantities under every stress (BRIEF §16)
     without re-tuning thresholds.

   CONDITIONAL GO only if all BRIEF §16 conditions hold; INVALID if any
   validity check fails; NO-GO / INCONCLUSIVE per BRIEF §16.
9. **Labelled follow-up stresses** (not baseline, reported separately, never
   used for tuning): wall thickness `{0.05, 0.20} m` (D8), and array
   calibration error (D11): i.i.d. per-element amplitude error `0.5 dB` rms
   and phase error `5 deg` rms applied to the PHYSICAL TX/RX arrays only
   (planner/twin unaware), 4 error realizations; report realized null depth
   toward the relay.

## Operational definitions (reproducibility only; no additional rule)

These fix how the rules above are computed so the result can be regenerated.
They do not add a decision criterion.

- **Seed streams.** The declared per-geometry seeds in `seeds.streams`
  (`physical_geometry`, `twin_error`, `probe_symbols`, `receiver_noise`,
  `relay_noise`, `diagnostic_beam_selection`, `noise_only_floor`) each hold 12
  calibration and 24 held-out values. Geometry slot `i` draws attempt `j` from
  `numpy.random.default_rng([physical_geometry[i], j])`; a rejected attempt is
  replaced by attempt `j+1` of the same slot. The three draws per attempt are,
  in order, candidate distance, relay distance, half-width. Record-level seeds
  are `stream_seed[i] * 100000 + block * 1000 + record * 10 + look`, where
  `block` indexes the action (direct, mapped wall probe per method, map-free
  look group) and noise-only blocks are separate. Direct-action seeds are
  common across methods, sigma, and worlds for a record; every extra action has
  its own block (P7-4). Twin `z+-` draws use `default_rng([twin_error[i], q])`
  for realization `q in 0..3`, independent per wall and common across sigma
  (P7-6). Record `r` uses realization `r mod 4`.
- **Relay feasibility** is `calibrate_matched_relay` on the drawn geometry's
  actual solved paths: a `LatencyFloorError` or `GeometryError` (latency floor
  or causality), or `power_feasible == False` (10 dBm cap), rejects the draw.
  Nothing else rejects a draw.
- **Receiver settings** are those frozen in Phase 6/7 and unchanged:
  delay-search window `[300, 500] ns`, AoA search `[-60, 60] deg`, detection
  floor 10 dB. A wall gate that extends past the delay-search window is
  truncated to the available lags; this is recorded per trial
  (`wall_gate_truncated`) and is not a rejection criterion.
- **Direct reliable** means the net direct gate energy exceeds the 10 dB floor
  over `N_D_hat`. **Twin reference detectable** means the twin wall reference
  energy `E_W,twin` exceeds the same 10 dB floor over `N_W_hat` (BRIEF §10.3,
  "r_DT predicts an undetectable reference path") and the twin direct energy is
  positive, so `r_DT` is finite. (This is validity fix P8-1 in
  `docs/DEVIATIONS.md`; it was applied before any calibration trial.)
- **Noise-only record** (rules 3). One receiver-noise-only action (no target,
  clutter, or relay signal; same OFDM profile, probe length, calibrated
  `N_hat`) per `(geometry, sigma, world, record)`, from the declared
  `noise_only_floor` stream, mapped by the actual receiver. Its wall-gate
  excess `x = (E_W - N_W_hat) / N_W_hat` is evaluated on that trial's own
  frozen wall gate for each mapped method. `k` is the order statistic
  `x_(ceil(0.99 n))` of the pooled sample for that method, so
  `P(x > k) <= 1%`. Floor passed means `(E_W - N_W_hat) > k * N_W_hat`.
  The calibration size is `12 x 5 x 2 x 32 = 3840` per mapped method.
- **`s_hi`**: `numpy.quantile(S, 0.995)` (linear) over target-world rows that
  pass the floor and have finite `S`. **`s_lo`**: with `n_allowed =
  floor(0.05 * N_replay)` where `N_replay` counts every replay-world row of the
  method (including ACQUISITION_FAILED), the smallest observed `S` among
  replay rows that pass direct-reliable, reference-detectable, floor, and
  `S <= s_hi` such that at most `n_allowed` such rows have `S >= s_lo`; if none
  exists, `s_lo = nextafter(max S, +inf)`.
- **`t_m`**: `numpy.quantile(wall_excess_snr, 0.95)` (linear) over
  replay-world rows with available `wall_excess_snr`, pooled over sigma and
  geometries, per wall-probing method (`exact_oracle`, `imperfect_narrow`,
  `imperfect_robust`, `mapfree_two_beam`, `mapfree_coarse_to_fine`).
  `direct_only` has no wall probe and no common rule. SUPPORTED_common rates use
  all rows of the world as denominator.
- **Twin interval inflation `c`.** The mapped planners build the wall delay
  gate from the central 99% (`[0.005, 0.995]` quantiles) of the twin ensemble,
  clamped to contain the predicted delay, then add the fixed measured-PSF and
  candidate-uncertainty expansion. `c` multiplies the two half-widths of that
  twin ensemble interval about the predicted delay before the fixed expansion is
  added. The AoA gate has no twin-ensemble component (the predicted arrival
  direction is the measured candidate direction), so `c` does not change it.
  `c = 1.0` reproduces the Phase 7 gate. The calibration run evaluates all four
  candidate values on the same observations; map-free gates have no twin
  interval and are unchanged.
- **Coverage truth** is evaluator-side: the true outgoing-wall path
  (`outgoing_wall_<side>_wall`) of the plan's wall side from the true solved
  target scene. A trial is covered when its true delay is inside the delay
  gate and its true AoA is inside the AoA gate. Coverage is computed on
  target-world trials of the mapped methods. The `>= 99%` requirement is
  applied to the primary method (`imperfect_robust`) pooled over sigma and
  geometries; per-method x sigma coverage is reported for all mapped methods.
  If no candidate `c` meets it, `c = 2.0` is applied and the failure is
  reported as unmet.
- **Rates** (SUPPORTED, SUPPORTED_common) use every row of the cell as
  denominator, so ACQUISITION_FAILED and unavailable-path rows count as not
  supported.
- **Descriptive vs gating labels in rule 8.** "Useful" (`>= 50%`) is a
  descriptive label. CONDITIONAL GO additionally requires the BRIEF §16
  provisional engineering targets (target SUPPORT `>= 0.80`, attack SUPPORT
  `<= 0.10`, NOT_OBSERVABLE `<= 0.20` in the declared unblocked subset).
- **Part A scope.** Part A runs only the baseline calibration geometries.
  The rule-9 stresses and all held-out evaluation are Part B and are not
  implemented or executed here.

## Amendment 1 — 2026-10-27

- **No rule in this plan changed.** No threshold, candidate value, seed,
  geometry, coverage definition or decision rule was edited, and no
  `phase8` config key changed.
- **What changed:** a receiver bug was fixed (`docs/DEVIATIONS.md` P8-4).
  The dense delay refinement summed the channel over OFDM symbols instead of
  RX antennas. That made it angularly blind, so in narrow corridors
  (`2h/D` below about 0.47) the candidate delay locked onto the reciprocal
  wall pair instead of the direct return. The refinement now follows the
  coarse angle-delay map's signal path, and its argmax is restricted to
  +/-1 coarse bin around the coarse peak.
- **Declared delay search window enforced** (`docs/DEVIATIONS.md` P8-5):
  acquisition now applies the declared `delay_search_ns` window
  [300, 500] ns at both edges, identically in both worlds. Before this, the
  lower edge was not applied and the map extended to 504.56 ns. The declared
  value itself is unchanged.
- **Calibration re-run:** the first calibration run (harness `4e44d3b`) is
  invalid. It is kept in `results/phase8a_invalid_v1/` and is not merged,
  reused or consulted for any threshold. Calibration is re-run from scratch
  with the same declared seeds and the same 12 calibration geometries, on the
  code that includes both fixes. A second attempt, started on `eb73949` before
  the window fix, was stopped within minutes, before any shard completed, and
  its partial output was deleted unused. A third attempt on `e241eaf`, on the
  shared GPU 0, was stopped after one geometry because of GPU contention, and
  its partial output was deleted unused. The calibration was restarted from
  scratch on the idle GPU 1 (H100 NVL, UUID
  `GPU-509d8ab3-0c8e-8b98-fbf8-0c55f6ba9058`). That fourth attempt was stopped
  before any geometry completed, and its output was deleted unused. The run
  was restarted from scratch with the 12 slots in parallel worker processes
  on the same GPU (`docs/DEVIATIONS.md` P8-6). This is an execution change
  only; the computation and seeds are unchanged.
- **Held-out:** no held-out geometry has been generated, solved or run.

## Amendment 2 — 2026-10-27

Written after calibration v5 (commit `8057404`, `results/phase8a`, 23040
rows, `c = 1.0`) and before any held-out geometry was generated, solved or
run. It uses calibration data only. The thresholds first frozen in `712b789`
(tag `pilot-frozen` at that commit) applied rules 4-5 as written; this
amendment supersedes their `s_lo` and `t_m` values. `k`, `s_hi` and `c` are
unchanged. No trial was re-run. The values come from
`scripts/phase8a_amendment2.py` applied to the existing
`calibration_trials.parquet`. That script first re-classifies every row with
the previous thresholds and checks that this reproduces the committed v5
`rates.csv` exactly.

- **A2.1 Like-for-like contrast (secondary).** In addition to the rule-8
  advantage (primary SUPPORTED minus map-free coarse-to-fine
  SUPPORTED_common), report the contrast in which `imperfect_robust` and
  `mapfree_coarse_to_fine` both use SUPPORTED_common, at the same sigma cells
  and with the same bootstrap. It is a secondary analysis; the primary
  contrast and its decision rule are unchanged.
  *Evidence:* the primary contrast compares a two-sided band rule
  (`s_lo <= S <= s_hi` plus floor `k`) against a one-sided energy rule. On
  calibration, the two rules gave different pooled replay rates
  (imperfect_robust SUPPORTED 0.050 and SUPPORTED_common 0.050 against
  coarse-to-fine SUPPORTED_common 0.052), and at sigma = 0 they gave
  different per-cell rates for the same method (robust 0.049 SUPPORTED vs
  0.081 SUPPORTED_common). So part of the primary contrast can come from the
  choice of rule rather than from the twin.
- **A2.2 Calibration granularity.** `s_lo` (each mapped method) and `t_m`
  (every wall-probing method) are the smallest value whose replay-world rate
  (SUPPORTED, or SUPPORTED_common) is `<= 5%` in EVERY sigma cell. Each cell
  uses the operational definition of rule 4 (`n_allowed = floor(0.05 * 384)
  = 19`, all replay rows of the cell as denominator); the frozen value is the
  largest of the per-cell values. `t_m` changes from a pooled 95th percentile
  to this per-cell rule.
  *Evidence:* with pooled calibration, per-sigma replay rates reached 0.076
  (imperfect_narrow SUPPORTED at sigma = 0; imperfect_robust SUPPORTED at
  sigma = 2). imperfect_robust SUPPORTED_common reached 0.081 at sigma = 0.
  Pooled `t_m` left the replay SUPPORTED_common rate at 0.052 for the oracle
  and both map-free methods. Calibrating exactly at 5% pooled makes a held-out
  point estimate `<= 5%` roughly a coin flip.
- **A2.3 Held-out acceptance clarification.** The rule-8 replay criterion
  (point estimate `<= 5%`, upper 95% bootstrap bound `<= 10%`) is applied to
  each sigma cell used for the advantage (`0.5` and `1.0` m) and to the pooled
  rate; all cells are reported.
- **Unit label.** `k` is a threshold on `rho_W = (E_W - N_W_hat) / N_W_hat`, a
  ratio to the wall-gate noise energy, not a number of noise standard
  deviations. The config key `noise_sigmas` keeps its historical name
  because `verification.classify_evidence` reads it.

Frozen values (`configs/pilot.yaml`, `phase8.frozen`):

| method | s_lo old -> new (binding sigma) | t_m old -> new (binding sigma) |
|---|---|---|
| exact_oracle | 0.016729 -> 0.016729 (0) | 25.537 -> 26.291 (0) |
| imperfect_narrow | 0.014672 -> 0.017580 (0) | 22.949 -> 27.709 (0) |
| imperfect_robust | 0.016168 -> 0.021784 (2.0) | 20.155 -> 25.572 (0.25) |
| mapfree_two_beam | - | 3.0849 -> 5.6104 (0) |
| mapfree_coarse_to_fine | - | 3.2792 -> 3.3299 (0) |

`k`: exact_oracle 1.8738, imperfect_narrow 1.8474, imperfect_robust 1.7217.
`s_hi`: 1.3178, 1.2873, 1.2868. `c = 1.0`.

- **Held-out:** no held-out geometry has been generated, solved or run.

## Amendment 3 — 2026-10-27

Written after the Amendment 2 freeze (`0350377`, tag `pilot-frozen`) and
before any held-out geometry was generated, solved or run. It adds secondary
analyses and fixes the stress design. **No threshold, gate, inflation factor,
primary rule, decision criterion or frozen config value changes.**

- **A3.1 Stratified reporting (prediction stated before held-out).** Every
  held-out rate, per method and sigma, is also reported separately for
  geometries with `delta_tau_W / psf < 1` and `>= 1`. `delta_tau_W` is the
  relay wall-path delay minus the target wall-path delay from the BRIEF §5.3
  image method on the drawn geometry, and `psf` is the trial's measured
  point-spread delay width. The stratification is evaluator-side only and
  never enters a decision.
  *Prediction from calibration:* false support concentrates in the `< 1`
  stratum. On calibration, the only geometry below 1 (cal_05, 0.884) held
  almost all replay support. The two geometries just above 1 (1.02, 1.07)
  held a little, and every geometry above 1.3 held none
  (`results/phase8a/amendment2/per_geometry_replay.csv`).
- **A3.2 Band trade-off (secondary).** Also report SUPPORTED with the upper
  bound `s_hi` removed. All other conditions stay as frozen (direct reliable,
  reference detectable, floor `k`, `S >= s_lo`).
- **A3.3 Stress design (BRIEF §16 robustness; frozen thresholds, no
  retuning).** Stresses are reported separately and never used for tuning.
  - *Grid per level:* the 24 held-out geometries; `sigma_map = 0.5 m` only;
    twin realizations 0 and 1; 8 records per world; methods
    `imperfect_robust`, `imperfect_narrow`, `mapfree_coarse_to_fine`, with the
    direct action shared. The seeds are the declared held-out streams, as in
    the main run. Baseline values for comparison come from the main held-out
    run.
  - *Levels:*
    - probe power `{-20, -10, 0, 20} dBm`;
    - bandwidth profiles `bw61` and `bw246`;
    - relay gain `{-10, +10, +20} dB` relative to the matched gain (replay
      world);
    - true wall material with relative permittivity and conductivity both
      scaled `x0.8` and `x1.2`; the twin keeps nominal material;
    - target paths illuminated via a reflection scaled in amplitude by
      `10^(+6/20)` and `10^(-6/20)`; the twin assumes constant RCS;
    - the Phase 4 physical blocker in the target world;
    - wall thickness `0.05` and `0.20 m` in BOTH truth and twin (labelled
      follow-up D8);
    - array calibration error `0.5 dB` and `5 deg` rms per element on the
      physical TX/RX arrays only, 4 realizations (labelled follow-up D11),
      with realized null depth toward the relay reported.
  - *Cost estimate (user's planning figure, not measured):* about 2,304
    trials per level, about 26 min per level, about 7.5 h in total.
  - Stress code is inactive by default and must leave the baseline path
    bit-identical. Before any stress run, this is shown on cal_00 x 2
    records: all non-timing columns identical to the baseline.
- **Held-out:** no held-out geometry has been generated, solved or run.

## Amendment 4 — 2026-10-28

Written while the held-out run was in progress (worktree at tag
`pilot-heldout`, started 2026-10-27 about 22:56 on GPU 1) and before any
held-out result file was opened, listed or summarized. It operationalizes the
BRIEF §16 label and fixes it in `scripts/phase8b_heldout_analysis.py`. No
threshold, gate, primary rule, criterion value or frozen config value changes.

- **A4.1 Decision labels.** The criteria are:
  - the §16 targets: target SUPPORT `>= 0.80` (primary, sigma `<= 0.5`),
    attack SUPPORT `<= 0.10` (primary, replay, pooled), NOT_OBSERVABLE
    `<= 0.20` (primary);
  - the rule-8 useful bar: target SUPPORT `>= 0.5` at sigma `<= 0.5`;
  - the rule-8 / A2.3 replay criterion: primary replay SUPPORTED `<= 5%`
    pooled and at sigma `0.5` and `1.0`, each with upper bound `<= 10%`;
  - the rule-8 advantage at sigma `0.5` or `1.0`.

  Each criterion gets a geometry-level 95% bootstrap interval (10,000 draws,
  seed 20261027, percentile interval; advantages are paired on the same
  draws). The labels are applied in this order:
  - **INVALID:** any validity check fails. The script checks complete
    decision columns, a full grid per geometry, the geometry count, the
    expected row count, the held-out split, decisions sourced from
    `phase8.frozen`, preflight PASS, H100, and a clean source commit. It also
    asserts that its own re-classification from `phase8.frozen` equals the
    run's stored decisions.
  - **NO-GO:** for at least one criterion, the whole 95% interval lies on the
    failing side. For example: replay lower bound `> 5%`, attack lower bound
    `> 0.10`, target upper bound `< 0.80`, NOT_OBSERVABLE lower bound `> 0.20`,
    or advantage upper bound `<= 0` at both sigma `0.5` and `1.0`.
  - **CONDITIONAL GO:** every point estimate meets its criterion (for the
    advantage, point `> 0` at sigma `0.5` or `1.0`) AND every interval
    criterion holds: replay upper bound `<= 10%` in every gating cell, and
    advantage lower bound `> 0` at sigma `0.5` or `1.0`. The non-statistical
    §16 conditions (Experiment A matched-information control, propagated
    evidence, no hidden-truth / budget / normalization / relay / leakage
    explanation) are reported separately from this label.
  - **INCONCLUSIVE:** otherwise.

  The label is reported for the primary contrast (the decision). The label
  that the A2.1 like-for-like contrast would give, with all other criteria
  identical, is reported separately as secondary; it is not the decision.
- **A4.2 Criteria table.** Every criterion's point estimate, interval,
  threshold and A4.1 tests go in one table (`criteria_primary.csv`,
  `criteria_like_for_like.csv`), so the label can be checked by hand. The
  advantage appears as its two sigma components plus the combined
  either-sigma criterion.
- **A4.3 Truth isolation test.** `tests/test_phase8a.py::
  test_frozen_decision_ignores_truth_and_labels` perturbs every
  truth/evaluator column (world labels swapped; true positions, wall offsets,
  path delays and AoAs, coverage flags, twin map errors, sigma, geometry and
  seed columns permuted and rescaled). It checks that decisions from
  `frozen_decision` + `classify_trials` are unchanged. Truth may enter only
  metrics and the A3.1 stratification.
- **Dry run:** on the calibration v5 trials, the script reproduces all 60
  method x sigma x world cells of the Amendment 2 SUPPORTED and
  SUPPORTED_common rates exactly (0 mismatches). Its label on calibration
  data is INCONCLUSIVE, with the advantage lower bound at or below 0 at both
  sigmas. That is calibration data, not a held-out result.

## Amendment 5 — 2026-10-28

Written after the held-out decision (INCONCLUSIVE, committed as
`docs/results/phase8b/`) and before any stress code was written or any stress
trial was generated, solved or run. It adds pre-registered stress predictions,
a labelled follow-up and a run order. **No threshold, gate, inflation factor,
primary rule, decision criterion, frozen config value or the held-out label
changes.** All stresses use `phase8.frozen` without retuning (A3.3).

- **A5.1 Pre-registered predictions for the bandwidth stress.** With `bw246`
  the point-spread width halves, so (a) the number of held-out geometries
  with `delta_tau_W / PSF(B) < 1` decreases, and (b) pooled replay SUPPORTED
  of the primary method decreases relative to baseline. With `bw61` both move
  the other way. Stratify by `delta_tau_W / PSF(B)` using each profile's own
  measured PSF.
- **A5.2 Labelled follow-up: relay standoff sweep.** On the 24 held-out
  geometries, place the relay at `R_A = R_T - Delta_R` with
  `Delta_R in {3, 6, 10, 15, 20, 30} m` (3 m is about the 20 ns latency
  floor), with the relay re-matched per geometry by the same evaluator rule;
  `sigma_map = 0.5`, the first 2 twin realizations, 8 records per world,
  methods `imperfect_robust`, `exact_oracle`, `mapfree_coarse_to_fine`;
  frozen thresholds. *Prediction:* replay SUPPORTED is high where the
  analytical `delta_tau_W / PSF < 1` and near zero where it is `>= 1`. Report
  replay SUPPORTED versus `delta_tau_W / PSF` with geometry-level intervals,
  and the analytical minimum resolvable standoff `Delta_R_min(B)` per
  geometry, solving `u(R_T - Delta_R) - u(R_T) = c * PSF(B)`.
- **A5.3 Run order** (stop and report after each block):
  1. bandwidth (`bw246`, `bw61`);
  2. standoff sweep (A5.2);
  3. array calibration error (D11);
  4. probe power;
  5. the remaining A3.3 levels (relay gain, material, RCS +/-6 dB, blocker,
     wall thickness D8).
- **Operational notes** (reproducibility only; no additional rule):
  - A bandwidth level switches the waveform and receiver processing profile
    (`phase6.processing.profile`, recorded in the `bandwidth_profile` column)
    to `bw61` or `bw246`. Everything else follows the A3.3 grid. The recorded
    noise calibration is re-measured with that profile, as in the main
    pipeline.
  - In `u(R) = sqrt(R^2 + 4 d^2) - R` (BRIEF §5.3 image method),
    `delta_tau_W = (u(R_A) - u(R_T)) / c`.
  - *Baseline* for A5.1(b) is the main held-out run's `sigma_map = 0.5` cell
    for the primary method (A3.3). With 2 twin realizations a stress record
    `r` uses twin `r mod 2`, while the main run used `r mod 4`. So a matched
    comparator, the main-run rows with `record_index in {0, 1, 4, 5}` (same
    seeds and same twin draw), is reported as secondary.
  - *Pooled* in A5.1(b) means all rows of a level for the primary method at
    `sigma_map = 0.5` (the only stress sigma), with the A4.1 geometry-level
    bootstrap (10,000 draws, seed 20261027) and paired level-minus-baseline
    differences on the same draws.
  - The A5.1(a) count uses each geometry's median `delta_tau_W / PSF(B)` over
    that level's rows, as in A3.1. A prediction is reported as *matched* when
    the point estimate moves in the stated direction (strictly), with the
    interval reported beside it. Stress results carry no GO/NO-GO label.

## Amendment 6 — 2026-10-28

Written after block 1 (bandwidth) was reported and versioned
(`docs/results/phase8c/`), and before any block 2 (standoff sweep) code was
written or any block 2 trial was generated, solved or run. **No threshold,
gate, primary rule, decision criterion, frozen config value, earlier label or
earlier prediction outcome changes.** Block 1 prediction A5.1(b) at `bw61`
stays NOT matched.

- **A6.1 Exploratory note on block 1 (exploratory, post hoc).** For all 24
  held-out geometries, compute the angular ratio `Delta_phi_W / HPBW_TX`.
  `Delta_phi_W = phi_W(R_A) - phi_W(R_T)` is the difference of the mixed-path
  departure angles. `HPBW_TX` is the measured -3 dB width of the primary
  method's wall beam at `phi_W(R_T)`. Report whether ho_10 and ho_19 are the
  geometries with `Delta_phi_W / HPBW_TX < 1` (or the smallest ratios). This
  is a hypothesis check, not a confirmatory result.
- **A6.2 Pre-registered two-ratio prediction for block 2 (standoff sweep).**
  - Mapped methods (`imperfect_robust`, `exact_oracle`): replay SUPPORTED is
    non-negligible only where BOTH `Delta_tau_W / PSF(B) < 1` AND
    `Delta_phi_W / HPBW_TX < 1`, and near zero where either ratio is `>= 1`.
  - `mapfree_coarse_to_fine`: replay SUPPORTED_common follows
    `Delta_tau_W / PSF(B) < 1` alone.
  - The discriminating level is `Delta_R` about 20 m in the nominal-like
    geometries, where the angle is resolved but the delay is not. The
    prediction there is low false support for the mapped methods and high for
    map-free.
  - Report the 2x2 table (delay resolved / not x angle resolved / not) with
    replay rates and geometry-level intervals per method, plus the
    per-geometry analytical standoffs `Delta_R_tau(B)` and `Delta_R_phi`. The
    predicted verification radius is `min(Delta_R_tau, Delta_R_phi)`.
- **Operational notes** (reproducibility only; no additional rule):
  - *Angles:* `phi_W(R) = atan(2 d / R)` (BRIEF §5.3 image method; it
    reproduces the Phase 5 reference 26.565 deg at `R = 60`, `2d = 30`). The
    true geometry is used, evaluator side only.
  - *HPBW_TX:* the primary wall beam is rebuilt through the unchanged
    planner (`phase7._plan_for`, `imperfect_robust`), target world,
    `sigma_map = 0.5`, records 0-7, twin `r mod 2` (the stress grid).
    `HPBW_TX` is the full -3 dB width of the lobe holding the pattern
    maximum (0.01 deg grid over [-60, 60] deg), taken as the median over the
    8 records (min and max reported). As a self-check, the rebuilt beam
    direction must equal the main run's `beam_phi_deg` on records 0, 1, 4, 5.
    The same per-geometry `HPBW_TX` is used for both mapped methods and for
    every `Delta_R`, because the wall beam is planned for the claimed
    candidate, not the relay (`scripts/phase8c_angular_ratio.py`).
  - *Standoffs (closed form):*
    - `Delta_R_tau(B)` solves `u(R_T - Delta_R) - u(R_T) = c * PSF(B)`.
    - `Delta_R_phi` solves `phi_W(R_T - Delta_R) - phi_W(R_T) = HPBW_TX`.
    - `PSF(B)` is each geometry's median measured point-spread delay at
      profile B (block 2 runs the baseline profile).
  - *Resolved:* a ratio `>= 1`. Block 2 ratios use `R_A = R_T - Delta_R` and
    that level's measured PSF.
  - *2x2 cells:* the units are (geometry, `Delta_R`) pairs. The interval is
    the A4.1 geometry-level bootstrap (10,000 draws, seed 20261027),
    resampling geometries with all of their rows in that cell.
  - *Descriptive only:* "non-negligible", "near zero", "low" and "high" get
    no numeric cut-off. The rates and intervals are reported, plus a
    per-`Delta_R` table, with `Delta_R = 20 m` shown separately.
  - *Relay feasibility:* if the re-matched relay at `R_T - Delta_R` is
    infeasible for a geometry (latency floor, causality or the 10 dBm cap),
    that (geometry, `Delta_R`) pair is reported as infeasible and not
    redrawn.

## Amendment 7 — 2026-10-28

Written after block 2 (standoff sweep) was reported and versioned
(`docs/results/phase8c/standoff/`), and before any block 3 (array calibration
error, D11) code was written or any block 3 trial was generated, solved or
run. **No threshold, gate, primary rule, decision criterion, frozen config
value, earlier label or earlier prediction outcome changes.** A6.1 stays not
supported.

- **A7.1 Predictions for D11** (0.5 dB and 5° rms per element on the physical
  arrays, 4 realizations, planner unaware):
  - realized null depth toward the relay, relative to the beam peak, has a
    median near −32 dB (analytical σ²/N with σ² from the stated errors), far
    shallower than the ideal-array baseline; report its distribution;
  - replay SUPPORTED of the primary method at σ = 0.5 changes by at most 0.05
    from baseline, and stays ≤ 5% in geometries with Δτ_W/PSF ≥ 1;
  - target SUPPORTED stays ≥ 0.95.

  Reasoning: the relay's extra LoS illumination replays at the direct-echo
  delay, about one point-spread width below the wall gate.
- **Operational notes** (reproducibility only; no additional rule):
  - *Physical arrays:* the gNB TX and gNB RX arrays (16 elements each). The
    relay is a single element and gets no error. Each element `n` gets a
    complex factor `e_n = 10^(a_n / 20) * exp(j * p_n)` with
    `a_n ~ N(0, 0.5)` dB and `p_n ~ N(0, 5)` deg, i.i.d.
  - *Where it acts:* only on the true per-path link matrices used to generate
    measurements: target and monostatic background `diag(e_RX) M diag(e_TX)`,
    relay downlink `M diag(e_TX)`, relay uplink `diag(e_RX) M`. The planner,
    twin maps, PSF measurement, receiver steering, noise calibration and the
    matched-relay calibration all keep the ideal arrays.
  - *Realizations:* 4 per geometry. Record `r` uses realization `r mod 4`, so
    each realization covers records `{k, k+4}` (twins 0 and 1), both worlds
    and every method of that record. Draws come from
    `numpy.random.default_rng([20261028, slot, k])` in the order TX
    amplitudes, TX phases, RX amplitudes, RX phases.
  - *Grid:* the A3.3 stress grid (σ = 0.5, twins 0 and 1, 8 records per world,
    methods `imperfect_narrow`, `imperfect_robust`,
    `mapfree_coarse_to_fine`), baseline bandwidth profile, frozen thresholds.
  - *Realized null depth:* for the primary method (`imperfect_robust`), both
    worlds, records 0–7, the wall beam `w` is rebuilt evaluator side through
    the unchanged pipeline on the perturbed links. As a self-check, the
    rebuilt `beam_phi_deg` must equal the run's. With `G(phi)` the
    `beam_gain` of `w ⊙ e_TX`, null depth is
    `10 log10(G(phi_relay) / max G)` over the 481-point grid on
    [−60, 60] deg used by `phase7._null_depth`, with `phi_relay` the true
    relay azimuth. *Ideal-array baseline:* the main held-out run's
    `null_depth_db` for the same method at σ = 0.5. Analytical reference:
    `10 log10(σ² / N)` with `σ² = (0.5 ln10 / 20)² + (5 π / 180)²` and
    `N = 16`.
  - *Baseline and strata* follow the A5 notes. Baseline is the main held-out
    run's σ = 0.5 cell, with matched records `{0, 1, 4, 5}` as the secondary
    comparator. Pooled paired differences use the A4.1 geometry-level
    bootstrap. The Δτ_W/PSF stratum uses each geometry's median at this level.
  - *Matched:* the second prediction is matched when the absolute pooled
    difference point estimate is ≤ 0.05 and the point estimate in the ≥ 1
    stratum is ≤ 0.05. The third is matched when the target point estimate is
    ≥ 0.95. The null-depth prediction is descriptive: median, interquartile
    range and range are reported beside the analytical value and the baseline
    median. "Near" and "far" get no numeric cut-off. Intervals are reported
    beside every point estimate.

## Amendment 8 — 2026-10-28

Written after block 3 (D11) was reported and versioned
(`docs/results/phase8c/array_error/`), and before any block 4 or block 5 code
was written or any block 4 or block 5 trial was generated, solved or run. The
exploratory s_lo sensitivity (`docs/results/phase8c/threshold_sensitivity/`)
uses only the main held-out run. **No threshold, gate, primary rule, decision
criterion, frozen config value, earlier label or earlier prediction outcome
changes.**

- **A8.1 Block 4, probe power {−20, −10, 0, 20} dBm, same design as A3.3.**
  Predictions:
  - (a) robust target SUPPORTED stays ≥ 0.95 at −10 dBm and drops at
    −20 dBm, where acquisition failures also appear (direct peak-to-noise
    ≈ 12 dB vs the 10 dB floor);
  - (b) coarse-to-fine target SUPPORTED_common falls below 0.5 at a power at
    least 10 dB higher than the power where robust target SUPPORTED falls
    below 0.5;
  - (c) replay SUPPORTED in the Δτ_W/PSF ≥ 1 stratum stays ≤ 5% at every
    power.
- **A8.2 Reduced block 5:** wall thickness 0.05 and 0.20 m (truth and twin,
  D8) and target RCS scaling ±6 dB on reflected-illumination paths (twin
  unaware). Predictions:
  - (a) thickness: replay in the Δτ_W/PSF ≥ 1 stratum stays ≤ 5% at both
    thicknesses; replay in the < 1 stratum is higher at 0.05 m than at
    baseline;
  - (b) RCS +6 dB: S rises about 4×, above s_hi, so robust target SUPPORTED
    falls below 0.2, while the A3.2 variant without s_hi stays ≥ 0.95;
    RCS −6 dB: target SUPPORTED stays ≥ 0.95; replay is unaffected by RCS.
- **A8.3 Not run, with reasons recorded.**
  - Relay gain levels: S is normalized by E_D, so a fixed relay gain change
    scales numerator and denominator together. This is an argument, not a
    result.
  - Material ±20%: bracketed by the thickness levels' reflection change.
  - Blocker: covered by test 15 and Phase 4.

  These A3.3 levels are marked **NOT RUN**.
- **Operational notes** (reproducibility only; no additional rule):
  - *Grid:* the A3.3 stress grid (σ = 0.5, twins 0 and 1, 8 records per
    world, methods `imperfect_narrow`, `imperfect_robust`,
    `mapfree_coarse_to_fine`), baseline bandwidth profile, frozen thresholds.
    Baseline is the main held-out run's σ = 0.5 cell (10 dBm, 0.10 m, constant
    RCS), with matched records `{0, 1, 4, 5}` as the secondary comparator.
    Intervals are the A4.1 geometry-level bootstrap. A stratum uses each
    geometry's median Δτ_W/PSF at that level (A5 notes).
  - *Probe power:* the level replaces the nominal 10 dBm for every probe
    action of every record (direct, mapped wall beam, map-free looks).
    Geometry acceptance and the matched-relay calibration stay at the nominal
    10 dBm. The relay keeps its matched gain and is linear (the 10 dBm cap
    applies only at calibration), so the forwarded signal scales with probe
    power. Receiver and relay noise are unchanged. Direct peak-to-noise is
    not a recorded column; the recorded direct-gate ratio
    `(E_D − N_D) / N_D` is reported descriptively instead.
  - *Thickness:* `wall.thickness_m` is set to the level in the configuration
    used for the true scene and for the twin (Sionna slab material).
    Geometry acceptance runs with that configuration. As a validity check,
    the accepted geometries must equal the main run's.
  - *RCS:* the true target link paths whose interaction sequence has a
    specular reflection before the sensing interaction are scaled in
    amplitude by `10^(±6/20)`. These are `outgoing_wall_*` and the
    multi-bounce paths with a reflection before the target. The direct and
    reciprocal-wall paths (LoS illumination) are unchanged, and the twin
    keeps constant RCS. The replay world contains no target paths. So, as a
    harness validity check (not evidence for the prediction), the replay rows
    on matched records must equal the main run's in every non-timing column.
  - *Matched rules (point estimates; intervals reported beside them):*
    - A8.1(a): robust target ≥ 0.95 at −10 dBm, strictly lower at −20 dBm
      than at −10 dBm, and at least one ACQUISITION_FAILED row at −20 dBm.
    - A8.1(b): `P_m` is the highest tested power, including the 10 dBm
      baseline, at which method `m`'s target rate is < 0.5. The prediction is
      matched when `P_c2f ≥ P_robust + 10`. If robust is never below 0.5,
      `P_robust < −20` and it is matched when `P_c2f ≥ −10`. It is not
      matched when c2f is never below 0.5, and indeterminate (reported as not
      matched) when only `P_c2f = −20` exists.
    - A8.1(c) and A8.2(a) first part: robust replay ≤ 0.05 in the ≥ 1 stratum
      at each level.
    - A8.2(a) second part: robust replay in the < 1 stratum at 0.05 m is
      strictly greater than the baseline rate over the same geometries.
    - A8.2(b): at +6 dB, robust target < 0.2 and the no-s_hi variant ≥ 0.95.
      At −6 dB, robust target ≥ 0.95. "About 4×" is descriptive: the median
      ratio of target S to the baseline S on matched records, plus the
      fraction of target rows with S > s_hi.

## Amendment 9 — 2026-10-28

Written after the block 4 level −20 dBm failed with an uncaught `SceneError` in
12 of 24 geometries (partial output kept only as evidence in
`FAILED_probe_power_m20dbm`, never merged), and before the change below was
written or −20 dBm was re-run. **No threshold, gate, primary rule, decision
criterion, frozen config value, earlier label or earlier prediction outcome
changes.** A8.1 predictions are unchanged.

- **A9.1 Behaviour change:** when the measured candidate lies outside the
  twin's free space (the scene validator would raise SceneError), the planner
  returns decision UNRESOLVED with path status NOT_OBSERVABLE and reason
  "candidate_outside_twin_free_space", instead of crashing. A candidate WAS
  detected, so ACQUISITION_FAILED is not the right label. The change only
  affects records that previously raised an uncaught exception; no previously
  produced row can change.
- **A9.2 Verification:**
  - (a) the cal_00 × 2 equivalence check;
  - (b) re-run one already completed level (−10 dBm) on 2 geometries with
    the new code and show bit-identical non-timing columns.

  No need to re-run the other completed levels.
- **A9.3** Re-run only the −20 dBm level. Predictions A8.1 are unchanged.
  Additionally report, evaluator side, per world:
  - how many acquisitions are false (candidate farther than 2 m from the
    true position);
  - how many records take the new outside-free-space path.
- **Operational notes** (reproducibility only; no additional rule):
  - *Where it acts:* two places validate the twin scene against the measured
    candidate.
    - The direct-action PSF twin (nominal walls) in `phase7._acquire_record`.
      There the record has no PSF and no direct gate, so every method of that
      record gets the A9.1 row (status UNRESOLVED, `path_availability`
      NOT_OBSERVABLE, the A9.1 reason, energies NaN). This includes the
      map-free methods and direct_only, which need the direct gate.
    - A mapped method's wall-plan twin (twin walls) in `phase7._plan_for`.
      There only that method's row gets the A9.1 row.
  - *Frozen classification:* a row with status UNRESOLVED and a NaN score is
    never SUPPORTED or SUPPORTED_common under `phase8.frozen`.
  - *False acquisition:* horizontal distance between the measured candidate
    (`candidate_range_m`, `candidate_aoa_deg` from the gNB) and the true
    claimed candidate position (`candidate_distance_m`,
    `true_candidate_azimuth_deg`) > 2 m, counted per (geometry, world,
    record) over detected records.
  - *Counts per world* are (geometry, world, record) units. A record takes
    the outside-free-space path if any of its rows has the A9.1 reason.
