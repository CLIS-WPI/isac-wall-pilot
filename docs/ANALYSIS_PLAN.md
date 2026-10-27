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
- **Calibration re-run:** the first calibration run (harness `4e44d3b`) is
  invalid. It is kept in `results/phase8a_invalid_v1/` and is not merged,
  reused or consulted for any threshold. Calibration is re-run from scratch
  with the same declared seeds and the same 12 calibration geometries.
- **Held-out:** no held-out geometry has been generated, solved or run.
