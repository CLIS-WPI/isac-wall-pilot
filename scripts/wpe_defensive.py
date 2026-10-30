"""ANALYSIS_PLAN Amendment 13 (defensive): channel coupling bound (B1), consistency statistic (B2), certified
region (B3). No attacker signal, relay output or evasion filter is constructed; every quantity is a property of
the EXISTING verifier and the frozen channel data.

The receive combiner after A12.1 is C(h; phi) = sum_n conj(a_n(phi)) w_hann_n h_n (receiver.angular_delay_map).
B1 uses it on the relay's uplink received vectors (prepare_geometry links; the relay is single-element).

Modes:
  couple RUN SLOT REF OUT      B1: G, kappa, delay offset, overlap, E_req per geometry (GPU; scene solve + passive rows)
                                   RUN = heldout | standoff_03m ... standoff_30m
  consistency RUN SLOT REF OUT B2: residual statistic C on GENUINE-target records (GPU)
                                   RUN = calibration | heldout | standoff_XXm
  calibrate CONS_DIR OUT       B2: c_hi = 0.99 quantile of C over calibration genuine targets (CPU; commit before eval)
  report COUPLE_DIR CONS_DIR OUT GEOMETRIES.csv   B1/B2/B3 tables + certified region (CPU)
  coverage SLOT OUT            Amdt 15: frozen B3 certificate on held-out SLOT over bandwidth x sigma_map grid (GPU)
  coverage_report COV_DIR OUT PAPER_OUT   Amdt 15: certified fraction per cell + overlap margins + figure (CPU)
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, array_response, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _acquire_record, _plan_for, _twin_map, _twin_offsets, _twin_paths
from isac_verify.phase8a import (HELDOUT, accepted_geometry, approve_heldout, geometry_config,
                                 geometry_seed_streams, prepare_geometry, twin_draws)
from isac_verify.receiver import (ProcessingChoices, _aoa_window, angular_delay_map, cfr_from_paths,
                                  delay_doppler_map, gate_energy, noise_calibration)
from isac_verify.stress import stress_config
from isac_verify.verification import GateWindow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wpc_passive import energy as gate_e  # noqa: E402
from wpc_passive import passive_gate  # noqa: E402

SIGMA, METHOD = 0.5, "imperfect_robust"
STANDOFFS = [f"standoff_{m:02d}m" for m in (3, 6, 10, 15, 20, 30)]


def combiner_power(h: torch.Tensor, phi_deg: float, array: ArrayGeometry, w: torch.Tensor) -> float:
    """Power the A12.1 receive combiner (Hann taper, steered to phi) produces for received vector h."""
    v = array_response(phi_deg, array, dtype=torch.complex64).conj()  # = steering_vector(phi)
    return float((v * w.to(torch.complex64) * h).sum().abs().square().item())


def _route_excess_s(config: PilotConfig, candidate_range_m: float, d_pos: float, d_neg: float,
                    sigma_map_m: float) -> float:
    """Largest second-order uplink path excess (hypot(R_A, o) - R_A)/c over every relay position on the bearing
    (R_A in [5 m, R_T - latency-floor standoff]) and every route, with wall offsets inflated by the 3-sigma twin
    error. Offsets: single bounce 2|d|, double bounce 2(|d_pos|+|d_neg|). Excess grows as R_A shrinks and o grows."""
    c = float(config.speed_of_light)
    standoff_m = c * float(config.get("geometry_reference", "latency_floor_ns")) * 1e-9 / 2.0
    r = np.linspace(5.0, max(candidate_range_m - standoff_m, 5.0), 64)
    infl = 3.0 * float(sigma_map_m)  # twin-error interval on each wall offset
    dp, dn = abs(d_pos) + infl, abs(d_neg) + infl
    offsets = (2.0 * dp, 2.0 * dn, 2.0 * (dp + dn))
    return max(float(((np.hypot(r, o) - r) / c).max()) for o in offsets)


def twin_window_lo_s(config: PilotConfig, refl_delay_min_s: float, candidate_range_m: float,
                     candidate_delay_sigma_s: float, d_pos: float, d_neg: float, psf_width_s: float,
                     sigma_map_m: float) -> float:
    """Design revision (item 3): shift the WHOLE reflector-gate interval left by the largest route path excess over
    all relay positions on the bearing, then widen by the point-spread width and the candidate range (delay) 3-sigma
    uncertainty; the twin-error interval is folded into the excess via inflated wall offsets. Twin walls only; no
    truth. Lower edge = refl.delay_min - max_excess - psf - 3*candidate_delay_sigma."""
    max_excess = _route_excess_s(config, candidate_range_m, d_pos, d_neg, sigma_map_m)
    return refl_delay_min_s - max_excess - psf_width_s - 3.0 * float(candidate_delay_sigma_s)


# ANALYSIS_PLAN Amendment 14 (pre-registered): the fresh confirmation seed base for the 24 NEW geometries. Frozen
# before any confirmation geometry was generated. Per-slot seed = CONFIRM_SEED_BASE + slot; same ranges/acceptance
# as phase8a.draw_geometry/accepted_geometry (candidate, relay, half-width drawn in that order).
CONFIRM_SEED_BASE = 814500


def _accepted_confirmation(config: PilotConfig, slot: int):
    geom = config.section("phase8", "geometry")
    for attempt in range(1001):
        rng = np.random.default_rng([CONFIRM_SEED_BASE + int(slot), int(attempt)])
        geometry = {"slot": int(slot), "attempt": int(attempt), "stream_seed": CONFIRM_SEED_BASE + int(slot),
                    "candidate_distance_m": float(rng.uniform(*geom["candidate_distance_m_range"])),
                    "relay_distance_m": float(rng.uniform(*geom["relay_distance_m_range"])),
                    "half_width_m": float(rng.uniform(*geom["true_corridor_half_width_m_range"])),
                    "wall_thickness_m": float(geom["wall_thickness_m"])}
        config_g = geometry_config(config, geometry)
        prepared = prepare_geometry(config_g)
        if prepared["rejection"] is None:
            return geometry, config_g, prepared
    raise RuntimeError(f"confirmation slot {slot}: no feasible geometry in 1000 draws")


def _with_profile(config: PilotConfig, name: str) -> PilotConfig:
    """Isolated bandwidth change: only the active OFDM profile (read by ProcessingChoices / the acquisition), so a
    coverage cell differs from frozen Amendment 14 in bandwidth alone, not in the A3.3 stress grid that
    stress_config would also apply. baseline_bandwidth_profile is reporting-only but kept consistent."""
    data = copy.deepcopy(config.data)
    data["phase6"]["processing"]["profile"] = name
    data["rf"]["baseline_bandwidth_profile"] = name
    return PilotConfig(path=config.path, data=data)


def _setup(run: str, slot: int, profile: str | None = None):
    from isac_verify.phase8a import SPLIT
    base_config = PilotConfig.load("configs/pilot.yaml")
    confirm = run.startswith("conf_")
    level = run[len("conf_"):] if confirm else run
    split = SPLIT if level == "calibration" else HELDOUT
    if split == HELDOUT:
        approve_heldout()
    config = stress_config(base_config, level) if level.startswith("standoff_") else base_config
    if profile is not None:
        config = _with_profile(config, profile)
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(choices=choices, config=config,
                                   records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
    direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
                                 ArrayGeometry.from_config(config)).weights
    if confirm:
        geometry, config_g, prepared = _accepted_confirmation(config, slot)
    else:
        geometry, config_g, prepared, _ = accepted_geometry(config, slot, split)
    standoff = config.section("phase8").get("stress_runtime", {}).get("relay_standoff_m")
    if standoff is not None:
        geometry = {**geometry, "relay_distance_m": geometry["candidate_distance_m"] - float(standoff)}
        config_g = geometry_config(config, geometry)
        prepared = prepare_geometry(config_g)
        assert prepared["rejection"] is None
    return base_config, config, config_g, choices, noise_full, direct_weights, geometry, prepared, split


def couple(run: str, slot: int, ref_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gid = f"{'cf' if run.startswith('conf_') else 'ho'}_{slot:02d}"
    _, config, config_g, choices, noise_full, direct_weights, geometry, prepared, split = _setup(run, slot)
    array, rx_array = ArrayGeometry.from_config(config_g), choices.receiver_array
    w = _aoa_window(choices)
    seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, 1)}
    twins = int(config.section("phase8", "grid")["twin_realizations_per_sigma"])
    z = twin_draws(config, HELDOUT, slot, twins)[0]
    base = _acquire_record(config=config_g, choices=choices, world="target", record_index=0,
                           direct_weights=direct_weights, links=prepared["links"], calibration=prepared["calibration"],
                           noise_full=noise_full, timing={}, seeds=seeds)
    row = _cert_row(run, gid, config, config_g, choices, array, rx_array, w, z, base, direct_weights, prepared,
                    geometry, SIGMA)
    pd.DataFrame([row]).to_parquet(out_dir / f"{run}_{slot:02d}.parquet", index=False)
    print(f"{run} {gid}: kappa={row.get('kappa_db', float('nan')):.2f} kappa_lb={row.get('kappa_lb_db', float('nan')):.2f} dB "
          f"routes={row.get('n_gate_routes')} win_lo={row.get('window_lo_ns', float('nan')):.1f}ns "
          f"ghosts_ok={row.get('cons_ghost_in_survey_all')}&{row.get('cons_ghost_out_direct_all')}", flush=True)


def _cert_row(run, gid, config, config_g, choices, array, rx_array, w, z, base, direct_weights, prepared,
              geometry, sigma):
    """Revised certificate row for one geometry at one map-uncertainty sigma; bandwidth is carried by config/choices.
    Factored out of couple so the coverage sweep (Amendment 15) can recompute it per (bandwidth, sigma) cell."""
    offsets = _twin_offsets(config_g, sigma, z["positive"], z["negative"])
    plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world="target", method=METHOD,
                           sigma_m=sigma, twin_offsets=offsets, direct_weights=direct_weights, timing={})
    refl = passive_gate(config_g, plan, base["candidate"], offsets, sigma)
    dgate, side = plan.direct_gate, plan.wall_side
    phi_w, cand = float(plan.predicted_aod_deg), float(base["candidate"].azimuth_deg)
    uplink = {p.family: p for p in prepared["links"]["uplink"]}
    los, wall = uplink.get("los"), uplink.get(f"wall_clutter_{side}_wall")
    row = {"label": "B13.1 pre-registered", "run": run, "geometry_id": gid, "wall_side": side,
           "twin_reciprocal_aoa_deg": phi_w, "candidate_bearing_deg": cand,
           "relay_distance_m": geometry["relay_distance_m"], "candidate_distance_m": geometry["candidate_distance_m"],
           "half_width_m": geometry["half_width_m"],
           "refl_delay_min_ns": refl.delay_min_s * 1e9, "refl_delay_max_ns": refl.delay_max_s * 1e9,
           "direct_delay_min_ns": dgate.delay_min_s * 1e9, "direct_delay_max_ns": dgate.delay_max_s * 1e9}
    if los is not None and wall is not None:
        h_los, h_wall = los.matrix[:, 0].to(torch.complex64), wall.matrix[:, 0].to(torch.complex64)
        # G: angle selectivity of the reflector gate (unit plane waves), wall arrival vs the steered peak.
        g = combiner_power(array_response(wall.aoa_deg, rx_array, dtype=torch.complex64), phi_w, rx_array, w) / \
            combiner_power(array_response(phi_w, rx_array, dtype=torch.complex64), phi_w, rx_array, w)
        # kappa: LoS copy into the candidate-bearing gate vs wall copy into the reflector gate (physical, |c|^2 built in).
        e_cand = combiner_power(h_los, cand, rx_array, w)
        e_refl = combiner_power(h_wall, phi_w, rx_array, w)
        offset_ns = (los.delay_s - wall.delay_s) * 1e9
        los_at_refl_center_ns = refl.center_delay_s * 1e9 + offset_ns
        row |= {"G_db": 10 * math.log10(max(g, 1e-30)), "kappa_db": 10 * math.log10(max(e_cand / e_refl, 1e-30)),
                "kappa": e_cand / e_refl, "los_uplink_aoa_deg": los.aoa_deg, "wall_uplink_aoa_deg": wall.aoa_deg,
                "los_uplink_delay_ns": los.delay_s * 1e9, "wall_uplink_delay_ns": wall.delay_s * 1e9,
                "delay_offset_los_minus_wall_ns": offset_ns, "los_copy_at_refl_center_ns": los_at_refl_center_ns,
                "overlap": bool(dgate.delay_min_s * 1e9 <= los_at_refl_center_ns <= dgate.delay_max_s * 1e9)}
        # Revised certificate (external-review addendum, SCOPE: second-order routes; for ANY signal emitted from the
        # EVALUATED relay position, ray-traced channels; NOT yet a verifier-side certificate over unknown relay
        # positions). Routes = every non-LoS uplink multipath (reciprocal wall, opposite wall, double bounces).
        routes = [p for p in prepared["links"]["uplink"] if p.family != "los"]
        # Upper bound on reflector-gate energy per unit emitted energy: (sum_i sqrt(g_i^refl))^2 (Cauchy-Schwarz).
        sqrt_refl = sum(math.sqrt(max(combiner_power(p.matrix[:, 0].to(torch.complex64), phi_w, rx_array, w), 0.0))
                        for p in routes)
        # Item 2: lower bound on forced bearing energy per unit emitted energy by reverse triangle inequality -
        # other routes leaking into the bearing can partly cancel the LoS ghost: (sqrt(g_L) - sum_i sqrt(g_i^bear))^2.
        sqrt_bear_others = sum(math.sqrt(max(combiner_power(p.matrix[:, 0].to(torch.complex64), cand, rx_array, w),
                                             0.0)) for p in routes)
        bear_lb_amp = max(math.sqrt(max(e_cand, 0.0)) - sqrt_bear_others, 0.0)
        kappa_lb = (bear_lb_amp ** 2) / sqrt_refl ** 2 if sqrt_refl > 0 else float("nan")
        kappa_cons = e_cand / sqrt_refl ** 2 if sqrt_refl > 0 else float("nan")  # old upper-numerator bound (kept)
        # Item 3: shift the WHOLE reflector-gate interval left by each route's path difference (tau_i - tau_LoS) and
        # check both edges against the twin-derived window and the excluded direct/reflector-gate regions.
        survey_hi_ns = float(config.get("phase6", "processing", "delay_search_ns")[1])
        window_lo_ns = twin_window_lo_s(config_g, refl.delay_min_s, float(base["candidate"].range_m),
                                        float(base["candidate"].delay_sigma_s), offsets["positive"],
                                        offsets["negative"], base["point_spread"].delay_width_s, sigma) * 1e9
        ghosts, in_window_all, out_excluded_all = [], True, True
        for p in routes:
            dlt = (p.delay_s - los.delay_s) * 1e9  # path difference vs LoS (ns); ghost interval = gate - dlt
            g_lo, g_hi = refl.delay_min_s * 1e9 - dlt, refl.delay_max_s * 1e9 - dlt
            in_win = (window_lo_ns <= g_lo) and (g_hi <= survey_hi_ns)
            ov_direct = not (g_hi < dgate.delay_min_s * 1e9 or g_lo > dgate.delay_max_s * 1e9)
            ov_refl = not (g_hi < refl.delay_min_s * 1e9 or g_lo > refl.delay_max_s * 1e9)
            in_window_all &= in_win
            out_excluded_all &= not (ov_direct or ov_refl)
            ghosts.append(f"{p.family}:{g_lo:.1f}-{g_hi:.1f}:{'W' if in_win else 'w'}"
                          f"{'X' if (ov_direct or ov_refl) else 'x'}")
        row |= {"kappa_cons": kappa_cons, "kappa_cons_db": 10 * math.log10(max(kappa_cons, 1e-30)),
                "kappa_lb": kappa_lb, "kappa_lb_db": 10 * math.log10(max(kappa_lb, 1e-30)),
                "n_gate_routes": len(routes), "cons_ghost_in_survey_all": bool(in_window_all),
                "cons_ghost_out_direct_all": bool(out_excluded_all), "gate_route_ghosts": ";".join(ghosts),
                "window_lo_ns": window_lo_ns,
                "scope": "2nd-order routes; evaluated relay position; ray-traced; not verifier-side over unknown positions"}
        twin_direct = _twin_map(config=config_g, choices=choices, weights=direct_weights,
                                twin_paths=_twin_paths(config_g, base["candidate"], offsets, {}), timing={})
        # Item 2 subtraction term: twin-predicted bearing residual the statistic removes (record 0, new window).
        aoa_half = 0.5 * (dgate.aoa_max_deg - dgate.aoa_min_deg)
        cand_gate = _cand_residual_gate(cand, dgate, refl, (window_lo_ns * 1e-9, survey_hi_ns * 1e-9), aoa_half)
        row["twin_bearing_resid_w"] = _residual_energy(twin_direct, cand_gate, dgate, refl)
        # Item 1: E_min = smallest reflector-gate energy that passes BOTH s_lo and the k floor of the frozen
        # receive-side check, given this record's measured E_D/noise. Computed in-line so it also covers the fresh
        # confirmation geometries (which have no stored C1 passive rows). r_pass = twin wall/direct energy ratio.
        th = json.loads(Path("docs/results/wpc_a12/c1_passive_thresholds.json").read_text())
        s_lo, floor_k = float(th["s_lo"]["value"]), float(th["floor_k"]["value"])
        eps = float(config.get("phase6", "evidence", "epsilon_w"))
        direct = base["direct_energy"]
        meas_pass, twin_pass, twin_dir = (gate_e(base["direct_map"], refl), gate_e(twin_direct, refl),
                                          gate_e(twin_direct, dgate))
        r_pass = twin_pass.energy_w / twin_dir.energy_w if twin_dir.energy_w > 0 else float("nan")
        n_pass = meas_pass.noise_w
        e_min = (max(s_lo * r_pass * max(direct.net_energy_w, eps), floor_k * n_pass) + n_pass
                 if math.isfinite(r_pass) else float("nan"))
        row |= {"E_min": e_min, "r_pass": r_pass, "N_pass": n_pass, "E_D_net": direct.net_energy_w}
    else:
        row |= {"G_db": float("nan"), "kappa_db": float("nan"), "kappa": float("nan"),
                "kappa_cons": float("nan"), "kappa_cons_db": float("nan"), "kappa_lb": float("nan"),
                "kappa_lb_db": float("nan"), "n_gate_routes": 0, "cons_ghost_in_survey_all": False,
                "cons_ghost_out_direct_all": False, "gate_route_ghosts": "", "window_lo_ns": float("nan"),
                "twin_bearing_resid_w": float("nan"), "E_min": float("nan"), "r_pass": float("nan"),
                "N_pass": float("nan"), "E_D_net": float("nan"), "scope": "missing uplink",
                "missing_uplink": f"los={los is not None} wall_{side}={wall is not None}"}
    return row


def leakage(run: str, slot: int, ref_path: Path, out_dir: Path) -> None:
    """Prop. 3 assumption check: share of a single emitter's reflector-gate energy arriving through paths OTHER
    than the reciprocal wall path, from the existing noiseless uplink channels (relay -> gNB, single element)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gid = f"ho_{slot:02d}"
    _, config, config_g, choices, noise_full, direct_weights, geometry, prepared, split = _setup(run, slot)
    profile = choices.profile
    seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, 1)}
    twins = int(config.section("phase8", "grid")["twin_realizations_per_sigma"])
    z = twin_draws(config, HELDOUT, slot, twins)[0]
    base = _acquire_record(config=config_g, choices=choices, world="target", record_index=0,
                           direct_weights=direct_weights, links=prepared["links"], calibration=prepared["calibration"],
                           noise_full=noise_full, timing={}, seeds=seeds)
    offsets = _twin_offsets(config_g, SIGMA, z["positive"], z["negative"])
    plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world="target", method=METHOD,
                           sigma_m=SIGMA, twin_offsets=offsets, direct_weights=direct_weights, timing={})
    refl = passive_gate(config_g, plan, base["candidate"], offsets, SIGMA)
    side = plan.wall_side
    wall_family = f"wall_clutter_{side}_wall"
    uplink = list(prepared["links"]["uplink"])
    ones1 = torch.ones(1, device="cuda", dtype=torch.complex64)
    pilots = torch.ones((profile.num_symbols, profile.fft_size), device="cuda", dtype=torch.complex64)

    def gate_energy_of(paths) -> float:  # noiseless reflector-gate energy of the uplink channel built from `paths`
        if not paths:
            return 0.0
        cfr = cfr_from_paths(paths, ones1, profile)  # [rx, freq]
        h_eff = cfr[None, :, :] * pilots[:, None, :]  # constant across symbols -> all energy at zero Doppler
        adm = angular_delay_map(delay_doppler_map(h_eff, choices), choices, noise_per_cell_w=0.0)
        return float(gate_e(adm, refl).energy_w)

    wall = [p for p in uplink if p.family == wall_family]
    others = [p for p in uplink if p.family != wall_family]
    e_all, e_wall, e_others = gate_energy_of(uplink), gate_energy_of(wall), gate_energy_of(others)
    row = {"label": "Prop3 assumption check (defensive)", "run": run, "geometry_id": gid, "wall_side": side,
           "n_uplink_paths": len(uplink), "n_wall_paths": len(wall), "n_other_paths": len(others),
           "e_gate_all_w": e_all, "e_gate_wall_only_w": e_wall, "e_gate_others_only_w": e_others,
           "share_others": (e_others / e_all) if e_all > 0 else float("nan"),
           "other_families": ",".join(sorted({p.family for p in others}))}
    pd.DataFrame([row]).to_parquet(out_dir / f"{run}_{slot:02d}.parquet", index=False)
    print(f"{run} {gid}: share_others={row['share_others']:.4g} "
          f"(others={len(others)} paths: {row['other_families']})", flush=True)


def _cand_residual_gate(cand_deg: float, dgate: GateWindow, refl: GateWindow, survey: tuple[float, float],
                        aoa_half: float) -> GateWindow:
    """Candidate-bearing angle window over the surveillance delay window (energy outside direct/refl is masked later)."""
    return GateWindow(delay_min_s=survey[0], delay_max_s=survey[1], aoa_min_deg=cand_deg - aoa_half,
                      aoa_max_deg=cand_deg + aoa_half, center_delay_s=0.5 * (survey[0] + survey[1]),
                      center_aoa_deg=cand_deg)


def _residual_energy(m, cand_gate: GateWindow, dgate: GateWindow, refl: GateWindow) -> float:
    """Energy in the candidate-bearing angle window, surveillance delays, OUTSIDE the direct and reflector delay ranges."""
    ang = (m.aoa_deg >= cand_gate.aoa_min_deg - 1e-12) & (m.aoa_deg <= cand_gate.aoa_max_deg + 1e-12)
    dly = (m.delay_s >= cand_gate.delay_min_s - 1e-15) & (m.delay_s <= cand_gate.delay_max_s + 1e-15)
    for g in (dgate, refl):
        dly &= ~((m.delay_s >= g.delay_min_s - 1e-15) & (m.delay_s <= g.delay_max_s + 1e-15))
    return float(m.power_w[dly][:, ang].sum().item())


def consistency(run: str, slot: int, ref_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config, config, config_g, choices, noise_full, direct_weights, geometry, prepared, split = _setup(run, slot)
    gid = f"{'cf' if run.startswith('conf_') else ('ho' if split == HELDOUT else 'cal')}_{slot:02d}"
    grid = config.section("phase8", "grid")
    records = int(grid["records_per_world"])
    sigmas = [float(v) for v in grid["sigma_map_m"]] if run == "calibration" else [SIGMA]
    survey = tuple(float(v) * 1e-9 for v in choices.delay_search_ns)
    seeds = {"streams": geometry_seed_streams(config, split, slot, records)}
    twins = int(grid["twin_realizations_per_sigma"])
    twin_z = twin_draws(config, split, slot, twins)
    rows = []
    for record in range(records):  # GENUINE targets only
        base = _acquire_record(config=config_g, choices=choices, world="target", record_index=record,
                               direct_weights=direct_weights, links=prepared["links"],
                               calibration=prepared["calibration"], noise_full=noise_full, timing={}, seeds=seeds)
        if base.get("point_spread") is None:
            for sigma in sigmas:
                rows.append({"run": run, "geometry_id": gid, "world": "target", "record_index": record,
                             "sigma_map_m": sigma, "status": "no_direct"})
            continue
        for sigma in sigmas:
            offsets = _twin_offsets(config_g, sigma, twin_z[record % twins]["positive"],
                                    twin_z[record % twins]["negative"])
            try:
                plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world="target", method=METHOD,
                                       sigma_m=sigma, twin_offsets=offsets, direct_weights=direct_weights, timing={})
            except Exception:
                rows.append({"run": run, "geometry_id": gid, "world": "target", "record_index": record,
                             "sigma_map_m": sigma, "status": "no_plan"})
                continue
            if plan.wall_gate is None:
                rows.append({"run": run, "geometry_id": gid, "world": "target", "record_index": record,
                             "sigma_map_m": sigma, "status": "no_wall"})
                continue
            refl = passive_gate(config_g, plan, base["candidate"], offsets, sigma)
            aoa_half = 0.5 * (plan.direct_gate.aoa_max_deg - plan.direct_gate.aoa_min_deg)
            # Twin-derived lower edge (design revision, item 3: whole-gate-span shift + widening); upper edge fixed.
            window_lo_s = twin_window_lo_s(config_g, refl.delay_min_s, float(base["candidate"].range_m),
                                           float(base["candidate"].delay_sigma_s), offsets["positive"],
                                           offsets["negative"], base["point_spread"].delay_width_s, sigma)
            survey_rec = (window_lo_s, survey[1])
            cand_gate = _cand_residual_gate(float(base["candidate"].azimuth_deg), plan.direct_gate, refl, survey_rec,
                                            aoa_half)
            twin_paths = _twin_paths(config_g, base["candidate"], offsets, {})
            twin_direct = _twin_map(config=config_g, choices=choices, weights=direct_weights, twin_paths=twin_paths,
                                    timing={})
            e_meas = _residual_energy(base["direct_map"], cand_gate, plan.direct_gate, refl)
            e_twin = _residual_energy(twin_direct, cand_gate, plan.direct_gate, refl)
            rows.append({"run": run, "geometry_id": gid, "world": "target", "record_index": record,
                         "sigma_map_m": sigma, "status": "computed", "C": e_meas - e_twin,
                         "E_resid_meas": e_meas, "E_resid_twin": e_twin, "window_lo_ns": window_lo_s * 1e9,
                         "candidate_bearing_deg": float(base["candidate"].azimuth_deg)})
    pd.DataFrame(rows).to_parquet(out_dir / f"{run}_{slot:02d}.parquet", index=False)
    print(f"{run} {gid}: rows={len(rows)} computed={sum(r.get('status') == 'computed' for r in rows)}", flush=True)


def _load(d: Path, pattern: str) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(d.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def calibrate(cons_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cal = _load(cons_dir, "calibration_*.parquet")
    c = cal[cal["status"] == "computed"]["C"].to_numpy(float)
    c_hi = float(np.quantile(c, 0.99))
    (out / "b2_threshold.json").write_text(json.dumps({
        "label": "B14.5 (revised window) pre-registered: 1% genuine false rejection over the 12 calibration geometries, "
                 "pooled sigma; 0.99 quantile of C under the twin-derived whole-gate-span window. Committed before eval.",
        "c_hi": c_hi, "n_calibration_target_records": int(len(c)),
        "realized_false_rejection": float((c > c_hi).mean())}, indent=2))
    print(json.dumps({"c_hi": c_hi, "n": len(c), "realized_fr": float((c > c_hi).mean())}, indent=1))


def report(couple_dir: Path, cons_dir: Path, out: Path, geometries_path: Path) -> None:
    from phase8b_heldout_analysis import Bootstrap
    from wpb_reanalysis import cp_upper_zero

    config = PilotConfig.load("configs/pilot.yaml")
    c_hi = json.loads((out / "b2_threshold.json").read_text())["c_hi"]
    geom_ids = sorted(pd.read_csv(geometries_path)["geometry_id"]) if geometries_path.exists() else None

    # B2 false rejection on genuine targets (revised window). Runs present in cons_dir decide which levels to report.
    cons_runs = sorted({p.name.rsplit("_", 1)[0] for p in cons_dir.glob("*_*.parquet")} - {"calibration"})
    order = [r for r in ["heldout", *STANDOFFS] if r in cons_runs] + [r for r in cons_runs
                                                                      if r not in ["heldout", *STANDOFFS]]
    b2 = []
    for run in order:
        d = _load(cons_dir, f"{run}_*.parquet")
        d = d[d["status"] == "computed"].assign(reject=lambda f: f["C"] > c_hi)
        gids = sorted(d["geometry_id"].unique())
        boot = Bootstrap(geom_ids if (geom_ids and set(gids) <= set(geom_ids)) else gids,
                         int(config.get("phase8", "heldout", "bootstrap_draws")))
        point, samples = boot.rate(d, "reject")
        b2.append({"run": run, "false_rejection": point, "ci95_lo": float(np.nanquantile(samples, 0.025)),
                   "ci95_hi": float(np.nanquantile(samples, 0.975)), "n_records": int(len(d)),
                   "n_rejected": int(d["reject"].sum())})
    pd.DataFrame(b2).to_csv(out / "b2_false_rejection.csv", index=False)

    # Revised certificate (addendum). cp carries kappa_lb, twin_bearing_resid_w, the item-3 ghost flags
    # (cons_ghost_in_survey_all = all shifted-interval ghosts inside the window; cons_ghost_out_direct_all = all
    # outside the excluded direct AND reflector regions), and window_lo_ns.
    eps_c_factor = 10 ** 0.3  # declared measurement/model margin epsilon_C = 3 dB
    cp = _load(couple_dir, "*.parquet")
    cp["kappa_lb_E_min"] = cp["kappa_lb"] * cp["E_min"]
    cp["forced_minus_subtracted"] = cp["kappa_lb_E_min"] - cp["twin_bearing_resid_w"]
    cp["c_hi_margin"] = c_hi * eps_c_factor
    ghosts_ok = cp["cons_ghost_in_survey_all"] & cp["cons_ghost_out_direct_all"]
    cp["certified_rev"] = ghosts_ok & cp["kappa_lb"].notna() & (cp["forced_minus_subtracted"] > cp["c_hi_margin"])

    def _loss_reason(r) -> str:  # why a geometry is NOT certified under the revised certificate
        if r["certified_rev"]:
            return ""
        if not r["cons_ghost_in_survey_all"]:
            return "ghost_outside_window"
        if not r["cons_ghost_out_direct_all"]:
            return "ghost_in_excluded_gate"
        return "bound_too_weak"

    cp["cons_loss_reason"] = cp.apply(_loss_reason, axis=1)
    cp.to_csv(out / "b1_b3_per_geometry.csv", index=False)
    cp[cp["cons_loss_reason"] != ""][["run", "geometry_id", "cons_loss_reason", "n_gate_routes", "kappa_db",
                                      "kappa_lb_db", "window_lo_ns", "gate_route_ghosts"]].to_csv(
        out / "b3_conservative_losses.csv", index=False)
    cert = []
    for run, part in cp.groupby("run"):
        cert.append({"run": run, "n_geometries": len(part),
                     "median_kappa_db": float(part["kappa_db"].median()),
                     "median_kappa_lb_db": float(part["kappa_lb_db"].median()),
                     "median_window_lo_ns": float(part["window_lo_ns"].median()),
                     "certified_fraction_rev": float(part["certified_rev"].mean()),
                     "n_certified_rev": int(part["certified_rev"].sum()),
                     "n_ghost_outside_window": int((part["cons_loss_reason"] == "ghost_outside_window").sum()),
                     "n_ghost_in_excluded_gate": int((part["cons_loss_reason"] == "ghost_in_excluded_gate").sum()),
                     "n_bound_too_weak": int((part["cons_loss_reason"] == "bound_too_weak").sum())})
    cp_runs = list(cp["run"].unique())
    cert_order = [r for r in ["heldout", *STANDOFFS] if r in cp_runs] + [r for r in cp_runs
                                                                         if r not in ["heldout", *STANDOFFS]]
    cert_df = pd.DataFrame(cert).set_index("run").reindex(cert_order).reset_index()
    cert_df.to_csv(out / "b3_certified_region.csv", index=False)

    # Prop. 3 assumption check: non-reciprocal-wall share of reflector-gate energy, median/max per level.
    # Primary metric is the per-path (incoherent) share e_others / (e_wall + e_others), which is bounded in [0,1]
    # and attributable to paths; the coherent share e_others / e_all is kept for reference but can exceed 1 where
    # the wall path and the others interfere destructively (the coherent total falls below a subset's energy).
    leak = _load(couple_dir.parent / "leak", "*.parquet")
    leak_rows = []
    if not leak.empty and set(leak["run"]) & set(cp_runs):  # skip for the fresh-set report (no matching leak)
        denom = leak["e_gate_wall_only_w"] + leak["e_gate_others_only_w"]
        leak["share_others_incoherent"] = np.where(denom > 0, leak["e_gate_others_only_w"] / denom, np.nan)
        leak.to_csv(out / "prop3_leakage_per_geometry.csv", index=False)
        for run in ["heldout", *STANDOFFS]:
            part = leak[leak["run"] == run]
            s, sc = part["share_others_incoherent"].dropna(), part["share_others"].dropna()
            if len(s):
                leak_rows.append({"run": run, "n_geometries": int(len(s)),
                                  "median_share_others": float(s.median()), "max_share_others": float(s.max()),
                                  "median_share_coherent": float(sc.median()), "max_share_coherent": float(sc.max())})
        pd.DataFrame(leak_rows).to_csv(out / "prop3_leakage.csv", index=False)
    (out / "README.json").write_text(json.dumps({
        "label": "Amendment 13 + revision addendum (defensive). Revised certificate: E_min attacker model, kappa_lb "
                 "lower bound with bearing leakage, twin-derived whole-gate-span window, c_hi*10^(eps_C/10) margin. "
                 "SCOPE: 2nd-order routes; ANY signal from the EVALUATED relay position (ray-traced); NOT a "
                 "verifier-side certificate over unknown relay positions. No attacker code.",
        "b1_b3_per_geometry.csv": "per geometry/level: kappa, kappa_lb, E_min, forced-minus-subtracted, item-3 ghost "
                                  "flags, window_lo_ns, certified_rev, loss reason",
        "b2_threshold.json": "revised c_hi (0.99 quantile of C over calibration genuine targets, revised window); committed before eval",
        "b2_false_rejection.csv": "genuine-target false rejection, revised window, per level",
        "b3_certified_region.csv": "revised certified fraction per level + loss-reason counts + median kappa_lb, window_lo",
        "b3_conservative_losses.csv": "per-geometry loss reason (ghost_outside_window / ghost_in_excluded_gate / bound_too_weak)",
        "prop3_leakage.csv": "Prop.3 check: non-reciprocal-wall share of reflector-gate energy, median/max per level",
        "eps_C_db": 3.0, "c_hi": c_hi, "scripts": ["scripts/wpe_defensive.py"]}, indent=2))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(b2).round(4).to_string())
    print(cert_df.round(4).to_string())
    if leak_rows:
        print(pd.DataFrame(leak_rows).round(6).to_string())


# Amendment 15 (descriptive): coverage of the FROZEN Amendment 14 certificate over bandwidth x map-uncertainty.
COV_BANDWIDTHS = ["bw61", "baseline", "bw246"]   # 61.44, 122.88 (frozen), 245.76 MHz
COV_SIGMAS = [0.0, 0.25, 0.5, 1.0]


def _direct_overlap_ns(ghosts: str, dmin: float, dmax: float) -> float:
    """Largest overlap (ns) of any route's widened ghost interval with the (excluded) direct gate; 0 if none reach it."""
    best = 0.0
    for seg in str(ghosts).split(";"):
        if not seg or ":" not in seg:
            continue
        _, rng, _ = seg.rsplit(":", 2)
        g_lo, g_hi = (float(x) for x in rng.split("-"))
        best = max(best, min(g_hi, dmax) - max(g_lo, dmin))
    return best


def coverage(slot: int, out_dir: Path) -> None:
    """Recompute the frozen Amendment 14 certificate on held-out geometry `slot` across the bandwidth x sigma_map
    grid. Descriptive only: frozen c_hi and the frozen receive-side s_lo/k floor; no threshold is re-fitted. Bandwidth
    enters via _cert_row's config/choices (PSF width, candidate uncertainty); sigma_map via the twin-error widening."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gid = f"ho_{slot:02d}"
    c_hi = json.loads(Path("docs/results/wpe/b2_threshold.json").read_text())["c_hi"]
    margin = c_hi * 10 ** 0.3   # eps_C = 3 dB, identical to report()
    rows = []
    for profile in COV_BANDWIDTHS:
        _, config, config_g, choices, noise_full, direct_weights, geometry, prepared, _ = _setup("heldout", slot, profile)
        bw_mhz = float(config.data["rf"]["profiles"][profile]["occupied_bandwidth_hz"]) / 1e6
        array, rx_array = ArrayGeometry.from_config(config_g), choices.receiver_array
        w = _aoa_window(choices)
        seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, 1)}
        twins = int(config.section("phase8", "grid")["twin_realizations_per_sigma"])
        z = twin_draws(config, HELDOUT, slot, twins)[0]
        base = _acquire_record(config=config_g, choices=choices, world="target", record_index=0,
                               direct_weights=direct_weights, links=prepared["links"],
                               calibration=prepared["calibration"], noise_full=noise_full, timing={}, seeds=seeds)
        for sigma in COV_SIGMAS:
            row = _cert_row("heldout", gid, config, config_g, choices, array, rx_array, w, z, base,
                            direct_weights, prepared, geometry, sigma)
            ghosts_ok = bool(row["cons_ghost_in_survey_all"]) and bool(row["cons_ghost_out_direct_all"])
            kappa_lb = float(row["kappa_lb"])
            forced = kappa_lb * float(row["E_min"]) - float(row["twin_bearing_resid_w"])
            certified = bool(ghosts_ok and math.isfinite(kappa_lb) and forced > margin)
            overlap = _direct_overlap_ns(row["gate_route_ghosts"], row["direct_delay_min_ns"],
                                         row["direct_delay_max_ns"])
            rows.append({"geometry_id": gid, "slot": slot, "bandwidth": profile, "bandwidth_mhz": bw_mhz,
                         "sigma_map_m": sigma, "certified_rev": certified,
                         "cons_ghost_in_survey_all": bool(row["cons_ghost_in_survey_all"]),
                         "cons_ghost_out_direct_all": bool(row["cons_ghost_out_direct_all"]),
                         "kappa_lb_db": float(row["kappa_lb_db"]), "window_lo_ns": float(row["window_lo_ns"]),
                         "direct_overlap_ns": overlap})
            print(f"cov {gid} {profile}({bw_mhz:.2f}MHz) sigma={sigma}: cert={certified} "
                  f"overlap={overlap:.1f}ns win_lo={row['window_lo_ns']:.1f}", flush=True)
    pd.DataFrame(rows).to_parquet(out_dir / f"ho_{slot:02d}.parquet", index=False)


def coverage_report(cov_dir: Path, out: Path, paper_out: Path) -> None:
    """Aggregate the coverage grid: certified fraction per (bandwidth, sigma_map) cell, overlap margins for the
    uncertified ones, and one figure (certified fraction vs bandwidth, one line per sigma_map)."""
    df = pd.concat([pd.read_parquet(p) for p in sorted(cov_dir.glob("ho_*.parquet"))], ignore_index=True)
    cells = []
    for (bw, bw_mhz, sig), part in df.groupby(["bandwidth", "bandwidth_mhz", "sigma_map_m"]):
        unc = part[~part["certified_rev"]]
        cells.append({"bandwidth": bw, "bandwidth_mhz": bw_mhz, "sigma_map_m": sig, "n_geometries": len(part),
                      "n_certified": int(part["certified_rev"].sum()),
                      "certified_fraction": float(part["certified_rev"].mean()),
                      "n_uncertified": len(unc),
                      "median_direct_overlap_ns": float(unc["direct_overlap_ns"].median()) if len(unc) else float("nan"),
                      "max_direct_overlap_ns": float(unc["direct_overlap_ns"].max()) if len(unc) else float("nan")})
    cells_df = pd.DataFrame(cells).sort_values(["sigma_map_m", "bandwidth_mhz"]).reset_index(drop=True)
    cells_df.to_csv(out / "b4_coverage.csv", index=False)
    from paper_figures_tables import OKABE, SIZE, plt, save
    colours = [OKABE["blue"], OKABE["green"], OKABE["orange"], OKABE["vermillion"]]
    fig, ax = plt.subplots(figsize=SIZE)
    for sig, colour in zip(COV_SIGMAS, colours):
        m = cells_df[cells_df["sigma_map_m"] == sig].sort_values("bandwidth_mhz")
        ax.plot(m["bandwidth_mhz"], m["certified_fraction"], marker="o", markersize=3, color=colour,
                label=fr"$\sigma_{{\rm map}}={sig}$ m")
    ax.set_xlabel("Occupied bandwidth (MHz)")
    ax.set_ylabel("Certified fraction (held-out)")
    ax.set_xticks([61.44, 122.88, 245.76])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, paper_out, "fig_c_coverage")
    cells_df.to_csv(paper_out / "fig_c_coverage.csv", index=False)
    pd.set_option("display.width", 220)
    print(cells_df.round(4).to_string())


def main() -> None:
    np.seterr(all="ignore")
    mode = sys.argv[1]
    if mode == "couple":
        couple(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    elif mode == "leakage":
        leakage(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    elif mode == "consistency":
        consistency(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    elif mode == "calibrate":
        calibrate(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "coverage":
        coverage(int(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "coverage_report":
        coverage_report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))


if __name__ == "__main__":
    main()
