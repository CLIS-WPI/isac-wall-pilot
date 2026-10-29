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
"""

from __future__ import annotations

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


def _setup(run: str, slot: int):
    from isac_verify.phase8a import SPLIT
    base_config = PilotConfig.load("configs/pilot.yaml")
    split = SPLIT if run == "calibration" else HELDOUT
    if split == HELDOUT:
        approve_heldout()
    config = stress_config(base_config, run) if run.startswith("standoff_") else base_config
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(choices=choices, config=config,
                                   records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
    direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
                                 ArrayGeometry.from_config(config)).weights
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
    gid = f"ho_{slot:02d}"
    _, config, config_g, choices, noise_full, direct_weights, geometry, prepared, split = _setup(run, slot)
    array, rx_array = ArrayGeometry.from_config(config_g), choices.receiver_array
    w = _aoa_window(choices)
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
        # Conservative Cauchy-Schwarz bound over ALL uplink routes that reach the reflector gate (delay in its
        # window): kappa_cons = e_cand / (sum_i sqrt(e_i^refl))^2 <= kappa (single route), since |a_i| sqrt(G_i) =
        # sqrt(combiner_power(h_i, phi_w)). Any signal segment reaching the gate through route i also reaches the
        # bearing through LoS, landing at refl_center - (tau_i - tau_LoS); every such ghost must be caught.
        survey_lo, survey_hi = (float(v) for v in config.get("phase6", "processing", "delay_search_ns"))
        # Routes that can carry signal energy to the reflector gate: every non-LoS uplink multipath (reciprocal
        # wall, opposite wall, double bounces). Uplink one-way delays are a different frame than the round-trip gate,
        # so inclusion is by route family, not absolute delay; each route's ghost (item 2) handles its delay.
        routes = [p for p in prepared["links"]["uplink"] if p.family != "los"]
        sqrt_sum, ghosts, in_survey_all, out_direct_all = 0.0, [], True, True
        for p in routes:
            e_i = combiner_power(p.matrix[:, 0].to(torch.complex64), phi_w, rx_array, w)
            sqrt_sum += math.sqrt(max(e_i, 0.0))
            ghost_ns = refl.center_delay_s * 1e9 + (los.delay_s - p.delay_s) * 1e9
            in_survey = survey_lo <= ghost_ns <= survey_hi
            in_direct = dgate.delay_min_s * 1e9 <= ghost_ns <= dgate.delay_max_s * 1e9
            in_survey_all &= in_survey
            out_direct_all &= not in_direct
            ghosts.append(f"{p.family}:{ghost_ns:.1f}:{'S' if in_survey else 's'}{'D' if in_direct else 'd'}")
        kappa_cons = e_cand / sqrt_sum ** 2 if sqrt_sum > 0 else float("nan")
        row |= {"kappa_cons": kappa_cons, "kappa_cons_db": 10 * math.log10(max(kappa_cons, 1e-30)),
                "n_gate_routes": len(routes), "cons_ghost_in_survey_all": bool(in_survey_all),
                "cons_ghost_out_direct_all": bool(out_direct_all), "gate_route_ghosts": ";".join(ghosts)}
    else:
        row |= {"G_db": float("nan"), "kappa_db": float("nan"), "kappa": float("nan"),
                "kappa_cons": float("nan"), "kappa_cons_db": float("nan"), "n_gate_routes": 0,
                "cons_ghost_in_survey_all": False, "cons_ghost_out_direct_all": False, "gate_route_ghosts": "",
                "missing_uplink": f"los={los is not None} wall_{side}={wall is not None}"}
    pd.DataFrame([row]).to_parquet(out_dir / f"{run}_{slot:02d}.parquet", index=False)
    print(f"{run} {gid}: G={row.get('G_db', float('nan')):.2f} dB kappa={row.get('kappa_db', float('nan')):.2f} dB "
          f"kappa_cons={row.get('kappa_cons_db', float('nan')):.2f} dB routes={row.get('n_gate_routes')} "
          f"overlap={row.get('overlap')}", flush=True)


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
    gid = f"{'ho' if split == HELDOUT else 'cal'}_{slot:02d}"
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
            cand_gate = _cand_residual_gate(float(base["candidate"].azimuth_deg), plan.direct_gate, refl, survey,
                                            aoa_half)
            twin_paths = _twin_paths(config_g, base["candidate"], offsets, {})
            twin_direct = _twin_map(config=config_g, choices=choices, weights=direct_weights, twin_paths=twin_paths,
                                    timing={})
            c = (_residual_energy(base["direct_map"], cand_gate, plan.direct_gate, refl)
                 - _residual_energy(twin_direct, cand_gate, plan.direct_gate, refl))
            rows.append({"run": run, "geometry_id": gid, "world": "target", "record_index": record,
                         "sigma_map_m": sigma, "status": "computed", "C": c, "candidate_bearing_deg":
                         float(base["candidate"].azimuth_deg)})
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
        "label": "B13.2 pre-registered: 1% false rejection of genuine targets on the 12 calibration geometries, pooled sigma",
        "c_hi": c_hi, "n_calibration_target_records": int(len(c)),
        "realized_false_rejection": float((c > c_hi).mean())}, indent=2))
    print(json.dumps({"c_hi": c_hi, "n": len(c), "realized_fr": float((c > c_hi).mean())}, indent=1))


def report(couple_dir: Path, cons_dir: Path, out: Path, geometries_path: Path) -> None:
    from phase8b_heldout_analysis import Bootstrap
    from wpb_reanalysis import cp_upper_zero

    config = PilotConfig.load("configs/pilot.yaml")
    c_hi = json.loads((out / "b2_threshold.json").read_text())["c_hi"]
    passive = pd.read_parquet(Path("docs/results/wpc_a12/c1_passive_rows.parquet"))
    passive = passive[(passive["passive_status"] == "computed") & (passive["world"] == "target")
                      & (passive["sigma_map_m"] == SIGMA)]
    eps = float(config.get("phase6", "evidence", "epsilon_w"))
    e_req = (passive.assign(E_req=np.maximum(passive["E_D"] - passive["N_D"], eps) * passive["r_pass"]
                            + passive["N_pass"]).groupby(["run", "geometry_id"])["E_req"].mean().reset_index())
    boot = Bootstrap(sorted(pd.read_csv(geometries_path)["geometry_id"]),
                     int(config.get("phase8", "heldout", "bootstrap_draws")))

    # B2 false rejection on genuine targets
    b2 = []
    for run in ["heldout", *STANDOFFS]:
        d = _load(cons_dir, f"{run}_*.parquet")
        d = d[d["status"] == "computed"].assign(reject=lambda f: f["C"] > c_hi)
        point, samples = boot.rate(d, "reject")
        b2.append({"run": run, "false_rejection": point, "ci95_lo": float(np.nanquantile(samples, 0.025)),
                   "ci95_hi": float(np.nanquantile(samples, 0.975)), "n_records": int(len(d)),
                   "n_rejected": int(d["reject"].sum())})
    pd.DataFrame(b2).to_csv(out / "b2_false_rejection.csv", index=False)

    # B1 + B3
    cp = _load(couple_dir, "*.parquet")
    cp = cp.merge(e_req, on=["run", "geometry_id"], how="left")
    cp["kappa_E_req"] = cp["kappa"] * cp["E_req"]
    cp["margin3db_factor"] = 10 ** 0.3
    # Window check (external review): the forced LoS copy must be caught by the consistency statistic,
    # i.e. its delay sits inside the surveillance window AND outside the direct gate (overlap == False).
    survey_lo, survey_hi = (float(v) for v in config.get("phase6", "processing", "delay_search_ns"))
    cp["los_in_survey"] = ((cp["los_copy_at_refl_center_ns"] >= survey_lo)
                           & (cp["los_copy_at_refl_center_ns"] <= survey_hi))
    certifiable = (~cp["overlap"]) & cp["los_in_survey"] & cp["kappa"].notna()
    cp["certified_m0"] = certifiable & (cp["kappa_E_req"] > c_hi)
    cp["certified_m3db"] = certifiable & (cp["kappa_E_req"] > c_hi * cp["margin3db_factor"])
    # Conservative certificate (multi-route): every gate route's LoS ghost caught AND kappa_cons * E_req > c_hi.
    cp["kappa_cons_E_req"] = cp["kappa_cons"] * cp["E_req"]
    cons_ghosts_ok = cp["cons_ghost_in_survey_all"] & cp["cons_ghost_out_direct_all"]
    cons_ok = cons_ghosts_ok & cp["kappa_cons"].notna()
    cp["certified_cons_m0"] = cons_ok & (cp["kappa_cons_E_req"] > c_hi)
    cp["certified_cons_m3db"] = cons_ok & (cp["kappa_cons_E_req"] > c_hi * cp["margin3db_factor"])

    def _loss_reason(r) -> str:  # why an old-certified geometry loses the conservative certificate
        if not r["certified_m0"] or r["certified_cons_m0"]:
            return ""
        if not r["cons_ghost_in_survey_all"]:
            return "ghost_outside_window"
        if not r["cons_ghost_out_direct_all"]:
            return "ghost_in_direct_gate"
        return "bound_too_weak"

    cp["cons_loss_reason"] = cp.apply(_loss_reason, axis=1)
    cp.to_csv(out / "b1_b3_per_geometry.csv", index=False)
    cp[cp["cons_loss_reason"] != ""][["run", "geometry_id", "cons_loss_reason", "n_gate_routes", "kappa_db",
                                      "kappa_cons_db", "gate_route_ghosts"]].to_csv(
        out / "b3_conservative_losses.csv", index=False)
    cert = []
    for run, part in cp.groupby("run"):
        n = len(part)
        cert.append({"run": run, "n_geometries": n, "overlap_fraction": float(part["overlap"].mean()),
                     "median_G_db": float(part["G_db"].median()), "median_kappa_db": float(part["kappa_db"].median()),
                     "median_kappa_cons_db": float(part["kappa_cons_db"].median()),
                     "certified_fraction_m0": float(part["certified_m0"].mean()),
                     "certified_fraction_m3db": float(part["certified_m3db"].mean()),
                     "n_certified_m0": int(part["certified_m0"].sum()),
                     "certified_fraction_cons_m0": float(part["certified_cons_m0"].mean()),
                     "certified_fraction_cons_m3db": float(part["certified_cons_m3db"].mean()),
                     "n_certified_cons_m0": int(part["certified_cons_m0"].sum()),
                     "n_window_failed": int(((~part["overlap"]) & (~part["los_in_survey"])).sum())})
    cert_df = pd.DataFrame(cert).set_index("run").reindex(["heldout", *STANDOFFS]).reset_index()
    cert_df.to_csv(out / "b3_certified_region.csv", index=False)

    # Prop. 3 assumption check: non-reciprocal-wall share of reflector-gate energy, median/max per level.
    # Primary metric is the per-path (incoherent) share e_others / (e_wall + e_others), which is bounded in [0,1]
    # and attributable to paths; the coherent share e_others / e_all is kept for reference but can exceed 1 where
    # the wall path and the others interfere destructively (the coherent total falls below a subset's energy).
    leak = _load(couple_dir.parent / "leak", "*.parquet")
    leak_rows = []
    if not leak.empty:
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
        "label": "Amendment 13 (defensive) pre-registered: B1 coupling, B2 consistency statistic, B3 certified region. No attacker code.",
        "b1_b3_per_geometry.csv": "per geometry/level: G (dB), kappa (dB), delay offset, overlap, E_req, kappa*E_req, certified flags",
        "b2_threshold.json": "B2 c_hi (0.99 quantile of C over calibration genuine targets); committed before eval",
        "b2_false_rejection.csv": "B2 genuine-target false rejection on held-out (sigma 0.5) and each standoff level",
        "b3_certified_region.csv": "certified fraction per level (margin 0 and 3 dB) vs overlap and median G",
        "prop3_leakage.csv": "Prop.3 check: non-reciprocal-wall share of reflector-gate energy, median/max per level",
        "c_hi": c_hi, "scripts": ["scripts/wpe_defensive.py"]}, indent=2))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(b2).round(4).to_string())
    print(cert_df.round(4).to_string())
    if leak_rows:
        print(pd.DataFrame(leak_rows).round(6).to_string())


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
    else:
        report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))


if __name__ == "__main__":
    main()
