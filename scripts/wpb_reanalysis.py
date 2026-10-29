"""WP-B (external review): re-analyses of existing rows only. No trials, no new thresholds.

Main held-out run = primary data; every stress / standoff level re-analysis is labelled "secondary".
Decisions are the frozen ones (`phase8.frozen`); the "no s_hi" variant is the A3.2 one-sided rule.
Intervals: A4.1 geometry-level bootstrap. B4 adds, for every zero rate, the number of geometries with any
supported row (0) and the one-sided 95% Clopper-Pearson upper bound for 0 of n geometries, 1 - 0.05**(1/n).

  B1 robust target / replay SUPPORTED with and without s_hi (main per sigma; every level at sigma 0.5)
  B2 coarse_to_fine SUPPORTED_common replay next to robust SUPPORTED replay (main per sigma; power levels)
  B3 power at target 0.5 / 0.8 by linear interpolation in dBm (approximate: 10 dB grid), replay there
  B4 zero rates with geometry counts and Clopper-Pearson bounds (B1, B2, E2, stratum tables)
  B5 row decomposition of the main run and the three OFDM profiles

Usage: python scripts/wpb_reanalysis.py MAIN_TRIALS.parquet GEOMETRIES.csv STRESS_ROOT OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.config import PilotConfig
from isac_verify.waveform import profile_from_config
from phase8b_heldout_analysis import Bootstrap, ci, classify

LEVELS = ["bw246", "bw61", *(f"standoff_{m:02d}m" for m in (3, 6, 10, 15, 20, 30)), "array_error_d11",
          "probe_power_m20dbm", "probe_power_m10dbm", "probe_power_0dbm", "probe_power_p20dbm",
          "wall_thickness_50mm", "wall_thickness_200mm", "rcs_p6db", "rcs_m6db"]
POWER = {"probe_power_m20dbm": -20.0, "probe_power_m10dbm": -10.0, "probe_power_0dbm": 0.0, "main": 10.0,
         "probe_power_p20dbm": 20.0}
SIGMA = 0.5
METHOD_RULE = {"imperfect_robust": "supported", "imperfect_narrow": "supported",
               "mapfree_coarse_to_fine": "supported_common"}


def cp_upper_zero(n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound for 0 successes in n units."""
    return 1.0 - 0.05 ** (1.0 / n)


def entry(boot: Bootstrap, rows: pd.DataFrame, column: str) -> dict:
    point, samples = boot.rate(rows, column)
    n_geom = int(rows["geometry_id"].nunique())
    any_geom = int(rows.groupby("geometry_id")[column].any().sum())
    return {"point": point, "ci95_lo": ci(samples)[0], "ci95_hi": ci(samples)[1], "n_rows": int(len(rows)),
            "n_geometries": n_geom, "geometries_with_any_supported": any_geom,
            "cp95_upper_if_zero": cp_upper_zero(n_geom) if any_geom == 0 else float("nan")}


def interpolate_crossing(power: np.ndarray, rate: np.ndarray, level: float) -> float | None:
    """First upward crossing of `level` by linear interpolation in dBm; None if never reached."""
    if rate[0] >= level:
        return float(power[0])  # reached at or below the lowest tested power
    for i in range(len(power) - 1):
        if rate[i] < level <= rate[i + 1]:
            return float(power[i] + (level - rate[i]) * (power[i + 1] - power[i]) / (rate[i + 1] - rate[i]))
    return None


def _check() -> None:
    assert abs(interpolate_crossing(np.array([-20.0, -10.0, 0.0]), np.array([0.0, 0.4, 0.8]), 0.6) + 5.0) < 1e-9
    assert interpolate_crossing(np.array([-20.0, -10.0]), np.array([0.0, 0.4]), 0.5) is None
    assert abs(cp_upper_zero(24) - 0.1173) < 1e-4


def main() -> None:
    main_path, geometries_path, stress_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    _check()
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    ids = sorted(pd.read_csv(geometries_path)["geometry_id"])
    boot = Bootstrap(ids, draws)
    main_frame = classify(pd.read_parquet(main_path), config)
    frames = {"main": main_frame}
    for level in LEVELS:
        frames[level] = classify(pd.read_parquet(stress_root / level / "heldout_trials.parquet"), config)
        assert sorted(frames[level]["geometry_id"].unique()) == ids, level

    def cells(source: str):
        frame = frames[source]
        if source == "main":
            yield "pooled", frame
            for s in sorted(frame["sigma_map_m"].unique()):
                yield f"{s:g}", frame[frame["sigma_map_m"] == s]
        else:
            yield f"{SIGMA:g}", frame[frame["sigma_map_m"] == SIGMA]

    b1, b2 = [], []
    for source in ["main", *LEVELS]:
        role = "primary" if source == "main" else "secondary"
        for sigma, part in cells(source):
            robust = part[part["method"] == "imperfect_robust"]
            for world in ("target", "replay"):
                w = robust[robust["world"] == world]
                for rule, column in (("with_s_hi", "supported"), ("no_s_hi", "supported_no_s_hi")):
                    b1.append({"analysis": role, "source": source, "sigma_map_m": sigma, "method": "imperfect_robust",
                               "world": world, "rule": rule, **entry(boot, w, column)})
            if source == "main" or source in POWER:
                for method, column in (("imperfect_robust", "supported"),
                                       ("mapfree_coarse_to_fine", "supported_common")):
                    w = part[(part["method"] == method) & (part["world"] == "replay")]
                    b2.append({"analysis": role, "source": source, "sigma_map_m": sigma, "method": method,
                               "world": "replay", "rule": column, **entry(boot, w, column)})
    b1, b2 = pd.DataFrame(b1), pd.DataFrame(b2)

    # B3: target and replay vs probe power, sigma 0.5 (10 dBm = main run)
    grid = []
    for source, power in POWER.items():
        part = frames[source][frames[source]["sigma_map_m"] == SIGMA]
        for method, column in METHOD_RULE.items():
            for world in ("target", "replay"):
                w = part[(part["method"] == method) & (part["world"] == world)]
                grid.append({"analysis": "primary" if source == "main" else "secondary", "source": source,
                             "power_dbm": power, "method": method, "rule": column, "world": world,
                             **entry(boot, w, column)})
    grid = pd.DataFrame(grid).sort_values(["method", "world", "power_dbm"])
    b3 = []
    for method in METHOD_RULE:
        t = grid[(grid["method"] == method) & (grid["world"] == "target")]
        r = grid[(grid["method"] == method) & (grid["world"] == "replay")]
        p, tr, rr = t["power_dbm"].to_numpy(), t["point"].to_numpy(), r["point"].to_numpy()
        for level in (0.5, 0.8):
            x = interpolate_crossing(p, tr, level)
            b3.append({"label": "secondary, approximate (linear interpolation on a 10 dB grid)", "method": method,
                       "rule": METHOD_RULE[method], "target_level": level, "power_dbm": x,
                       "replay_at_power": float(np.interp(x, p, rr)) if x is not None else float("nan"),
                       "reached_at_lowest_power": bool(x is not None and x == p[0])})
        for ceiling in (0.05, 0.10):
            ok = rr <= ceiling
            b3.append({"label": "secondary, measured grid points only", "method": method,
                       "rule": METHOD_RULE[method], "replay_ceiling": ceiling,
                       "best_target_at_ceiling": float(tr[ok].max()) if ok.any() else float("nan"),
                       "power_dbm": float(p[ok][np.argmax(tr[ok])]) if ok.any() else float("nan")})
    b3 = pd.DataFrame(b3)

    # B4: zero rates from B1, B2, the power grid, E2 and the stratum tables
    zeros = []
    for name, frame in (("B1", b1), ("B2", b2), ("B3_grid", grid)):
        z = frame[frame["point"] == 0.0]
        for r in z.itertuples():
            zeros.append({"table": name, "cell": f"{r.source} sigma={r.sigma_map_m if 'sigma_map_m' in z else SIGMA} "
                                                f"{r.method} {r.world} {r.rule}",
                          "n_geometries": r.n_geometries, "geometries_with_any_supported": r.geometries_with_any_supported,
                          "cp95_upper": cp_upper_zero(r.n_geometries)})
    e2 = pd.read_csv("docs/results/paper/e2_escalation.csv")
    for r in e2.itertuples():
        for world in ("target", "replay"):
            if getattr(r, f"{world}_supported") == 0.0:
                zeros.append({"table": "E2", "cell": f"{r.pairing} {r.policy} {world}", "n_geometries": r.n_geometries,
                              "geometries_with_any_supported": 0, "cp95_upper": cp_upper_zero(int(r.n_geometries))})
    for path in ("docs/results/phase8c/rates_by_level_stratum.csv", "docs/results/phase8c/block45/rates_by_level_stratum.csv",
                 "docs/results/phase8c/array_error/rates_by_stratum.csv"):
        s = pd.read_csv(path)
        trivial = s["method"].str.startswith("mapfree") & (s["rate"] == "supported")  # map-free has no S
        for r in s[(s["point"] == 0.0) & ~trivial].itertuples():
            desc = " ".join(f"{c}={getattr(r, c)}" for c in s.columns if c not in ("point", "ci95", "n_rows", "n_geometries"))
            zeros.append({"table": Path(path).name, "cell": desc, "n_geometries": int(r.n_geometries),
                          "geometries_with_any_supported": 0, "cp95_upper": cp_upper_zero(int(r.n_geometries))})
    b4 = pd.DataFrame(zeros)

    # B5
    m = main_frame
    counts = {"geometries": m["geometry_id"].nunique(), "sigma_map_m": m["sigma_map_m"].nunique(),
              "records_per_world": m["record_index"].nunique(), "hypotheses": m["world"].nunique(),
              "methods": m["method"].nunique()}
    product = int(np.prod(list(counts.values())))
    assert product == len(m) == 46080, (product, len(m))
    assert (m["twin_index"] == m["record_index"] % 4).all()
    profiles = []
    for name in ("bw61", "baseline", "bw246"):
        p = profile_from_config(config, name)
        profiles.append({"profile": name, "bandwidth_mhz": p.occupied_bandwidth_hz / 1e6, "fft_size_N": p.fft_size,
                         "subcarrier_spacing_khz": p.subcarrier_spacing_hz / 1e3, "cp_length": p.cp_length,
                         "sample_rate_mhz": p.sample_rate_hz / 1e6, "symbol_duration_us_incl_cp": p.symbol_duration_s * 1e6,
                         "symbols_per_action": p.num_symbols, "action_duration_ms": p.record_duration_s * 1e3,
                         "delay_bin_ns": p.delay_bin_spacing_s * 1e9})
    b5 = {"rows": len(m), "factors": {k: int(v) for k, v in counts.items()},
          "methods": sorted(m["method"].unique()), "sigmas": sorted(float(s) for s in m["sigma_map_m"].unique()),
          "twin_realizations": int(m["twin_index"].nunique()), "twin_rule": "twin_index = record_index mod 4",
          "profiles": profiles}

    b1.to_csv(out / "b1_one_sided_rule.csv", index=False)
    b2.to_csv(out / "b2_c2f_vs_robust_replay.csv", index=False)
    grid.to_csv(out / "b3_power_grid.csv", index=False)
    b3.to_csv(out / "b3_operating_points.csv", index=False)
    b4.to_csv(out / "b4_zero_rates.csv", index=False)
    (out / "b5_decomposition.json").write_text(json.dumps(b5, indent=2))
    (out / "README.json").write_text(json.dumps({
        "label": "WP-B: re-analysis of existing rows; main held-out run = primary; stress/standoff levels = secondary",
        "b4_note": "The geometry bootstrap interval [0, 0] for a zero rate is not a bound: every resample of a "
                   "zero cell is zero. cp95_upper is the one-sided 95% Clopper-Pearson bound for 0 of n "
                   "geometries with any supported row; it also bounds the row rate when rows per geometry are equal.",
        "b3_note": "approximate: linear interpolation in dBm on a 10 dB grid; 10 dBm point is the main run",
    }, indent=2))
    pd.set_option("display.width", 250)
    show = ["source", "sigma_map_m", "world", "rule", "point", "ci95_lo", "ci95_hi", "geometries_with_any_supported"]
    print(b1[show].round(3).to_string())
    print(b2[show[:2] + ["method"] + show[2:]].round(3).to_string())
    print(grid[["method", "world", "power_dbm", "point", "ci95_lo", "ci95_hi"]].round(3).to_string())
    print(b3.round(3).to_string())
    print(b4.round(3).to_string())
    print(json.dumps(b5, indent=1))


if __name__ == "__main__":
    main()
