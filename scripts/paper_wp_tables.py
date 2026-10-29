"""LaTeX table bodies (booktabs rows only) for WP-B, WP-C and D1, from their saved CSVs. No new numbers.

  table_wpb_operating_points.tex  B3 (secondary, approximate: 10 dB power grid)
  table_wpc_passive.tex           C1 re-run with the corrected gate (A12.4 pre-registered), passive vs active
  table_wpc_passive_a11.tex       C1 as run under A11.1 (mirrored gate, DEVIATIONS P8-14)
  table_wpc_controller.tex        C2 (A11.2 pre-registered): probes per candidate and spectrum-time cost
  table_wpd_threshold_free.tex    D1 definitions on the A12.4 rows; _a11: on the A11.1 rows

Usage: python scripts/paper_wp_tables.py OUT_DIR   (reads docs/results/{wpb,wpc,wpd,wpc_a12,wpd_a12})
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from paper_figures_tables import R, fmt

METHOD = {"imperfect_robust": "Robust (map)", "imperfect_narrow": "Narrow (map)",
          "mapfree_coarse_to_fine": "Coarse-to-fine (map-free)"}
RUN = {"heldout": r"Held-out, $\sigma=0.5$", **{f"standoff_{m:02d}m": f"$\\Delta R={m}$ m" for m in (3, 6, 10, 15, 20, 30)}}


def write(out: Path, name: str, header: list[str], rows: list[str]) -> None:
    (out / name).write_text("\n".join(header + rows + [r"\bottomrule"]) + "\n")


def b3(out: Path) -> None:
    d = pd.read_csv(R / "wpb/b3_operating_points.csv")
    rows = []
    for method, part in d.groupby("method", sort=False):
        op = part[part["target_level"].notna()].set_index("target_level")
        ceil = part[part["replay_ceiling"].notna()].set_index("replay_ceiling")
        rows.append(f"{METHOD[method]} & {op.loc[0.5, 'power_dbm']:.1f} & {op.loc[0.5, 'replay_at_power']:.3f} & "
                    f"{op.loc[0.8, 'power_dbm']:.1f} & {op.loc[0.8, 'replay_at_power']:.3f} & "
                    f"{ceil.loc[0.05, 'best_target_at_ceiling']:.3f} ({ceil.loc[0.05, 'power_dbm']:+.0f}) & "
                    f"{ceil.loc[0.1, 'best_target_at_ceiling']:.3f} ({ceil.loc[0.1, 'power_dbm']:+.0f}) \\\\")
    write(out, "table_wpb_operating_points.tex", [
        "% B3 (WP-B): secondary analysis of the probe-power stress levels plus the main run (10 dBm), sigma 0.5.",
        "% Power for target SUPPORTED 0.5 / 0.8 by linear interpolation in dBm: approximate (10 dB grid).",
        "% Best target SUPPORTED at a replay ceiling: measured grid points only, power in dBm in parentheses.",
        "% Coarse-to-fine uses SUPPORTED_common. Source: docs/results/wpb/b3_operating_points.csv.",
        "% columns: l r r r r r r", r"\toprule",
        r"Method & $P_{0.5}$ (dBm) & Replay & $P_{0.8}$ (dBm) & Replay & Target @ replay$\le$0.05 & "
        r"Target @ replay$\le$0.10 \\", r"\midrule"], rows)


def c1(out: Path, src: str, name: str, notes: list[str]) -> None:
    d = pd.read_csv(R / src / "c1_passive_vs_active.csv")
    th = json.loads((R / src / "c1_passive_thresholds.json").read_text())
    rows = []
    for run, label in RUN.items():
        t = d[(d["run"] == run) & (d["world"] == "target")].iloc[0]
        r = d[(d["run"] == run) & (d["world"] == "replay")].iloc[0]
        rows.append(f"{label} & {t.passive_point:.3f} & {t.active_point:.3f} & "
                    f"{fmt(r.passive_point, r.passive_ci95_lo, r.passive_ci95_hi)} & "
                    f"{fmt(r.active_point, r.active_ci95_lo, r.active_ci95_hi)} & "
                    f"{fmt(r.passive_minus_active, r.diff_ci95_lo, r.diff_ci95_hi)} \\\\")
    write(out, name, [
        "% Passive twin baseline (direct action only) vs the frozen active imperfect_robust probe, SUPPORTED rate,",
        "% sigma 0.5; geometry bootstrap 95% CI. Passive thresholds from the 12 calibration geometries:",
        f"% k={th['floor_k']['value']:.4f}, s_lo={th['s_lo']['value']:.4f} (binding sigma={th['s_lo']['binding_sigma']}),"
        f" s_hi={th['s_hi']['value']:.4f}; active s_lo=0.022.",
        *notes, f"% Source: docs/results/{src}/c1_passive_vs_active.csv.",
        "% columns: l r r r c r c r c", r"\toprule",
        r"Run & \multicolumn{2}{c}{Target} & \multicolumn{4}{c}{Replay SUPPORTED} & "
        r"\multicolumn{2}{c}{Passive $-$ active} \\",
        r"& Passive & Active & \multicolumn{2}{c}{Passive} & \multicolumn{2}{c}{Active} & & \\", r"\midrule"], rows)


def c2(out: Path) -> None:  # probes per candidate = 1 + escalations / candidates
    d = pd.read_csv(R / "wpc/c2_controller.csv")
    rows = []
    for run, label in (("heldout", r"Held-out, $\sigma=0.5$"), ("standoff_15m", r"$\Delta R=15$ m")):
        r = d[(d["run"] == run) & (d["world"] == "replay")].iloc[0]
        t = d[(d["run"] == run) & (d["world"] == "target")].iloc[0]
        rows.append(f"{label} & {int(r['count'])}/{int(r.candidates)} & [{r.ci95_lo:.3f}, {r.ci95_hi:.3f}] & "
                    f"{int(t['count'])}/{int(t.candidates)} & {int(r.escalated)} & {int(r.fresh_supported)} & "
                    f"{1 + r.escalated / r.candidates:.3f} & {1 + t.escalated / t.candidates:.3f} & "
                    f"{r.mean_cost_wall_probes:.3f} & {r.mean_cost_incl_fresh_direct:.3f} & "
                    f"{t.mean_cost_wall_probes:.2f} \\\\")
    write(out, "table_wpc_controller.tex", [
        "% C2 (ANALYSIS_PLAN A11.2, pre-registered): escalate a SUPPORTED first probe with S < s_esc = 0.1784 to a",
        "% FRESH 245.76 MHz trial (new seed bases 33201/34201/35201, same twin draw); authorize only if SUPPORTED.",
        "% Probes per candidate = 1 + escalations / candidates (wall probes sent, any bandwidth).",
        "% Spectrum-time cost unit: one 122.88 MHz wall probe (a 245.76 MHz probe costs 2); '+direct' adds the",
        "% fresh trial's own 245.76 MHz direct action.",
        "% 74 new trials (47 held-out, 27 at 15 m). Source: docs/results/wpc/c2_controller.csv.",
        "% columns: l r c r r r r r r r r", r"\toprule",
        r"Run & Replays authorized & 95\% CI & Targets held & Escalated & Fresh SUPPORTED & "
        r"Probes/cand. (replay) & Probes/cand. (target) & "
        r"Cost (replay) & Cost +direct (replay) & Cost (target) \\", r"\midrule"], rows)


def d1(out: Path, src: str, name: str) -> None:
    d = pd.read_csv(R / src / "d1_threshold_free.csv")
    d = d[d["run"] != "calibration"]
    rows = []
    for run, label in RUN.items():
        a = d[(d["run"] == run) & (d["score"] == "active")].iloc[0]
        p = d[(d["run"] == run) & (d["score"] == "passive")].iloc[0]
        rows.append(f"{label} & {a.auc:.3f} & {p.auc:.3f} & {a.oracle_s_lo:.3f} & {a.oracle_replay:.3f} & "
                    f"{p.oracle_s_lo:.3f} & {p.oracle_replay:.3f} & {a.frozen_replay:.3f} & {p.frozen_replay:.3f} \\\\")
    write(out, name, [
        "% D1: EXPLORATORY, post hoc. AUC of the score (target vs replay; rows without a finite score rank lowest).",
        "% Oracle s_lo: largest s_lo with target SUPPORTED >= 0.99 on that level's own rows, every other frozen",
        "% condition kept; replay SUPPORTED at that s_lo. Not a usable operating point (fitted on the evaluated rows).",
        f"% Source: docs/results/{src}/d1_threshold_free.csv (bootstrap CIs there).",
        "% columns: l r r r r r r r r", r"\toprule",
        r"Run & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{Active, oracle} & \multicolumn{2}{c}{Passive, oracle} & "
        r"\multicolumn{2}{c}{Replay, frozen} \\",
        r"& Active & Passive & $s_{lo}$ & Replay & $s_{lo}$ & Replay & Active & Passive \\", r"\midrule"], rows)


def e_b3(out: Path) -> None:  # Amendment 13 (defensive): B3 certified region, with B2 false rejection in the caption.
    cert = pd.read_csv(R / "wpe/b3_certified_region.csv").set_index("run")
    fr = pd.read_csv(R / "wpe/b2_false_rejection.csv").set_index("run")
    c_hi = json.loads((R / "wpe/b2_threshold.json").read_text())["c_hi"]
    rows = []
    for run, label in RUN.items():
        c = cert.loc[run]
        rows.append(f"{label} & {c.certified_fraction_m0:.3f} ({int(c.n_certified_m0)}/{int(c.n_geometries)}) & "
                    f"{c.overlap_fraction:.3f} & {c.median_G_db:+.1f} & {c.median_kappa_db:+.1f} \\\\")
    fr_note = "; ".join(
        f"{RUN[run].replace('$', '')}: {fr.loc[run].false_rejection:.4f} "
        f"[{fr.loc[run].ci95_lo:.4f}, {fr.loc[run].ci95_hi:.4f}] ({int(fr.loc[run].n_rejected)}/{int(fr.loc[run].n_records)})"
        for run in RUN)
    write(out, "table_wpe_certified.tex", [
        "% Amendment 13 (defensive), B3: certified region against the single-antenna-emitter relay class, per level",
        "% (n=24 held-out geometries, 24 per standoff). CERTIFIED = relay delay-resolved (LoS copy outside the direct",
        "% gate) AND its forced LoS copy falls inside the surveillance window AND kappa*E_req > c_hi; the 3 dB-margin",
        "% variant is identical. G = reflector-gate angle response at the relay wall arrival vs the steered peak;",
        "% kappa = LoS-into-candidate-gate over wall-into-reflector-gate. Source: docs/results/wpe/b3_certified_region.csv.",
        f"% c_hi = {c_hi:.3e} (0.99 quantile of C over 1920 calibration genuine targets; realized calibration FR 1.04%).",
        f"% B2 genuine-target false rejection (point [95% CI] (rejected/records)): {fr_note}.",
        "% columns: l r r r r", r"\toprule",
        r"Level & Certified & Overlap & Median $G$ (dB) & Median $\kappa$ (dB) \\", r"\midrule"], rows)


def main() -> None:
    out = Path(sys.argv[1])
    b3(out)
    c1(out, "wpc_a12", "table_wpc_passive.tex", [
        "% ANALYSIS_PLAN A12.4 (pre-registered): C1 re-run with the A12.1 receiver fix; the gate lies on the planned",
        "% wall side. Thresholds re-fitted by the A11.1 rules before any held-out or standoff score."])
    c1(out, "wpc", "table_wpc_passive_a11.tex", [
        "% ANALYSIS_PLAN A11.1 as run: pre-fix receiver, so the passive gate observed the opposite wall's",
        "% reciprocal path (DEVIATIONS P8-14). Prediction (passive replay higher at 15 and 20 m) NOT matched."])
    c2(out)
    d1(out, "wpd_a12", "table_wpd_threshold_free.tex")
    d1(out, "wpd", "table_wpd_threshold_free_a11.tex")
    e_b3(out)
    for name in ("wpb_operating_points", "wpc_passive", "wpc_passive_a11", "wpc_controller", "wpd_threshold_free",
                 "wpd_threshold_free_a11", "wpe_certified"):
        print((out / f"table_{name}.tex").read_text())


if __name__ == "__main__":
    main()
