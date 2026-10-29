"""Paper figures and LaTeX table bodies from saved results only (no new trials, no new thresholds).

Fig. A (standoff sweep) and Fig. B (power sweep): IEEE single column, 3.5 in wide, 8 pt, Okabe-Ito colours,
no titles, 95% geometry-bootstrap bands. Tables: booktabs rows only (no tabular environment).

The one number computed here that no earlier output holds is the 10 dBm imperfect_narrow target rate in Fig. B
(main held-out run, sigma 0.5); it is labelled "exploratory, post hoc" in fig_b_power_sweep.csv.

Usage: python scripts/paper_figures_tables.py MAIN_TRIALS.parquet GEOMETRIES.csv OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from isac_verify.config import PilotConfig  # noqa: E402
from phase8b_heldout_analysis import Bootstrap, rate_entry  # noqa: E402
from phase8c_block45_analysis import POWER_LEVELS, cell  # noqa: E402

R = Path("docs/results")
OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00",
         "purple": "#CC79A7", "black": "#000000"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "savefig.dpi": 600})
SIZE = (3.5, 2.5)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout(pad=0.3)
    fig.savefig(out / f"{name}.pdf", metadata={"Subject": "data provenance: docs/results/paper/README.json"})
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def series(ax, x, frame: pd.DataFrame, colour: str, label: str, marker: str, dashed: bool = False) -> None:
    ax.fill_between(x, frame["lo"], frame["hi"], color=colour, alpha=0.18, linewidth=0)
    ax.plot(x, frame["point"], color=colour, marker=marker, markersize=3.5, label=label,
            linestyle="--" if dashed else "-")


def fig_a(out: Path) -> pd.DataFrame:
    d = pd.read_csv(R / "phase8c/standoff/replay_by_level.csv")
    d["delta_r_m"] = d["level"].str.extract(r"standoff_(\d+)m")[0].astype(float)
    d[["lo", "hi"]] = pd.DataFrame(d["ci95"].map(json.loads).tolist(), index=d.index)
    s = pd.read_csv(R / "phase8c/standoff/standoffs.csv")
    phi, tau = float(s["delta_r_phi_m"].median()), float(s["delta_r_tau_baseline_m"].median())
    fig, ax = plt.subplots(figsize=SIZE)
    for method, colour, label, marker in (("exact_oracle", OKABE["blue"], "Oracle", "o"),
                                          ("imperfect_robust", OKABE["vermillion"], "Robust probe", "s"),
                                          ("mapfree_coarse_to_fine", OKABE["green"], "Map-free", "^")):
        m = d[d["method"] == method].sort_values("delta_r_m")
        series(ax, m["delta_r_m"], m, colour, label, marker)
    # C1 passive check (A12.4 re-run with the corrected gate, its frozen thresholds), docs/results/wpc_a12
    c1 = pd.read_csv(R / "wpc_a12/c1_passive_vs_active.csv")
    c1 = c1[(c1["world"] == "replay") & c1["run"].str.startswith("standoff_")].rename(
        columns={"passive_point": "point", "passive_ci95_lo": "lo", "passive_ci95_hi": "hi",
                 "passive_n_geometries": "n_geometries"})
    c1["delta_r_m"] = c1["run"].str.extract(r"standoff_(\d+)m")[0].astype(float)
    c1 = c1.sort_values("delta_r_m").assign(method="c1_passive", rate="supported_pass")
    series(ax, c1["delta_r_m"], c1, OKABE["purple"], "Receive-side", "D", dashed=True)
    # Fraction of geometries NOT certified against any single-antenna emitter, conservative multi-route
    # certificate (B3), docs/results/wpe: grey dotted line with small markers at the six standoff levels.
    cert = pd.read_csv(R / "wpe/b3_certified_region.csv")
    cert = cert[cert["run"].str.startswith("standoff_")].copy()
    cert["delta_r_m"] = cert["run"].str.extract(r"standoff_(\d+)m")[0].astype(float)
    cert = cert.sort_values("delta_r_m")
    cert["point"] = 1.0 - cert["certified_fraction_cons_m0"]
    ax.plot(cert["delta_r_m"], cert["point"], color="0.5", linestyle=":", linewidth=1.0,
            marker="o", markersize=2.5, label="Not certified")
    for x, text, style in ((phi, r"median $\Delta R_\phi$", ":"), (tau, r"median $\Delta R_\tau$", "-.")):
        ax.axvline(x, color=OKABE["black"], linestyle=style, linewidth=0.8)
        ax.text(x + 0.4, 0.8, text, rotation=90, fontsize=7, va="center")
    ax.set_xlabel(r"Relay standoff $\Delta R$ (m)")
    ax.set_ylabel("Replay SUPPORTED rate")
    ax.set_xlim(2, 31)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, out, "fig_a_standoff_sweep")
    unc = cert.assign(method="uncertified_single_emitter", rate="uncertified_fraction", lo=None, hi=None,
                      source="docs/results/wpe/b3_certified_region.csv")
    d = pd.concat([d.assign(source="docs/results/phase8c/standoff"),
                   c1.assign(source="docs/results/wpc_a12/c1_passive_vs_active.csv"), unc], ignore_index=True)
    d = d[["method", "rate", "delta_r_m", "point", "lo", "hi", "n_geometries", "source"]].assign(
        median_delta_r_phi_m=phi, median_delta_r_tau_m=tau)
    d.to_csv(out / "fig_a_standoff_sweep.csv", index=False)
    return d


def fig_b(out: Path, trials_path: Path, geometries_path: Path) -> pd.DataFrame:
    config = PilotConfig.load("configs/pilot.yaml")
    block = json.loads((R / "phase8c/block45/block45_results.json").read_text())
    base = pd.read_parquet(trials_path)
    boot = Bootstrap(sorted(pd.read_csv(geometries_path)["geometry_id"]), int(config.get("phase8", "heldout",
                                                                                          "bootstrap_draws")))
    spec = {"robust_target": ("imperfect_robust", "target", "supported"),
            "narrow_target": ("imperfect_narrow", "target", "supported"),
            "c2f_target_common": ("mapfree_coarse_to_fine", "target", "supported_common"),
            "robust_replay": ("imperfect_robust", "replay", "supported")}
    rows = []
    for key, (method, world, column) in spec.items():
        e = rate_entry(boot, cell(base, method, world), column)
        if key in block["baseline"]:  # same computation as the block 4-5 analysis
            assert abs(e["point"] - block["baseline"][key]["point"]) < 1e-12, key
        rows.append({"series": key, "power_dbm": 10.0, "point": e["point"], "lo": e["ci95"][0], "hi": e["ci95"][1],
                     "source": "main held-out run, sigma 0.5",
                     "label": "" if key in block["baseline"] else "exploratory, post hoc"})
        for level, power in POWER_LEVELS.items():
            v = block["levels"][level][key]
            rows.append({"series": key, "power_dbm": power, "point": v["point"], "lo": v["ci95"][0],
                         "hi": v["ci95"][1], "source": f"docs/results/phase8c/block45 ({level})", "label": ""})
    d = pd.DataFrame(rows).sort_values(["series", "power_dbm"])
    fig, ax = plt.subplots(figsize=SIZE)
    for key, colour, label, marker, dashed in (
            ("robust_target", OKABE["vermillion"], "Robust, target", "s", False),
            ("narrow_target", OKABE["orange"], "Narrow, target", "D", False),
            ("c2f_target_common", OKABE["green"], "Coarse-to-fine (common), target", "^", False),
            ("robust_replay", OKABE["vermillion"], "Robust, replay", "s", True)):
        m = d[d["series"] == key]
        series(ax, m["power_dbm"], m, colour, label, marker, dashed)
    ax.axvline(10.0, color=OKABE["black"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("Probe power (dBm)")
    ax.set_ylabel("SUPPORTED rate")
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, out, "fig_b_power_sweep")
    d.to_csv(out / "fig_b_power_sweep.csv", index=False)
    return d


def fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:.3f} & [{lo:.3f}, {hi:.3f}]"


def tex_escape(text: str) -> str:
    for a, b in (("§", r"\S"), ("sigma<=0.5", r"$\sigma\le0.5$"), ("sigma=", r"$\sigma=$"),
                 ("sigma 0.5 or 1.0", r"$\sigma=0.5$ or $1.0$"), (", 0.5 (", r", $\sigma=0.5$ ("),
                 (", 1.0 (", r", $\sigma=1.0$ ("), ("<=", r"$\le$"),
                 (">=", r"$\ge$"), ("> 0", r"$>0$"), ("_", r"\_")):
        text = text.replace(a, b)
    return text.replace(r"$\sigma=$0.5", r"$\sigma=0.5$").replace(r"$\sigma=$1.0", r"$\sigma=1.0$")


def table_criteria(out: Path) -> None:
    primary = pd.read_csv(R / "phase8b/criteria_primary.csv")
    like = pd.read_csv(R / "phase8b/criteria_like_for_like.csv")
    label = json.loads((R / "phase8b/heldout_results.json").read_text())["decision"]["label"]
    lines = ["% Held-out criteria (ANALYSIS_PLAN A4.2); confirmatory, from docs/results/phase8b. No new numbers.",
             "% columns: l l r c c c", r"\toprule",
             r"Criterion & Pass if & Point & 95\% CI & Point meets & Interval holds \\", r"\midrule"]
    extra = like[like["criterion"].str.contains("a2_1_like_for_like")]
    for frame in (primary, extra):
        for r in frame.itertuples():
            point = "" if pd.isna(r.point) else fmt(r.point, r.ci95_lo, r.ci95_hi)
            point = point or " & "
            holds = "" if pd.isna(r.interval_criterion_holds) else str(r.interval_criterion_holds).lower()
            name = tex_escape(r.criterion.strip()).replace("a2\\_1\\_like\\_for\\_like", "like-for-like")
            indent = r"\quad " if r.role == "component" else ""
            lines.append(f"{indent}{name} & {tex_escape(str(r.pass_if))} & {point} & "
                         f"{str(r.point_meets).lower()} & {holds} \\\\")
    lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{A4.1 label: {label}}} \\", r"\bottomrule"]
    (out / "table_heldout_criteria.tex").write_text("\n".join(lines) + "\n")


def table_stress(out: Path) -> None:
    bw = json.loads((R / "phase8c/bandwidth_results.json").read_text())["levels"]
    st_t = pd.read_csv(R / "phase8c/standoff/target_by_level.csv")
    st_r = pd.read_csv(R / "phase8c/standoff/replay_by_level.csv")
    d11 = json.loads((R / "phase8c/array_error/array_error_results.json").read_text())
    b45 = json.loads((R / "phase8c/block45/block45_results.json").read_text())["levels"]

    def e(v: dict) -> str:
        return fmt(v["point"], *v["ci95"])

    def st(frame: pd.DataFrame, level: str) -> str:
        r = frame[(frame["method"] == "imperfect_robust") & (frame["level"] == level)].iloc[0]
        return fmt(r["point"], *json.loads(r["ci95"]))

    rows = [("1", "Bandwidth 245.76 MHz", e(bw["bw246"]["target_supported_primary"]), e(bw["bw246"]["replay_supported"]),
             "A5.1(a), (b) matched"),
            ("1", "Bandwidth 61.44 MHz", e(bw["bw61"]["target_supported_primary"]), e(bw["bw61"]["replay_supported"]),
             "A5.1(a) matched; (b) not matched")]
    for m in (3, 6, 10, 15, 20, 30):
        level = f"standoff_{m:02d}m"
        note = "A6.2 at 20 m matched (descriptive)" if m == 20 else "A6.2 2$\\times$2 (pooled): mapped matched (descriptive)"
        rows.append(("2", f"Standoff $\\Delta R={m}$ m", st(st_t, level), st(st_r, level), note))
    rows.append(("3", "Array calibration error (D11)", e(d11["target_supported_primary"]), e(d11["replay_supported"]),
                 "A7.1 replay and target matched"))
    notes = {"probe_power_m20dbm": "A8.1(a) not matched; (b), (c) matched",
             "probe_power_m10dbm": "A8.1(a) not matched (0.646 $<$ 0.95); (b), (c) matched",
             "probe_power_0dbm": "A8.1(b), (c) matched", "probe_power_p20dbm": "A8.1(b), (c) matched",
             "wall_thickness_50mm": "A8.2(a) $\\ge$1 stratum not matched (0.059); $<$1 stratum matched",
             "wall_thickness_200mm": "A8.2(a) $\\ge$1 stratum 0.000 (joint prediction not matched)",
             "rcs_p6db": "A8.2(b) +6 dB matched", "rcs_m6db": "A8.2(b) $-$6 dB matched"}
    names = {"probe_power_m20dbm": "Probe power $-20$ dBm", "probe_power_m10dbm": "Probe power $-10$ dBm",
             "probe_power_0dbm": "Probe power 0 dBm", "probe_power_p20dbm": "Probe power $+20$ dBm",
             "wall_thickness_50mm": "Wall thickness 0.05 m", "wall_thickness_200mm": "Wall thickness 0.20 m",
             "rcs_p6db": "Target RCS $+6$ dB", "rcs_m6db": "Target RCS $-6$ dB"}
    for level, name in names.items():
        v = b45[level]
        rows.append(("4" if level.startswith("probe") else "5", name, e(v["robust_target"]), e(v["robust_replay"]),
                     notes[level]))
    lines = ["% Stress summary: imperfect_robust at sigma 0.5, geometry-bootstrap 95% CI; from docs/results/phase8c.",
             "% Pre-registered outcomes as reported (A5-A8). No new numbers.", "% columns: c l r c r c l",
             r"\toprule",
             r"Block & Level & \multicolumn{2}{c}{Target SUPPORTED} & \multicolumn{2}{c}{Replay SUPPORTED} & Prediction \\",
             r"\midrule"]
    lines += [f"{b} & {n} & {t} & {r} & {p} \\\\" for b, n, t, r, p in rows]
    lines += [r"\midrule"]
    for name, why in (("Relay gain", "S is normalized by $E_D$ (argument, not a result)"),
                      ("Wall material $\\pm$20\\%", "bracketed by the thickness levels"),
                      ("Blocker", "covered by test 15 and Phase 4")):
        lines.append(f"5 & {name} & \\multicolumn{{4}}{{c}}{{NOT RUN}} & {why} \\\\")
    lines.append(r"\bottomrule")
    (out / "table_stress_summary.tex").write_text("\n".join(lines) + "\n")


def table_e2(out: Path) -> None:
    d = pd.read_csv(out / "e2_escalation.csv")
    d = d[d["pairing"] == "matched_records_0_1_4_5"]
    text = {"P0": "P0: 122.88 MHz only", "P1": "P1: 245.76 MHz only", "P2": "P2: escalate on SUPPORTED",
            "P3": f"P3: escalate if $S<s_{{\\mathrm{{esc}}}}={d['s_esc'].dropna().iloc[0]:.3f}$"}
    lines = ["% E2 escalation policy: exploratory, post hoc. imperfect_robust, sigma 0.5, records 0,1,4,5 paired",
             "% across the main run and bw246 (shared seeds: second probe not an independent realization).",
             "% Cost: mean wall-probe spectrum-time per candidate, unit = one 122.88 MHz probe.",
             "% s_esc: 1st percentile of calibration imperfect_robust target S among SUPPORTED rows, pooled over",
             "% sigma (results/phase8a/calibration_trials.parquet); see e2_escalation_README.json.",
             "% columns: l r c r c r r", r"\toprule",
             r"Policy & \multicolumn{2}{c}{Target SUPPORTED} & \multicolumn{2}{c}{Replay SUPPORTED} & "
             r"Cost (target) & Cost (replay) \\", r"\midrule"]
    for r in d.itertuples():
        lines.append(f"{text[r.policy]} & {fmt(r.target_supported, r.target_ci95_lo, r.target_ci95_hi)} & "
                     f"{fmt(r.replay_supported, r.replay_ci95_lo, r.replay_ci95_hi)} & {r.target_cost:.2f} & "
                     f"{r.replay_cost:.2f} \\\\")
    lines.append(r"\bottomrule")
    (out / "table_e2_policy.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    trials_path, geometries_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    a, b = fig_a(out), fig_b(out, trials_path, geometries_path)
    table_criteria(out)
    table_stress(out)
    table_e2(out)
    print(a.round(3).to_string())
    print(b.round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
