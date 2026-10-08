"""Command line: bearing-audit {fetch,verify,physics,audit,industrial,figures,all}.

Every number in the README comes out of one of these commands and is written
to reports/results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

from . import catalog, config, evaluate, features, loader, verify

DATA = Path("data/raw")
MANIFEST = Path("data/manifest.csv")
RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")

# Tried in order. A download is only kept if its SHA-256 matches the manifest,
# so it doesn't matter which source it came from.
SOURCES = (
    "https://engineering.case.edu/sites/default/files/{record}.mat",
    "https://raw.githubusercontent.com/XiongMeijing/CWRU-1/master/Data/{folder}/{name_url}.mat",
)


def _expected() -> dict[str, str]:
    m = pd.read_csv(MANIFEST)
    return dict(zip(m.file, m.sha256, strict=True))


def cmd_fetch(args: argparse.Namespace) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    expected, failed = _expected(), []
    for r in catalog.ALL:
        dest = DATA / r.filename
        if dest.exists() and loader.sha256(dest) == expected[r.filename]:
            continue
        if args.from_dir:
            src = Path(args.from_dir) / r.filename
            if src.exists():
                shutil.copyfile(src, dest)
        else:
            folder = "Normal" if r.fault == "Normal" else "12k_DE"
            for tpl in SOURCES:
                url = tpl.format(record=r.record, folder=folder, name_url=urllib.parse.quote(r.name))
                try:
                    urllib.request.urlretrieve(url, dest)
                except OSError:
                    continue
                if loader.sha256(dest) == expected[r.filename]:
                    break
        ok = dest.exists() and loader.sha256(dest) == expected[r.filename]
        print(f"{'ok ' if ok else 'BAD'} {r.filename}")
        if not ok:
            failed.append(r.filename)
            dest.unlink(missing_ok=True)
    print(f"{len(catalog.ALL) - len(failed)}/{len(catalog.ALL)} files verified against {MANIFEST}")
    return 1 if failed else 0


def cmd_verify(args: argparse.Namespace) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    man = verify.manifest(DATA)
    if MANIFEST.exists():
        bad = man.merge(pd.read_csv(MANIFEST)[["file", "sha256"]], on="file", suffixes=("", "_expected"))
        bad = bad[bad.sha256 != bad.sha256_expected]
        if len(bad):
            print("checksum mismatch:", ", ".join(bad.file))
            return 1
    else:
        man.to_csv(MANIFEST, index=False)
    rate = verify.sampling_rate_check(DATA)
    rate.to_csv(RESULTS / "sampling_rate_check.csv", index=False)
    summary: dict[str, dict[str, object]] = {}
    for grp, g in rate.groupby("group"):
        summary[grp] = {f"{fs}k": int(g[f"match_{fs}k"].sum()) for fs in (12, 48)}
        summary[grp]["n"] = len(g)
    for grp, g in rate.groupby("group"):
        for fs in (12, 48):
            summary[grp][f"duration_{fs}k_s_range"] = [
                round(g[f"duration_{fs}k_s"].min(), 2),
                round(g[f"duration_{fs}k_s"].max(), 2),
            ]
    conf = verify.acquisition_confound(DATA)
    out = {
        "sampling_rate": summary,
        "lowpass_gain_at_hz": verify.lowpass_response(),
        "acquisition_confound": conf,
        "rpm_from_catalogue": man.loc[man.rpm_source == "catalogue", "file"].tolist(),
    }
    (RESULTS / "data_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def cmd_physics(args: argparse.Namespace) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    phys = verify.physics_check(DATA)
    phys.to_csv(RESULTS / "physics_check.csv", index=False)
    summ = phys.groupby("fault").agg(
        recordings=("present", "size"),
        present=("present", "sum"),
        median_prominence=("prominence", "median"),
        median_prominence_sk_band=("prominence_sk_band", "median"),
    )
    pres = phys[phys.present]
    summ["median_abs_error_pct_when_present"] = pres.groupby("fault").measured_over_predicted.apply(
        lambda s: float((s - 1).abs().median() * 100)
    )
    summ.to_csv(RESULTS / "physics_summary.csv")
    print(summ.round(3).to_string())
    return 0


def _feature_table(reuse: bool = False) -> pd.DataFrame:
    path = RESULTS / "features.csv"
    if reuse and path.exists():
        return pd.read_csv(path)
    df = features.build_table(catalog.CORE, DATA)
    df.to_csv(path, index=False)
    return df


def cmd_audit(args: argparse.Namespace) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = _feature_table(reuse=getattr(args, "reuse_features", False))
    results = evaluate.run_all(df)
    (RESULTS / "audit.json").write_text(json.dumps(evaluate.to_records(results), indent=2))
    table = pd.DataFrame(
        [
            {
                "task": r.task,
                "protocol": r.protocol,
                "model": r.model,
                "folds": r.n_folds,
                "mean_macro_f1": round(r.mean_macro_f1, 3),
                "sd_macro_f1": round(r.sd_macro_f1, 3),
                "pooled_accuracy": round(r.pooled_accuracy, 3),
                **({f"recall_{k}": round(v, 3) for k, v in r.recall.items()} if r.task == "type4" else {}),
            }
            for r in results
        ]
    )
    table.to_csv(RESULTS / "audit_summary.csv", index=False)
    gates = evaluate.gates(results)
    (RESULTS / "gate.json").write_text(
        json.dumps(
            {
                "criterion": f"protocol C, 4-class: mean macro-F1 >= {config.GATE_MIN_MACRO_F1} "
                f"and every class recall >= {config.GATE_MIN_CLASS_RECALL}",
                "candidates": gates,
                "ship": any(g["passed"] for g in gates),
            },
            indent=2,
        )
    )
    preds = evaluate.protocol_c_predictions(df)
    preds.to_csv(RESULTS / "predictions_protocol_C.csv", index=False)
    evaluate.per_bearing(preds).to_csv(RESULTS / "per_bearing_protocol_C.csv")
    # post-hoc: these explain the result, they never feed the gate
    (RESULTS / "ablation_posthoc.json").write_text(json.dumps(evaluate.ablation(df), indent=2))
    (RESULTS / "seed_sweep_posthoc.json").write_text(json.dumps(evaluate.seed_sweep(df), indent=2))
    (RESULTS / "bandwidth_sensitivity_posthoc.json").write_text(
        json.dumps(evaluate.bandwidth_sensitivity(DATA), indent=2)
    )
    print(table.to_string(index=False))
    print("\nship:", any(g["passed"] for g in gates))
    return 0


def cmd_industrial(args: argparse.Namespace) -> int:
    """Industrial evaluation: recording-level, three-state, under protocol C."""
    from . import industrial

    RESULTS.mkdir(parents=True, exist_ok=True)
    df = _feature_table(reuse=True)
    windows, extra = industrial.window_states(df)
    decisions = industrial.recording_decisions(windows)
    windows.to_csv(RESULTS / "industrial_window_states_protocol_C.csv", index=False)
    industrial.signature_extras(DATA).round(3).to_csv(RESULTS / "signature_extras.csv", index=False)
    decisions.to_csv(RESULTS / "industrial_recording_decisions_protocol_C.csv", index=False)
    summary = industrial.summarise(windows, decisions)
    out = {
        "gate_criterion": config.PLANT_GATE,
        "cost_model": config.OUTCOME_COST,
        "systems": summary,
        "ship": any(v["gate"]["passed"] for v in summary.values()),
        **extra,
    }
    (RESULTS / "industrial_summary.json").write_text(json.dumps(out, indent=2))
    rows = [
        {
            "system": k,
            **{
                m: round(v["recording_level"][m], 3)
                for m in ("false_alarm_rate", "detection_rate", "coverage", "diagnosis_precision", "cost_per_recording")
            },
            "gate": v["gate"]["passed"],
        }
        for k, v in summary.items()
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nship:", out["ship"])
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    # Agg is chosen here in the CLI, not in plots.py: setting the backend on import
    # breaks inline plots for anyone using the package in Jupyter.
    import matplotlib

    matplotlib.use("Agg")
    from . import plots

    FIGURES.mkdir(parents=True, exist_ok=True)
    plots.make_all(DATA, RESULTS, FIGURES)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_verify, cmd_physics, cmd_audit, cmd_industrial, cmd_figures):
        if step(args):
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    p = argparse.ArgumentParser(prog="bearing-audit", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the CWRU files and verify checksums")
    f.add_argument("--from-dir", help="copy from a folder you already downloaded into, instead of downloading")
    f.set_defaults(func=cmd_fetch)
    for name, fn, text in (
        ("verify", cmd_verify, "manifest, sampling-rate proof, acquisition confound"),
        ("physics", cmd_physics, "fault-frequency verification per recording"),
        ("audit", cmd_audit, "classification under three split protocols + gate"),
        (
            "industrial",
            cmd_industrial,
            "plant-level evaluation: detection, three-state diagnosis, cost, gate (protocol C)",
        ),
        ("figures", cmd_figures, "render every figure in reports/figures"),
        ("all", cmd_all, "verify, physics, audit, industrial, figures"),
    ):
        sp = sub.add_parser(name, help=text)
        sp.set_defaults(func=fn)
        if name in ("audit", "all"):
            sp.add_argument(
                "--reuse-features",
                action="store_true",
                help="use reports/results/features.csv instead of recomputing from the recordings",
            )
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
