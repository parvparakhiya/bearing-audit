"""README figures, drawn from the files in reports/results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from . import bearing, catalog, config, dsp, loader

# Colours 1-3 checked for colour-blind safety; healthy is neutral grey.
COLOR = {"IR": "#2a78d6", "B": "#eb6834", "OR": "#1baf7a", "Normal": "#898781"}
MARKER = {"IR": "o", "B": "s", "OR": "^", "Normal": "D"}
NAME = {"IR": "inner race", "B": "ball", "OR": "outer race", "Normal": "healthy"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_COLOR = {"forest": "#2a78d6", "rule_fixed_band": "#eb6834", "rule_sk_band": "#1baf7a"}
MODEL_NAME = {
    "forest": "random forest (all features)",
    "rule_fixed_band": "physics rule, fixed band",
    "rule_sk_band": "physics rule, kurtosis band",
}
PROTO_NAME = {
    "A_window_random": "A  random windows",
    "B_leave_one_load_out": "B  unseen load",
    "C_unseen_bearing": "C  unseen bearing",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK2,
        "lines.linewidth": 1.4,
        "font.size": 9.5,
    }
)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sampling_rate(data_dir: Path, results: Path, out: Path) -> None:
    rate = pd.read_csv(results / "sampling_rate_check.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    rec = catalog.get("Normal_3")
    x, rpm = loader.read_raw(data_dir / rec.filename, rec)
    if rpm is None:
        raise ValueError(f"{rec.filename} stores no rpm; this figure needs the measured shaft speed")
    for fs, col, lab in ((48_000, COLOR["IR"], "read as 48 kHz"), (12_000, COLOR["B"], "read as 12 kHz")):
        f, amp = dsp.amplitude_spectrum(x, fs)
        w = (f > 20) & (f < 40)
        a.semilogy(f[w], amp[w], color=col, lw=1.0, label=lab)
    a.axvline(rpm / 60, color=INK, ls="--", lw=1)
    a.text(
        rpm / 60 - 0.25, a.get_ylim()[1] * 0.5, f"shaft\n{rpm / 60:.2f} Hz", ha="right", va="top", color=INK, fontsize=8
    )
    a.axvline(30.0, color=MUTED, ls=":", lw=1)
    a.text(30.25, a.get_ylim()[1] * 0.5, "30.00 Hz\n= 120 Hz mains / 4", ha="left", va="top", color=INK2, fontsize=8)
    a.set(
        xlabel="frequency (Hz)", ylabel="amplitude", title="Normal_3 (3 HP): the same file, two sampling-rate guesses"
    )
    a.legend(loc="upper left")
    n = rate[rate.group == "Normal"].sort_values("shaft_hz", ascending=False)
    loads = np.arange(len(n))
    b.plot(loads, n.err_12k_hz, "-o", color=COLOR["B"], ms=6, label="read as 12 kHz")
    b.plot(loads, n.err_48k_hz, "-o", color=COLOR["IR"], ms=6, label="read as 48 kHz")
    b.axhspan(-0.3, 0.3, color=GRID, alpha=0.6, lw=0)
    b.set_xticks(loads, [f"{r}\n{s:.2f} Hz" for r, s in zip(n.recording, n.shaft_hz, strict=True)])
    b.set(ylabel="tallest line near shaft speed\nminus rpm/60 (Hz)", title="Healthy files, 0 to 3 HP")
    b.annotate(
        "follows the shaft",
        (3, n.err_48k_hz.iloc[-1]),
        (1.6, 0.35),
        color=COLOR["IR"],
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": COLOR["IR"], "lw": 0.8},
    )
    b.annotate(
        "stuck at 30.0 Hz",
        (3, n.err_12k_hz.iloc[-1]),
        (1.3, 1.05),
        color=COLOR["B"],
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": COLOR["B"], "lw": 0.8},
    )
    fig.tight_layout()
    _save(fig, out / "fig1_sampling_rate.png")


def acquisition_confound(data_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for r in catalog.CORE:
        s = loader.load(r.name, data_dir)
        f, p = welch(s.x, s.fs, nperseg=2048)
        p = p / np.median(p[(f > 1000) & (f < 4000)])
        w = f > 3500
        ax.semilogy(
            f[w],
            p[w],
            color=COLOR["Normal"] if r.fault == "Normal" else COLOR["IR"],
            lw=0.8 if r.fault == "Normal" else 0.5,
            alpha=0.9 if r.fault == "Normal" else 0.35,
        )
    ax.axvspan(config.BANDWIDTH_HZ, 6000, color=GRID, alpha=0.6, lw=0)
    ax.axvline(config.BANDWIDTH_HZ, color=INK, lw=1)
    ax.text(
        config.BANDWIDTH_HZ + 40, 3e-4, "excluded:\nno feature looks\nabove 5 kHz", fontsize=8.5, color=INK, va="top"
    )
    ax.text(3550, 3e-6, "faulted (native 12 kHz): recorder anti-alias roll-off", color=COLOR["IR"], fontsize=8.5)
    ax.text(3550, 1.2e-6 / 4, "healthy (48 kHz, resampled here): flat to 6 kHz", color=INK2, fontsize=8.5)
    ax.set(
        xlim=(3500, 6000),
        xlabel="frequency (Hz)",
        ylabel="PSD, relative to 1-4 kHz level",
        title="The healthy and faulted files went through different anti-alias filters",
    )
    fig.tight_layout()
    _save(fig, out / "fig2_acquisition_confound.png")


def raw_vs_envelope(data_dir: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharex=True)
    for col, name in enumerate(("Normal_0", "IR007_0")):
        s = loader.load(name, data_dir)
        x = dsp.lowpass(s.x, s.fs, config.BANDWIDTH_HZ)
        bpfi = bearing.SKF_6205.frequencies(s.shaft_hz)["BPFI"]
        f, a = dsp.amplitude_spectrum(x, s.fs)
        fe, ae = dsp.envelope_spectrum(x, s.fs, config.ENVELOPE_BAND_HZ)
        color = COLOR["IR"] if name.startswith("IR") else COLOR["Normal"]
        for row, (ff, aa, what) in enumerate(((f, a, "raw spectrum"), (fe, ae, "envelope spectrum, 2-4.9 kHz band"))):
            ax = axes[row, col]
            w = (ff > 2) & (ff < 520)
            ax.plot(ff[w], aa[w], color=color, lw=0.8)
            for k in (1, 2, 3):
                ax.axvline(k * bpfi, color=INK, ls=":", lw=0.9)
            ax.text(
                bpfi + 4, ax.get_ylim()[1] * 0.92, f"BPFI {bpfi:.1f} Hz\nand harmonics", fontsize=8, color=INK, va="top"
            )
            ax.set_title(f"{name}  ({NAME[s.recording.fault]})  {what}")
            if row == 1:
                ax.set_xlabel("frequency (Hz)")
    fig.tight_layout()
    _save(fig, out / "fig3_raw_vs_envelope.png")


def physics(results: Path, out: Path) -> None:
    p = pd.read_csv(results / "physics_check.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    pres = p[p.present]
    lo, hi = 98, 166
    a.plot([lo, hi], [lo, hi], color=AXIS, lw=1, zorder=1)
    for fault in ("IR", "OR"):
        q = pres[pres.fault == fault]
        a.scatter(
            q.predicted_hz,
            q.measured_hz,
            s=34,
            color=COLOR[fault],
            marker=MARKER[fault],
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
            label=f"{NAME[fault]} ({len(q)})",
        )
    a.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="predicted from geometry and rpm (Hz)",
        ylabel="measured envelope peak (Hz)",
        title="Measured vs predicted, where present",
    )
    a.text(100, 160, "BPFI cluster:\n155-162 Hz", fontsize=8, color=COLOR["IR"], va="top")
    a.text(109, 101.5, "BPFO cluster: 103-108 Hz", fontsize=8, color=COLOR["OR"])
    a.legend(loc="lower right")
    rank = {"IR": 0, "OR": 1, "B": 2}
    order = sorted(p.bearing.unique(), key=lambda bb: (rank[p[p.bearing == bb].fault.iloc[0]], bb))
    ypos = {bb: i for i, bb in enumerate(order)}
    for _, r in p.iterrows():
        b.scatter(
            r.prominence,
            ypos[r.bearing],
            s=30,
            color=COLOR[r.fault],
            marker=MARKER[r.fault],
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
            alpha=0.95,
        )
    b.axvline(config.PRESENT_PROMINENCE, color=INK, lw=1, ls="--")
    b.text(
        config.PRESENT_PROMINENCE * 1.08,
        -0.3,
        "present above\n10x local level",
        fontsize=8,
        color=INK,
        va="top",
    )
    b.set_xscale("log")
    b.set_yticks(range(len(order)), order)
    b.invert_yaxis()
    b.set(
        xlabel="envelope prominence at the predicted frequency (x local median)",
        title="Signature prominence, every recording and load",
    )
    fig.tight_layout()
    _save(fig, out / "fig4_physics_check.png")


def split_audit(results: Path, out: Path) -> None:
    t = pd.read_csv(results / "audit_summary.csv")
    t = t[t.task == "type4"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    xs = np.arange(3)
    for model in ("forest", "rule_fixed_band", "rule_sk_band"):
        q = t[t.model == model].set_index("protocol").loc[list(PROTO_NAME)]
        lw = 2.2 if model == "forest" else 1.6
        ax.plot(
            xs,
            q.mean_macro_f1,
            "-o",
            color=MODEL_COLOR[model],
            lw=lw,
            ms=7,
            label=MODEL_NAME[model],
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            zorder=3,
        )
        for x, v in zip(xs, q.mean_macro_f1, strict=True):
            dy = 0.035 if model != "rule_sk_band" else -0.055
            ax.text(float(x), v + dy, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(config.GATE_MIN_MACRO_F1, color=INK, lw=1, ls="--")
    ax.text(
        2.08, config.GATE_MIN_MACRO_F1 + 0.012, "pre-registered\nship gate 0.90", fontsize=8, color=INK, va="bottom"
    )
    ax.set_xticks(xs, list(PROTO_NAME.values()))
    ax.set(
        ylim=(0.4, 1.08),
        xlim=(-0.3, 2.55),
        ylabel="macro-F1, fault type (4 classes)",
        title="The forest's 1.00 is recognition of bearings it has already seen",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, out / "fig5_split_audit.png")


def per_bearing(results: Path, out: Path) -> None:
    pb = pd.read_csv(results / "per_bearing_protocol_C.csv", index_col=0)
    phys = pd.read_csv(results / "physics_check.csv")
    present = phys[phys.core].groupby("bearing").present.mean()
    order = ["IR007", "IR014", "IR021", "OR007@6", "OR021@6", "OR014@6", "B007", "B014", "B021", "Normal"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(order))
    for model, dy in (("forest", -0.12), ("rule_fixed_band", 0.12)):
        ax.scatter(
            pb.loc[order, model],
            y + dy,
            s=46,
            color=MODEL_COLOR[model],
            label=MODEL_NAME[model],
            edgecolor=SURFACE,
            linewidth=1,
            zorder=3,
            marker="o" if model == "forest" else "s",
        )
    ax.axhspan(-0.5, 4.5, color=GRID, alpha=0.35, lw=0)
    ax.text(
        1.02,
        2,
        "signature present\nin every recording",
        fontsize=8,
        color=INK,
        va="center",
        transform=ax.get_yaxis_transform(),
    )
    ax.text(1.02, 7, "no signature", fontsize=8, color=INK, va="center", transform=ax.get_yaxis_transform())
    ax.text(
        1.02,
        9,
        "one healthy bearing:\nnever unseen",
        fontsize=8,
        color=INK2,
        va="center",
        transform=ax.get_yaxis_transform(),
    )
    assert (present.reindex(order[:5]) == 1).all() and (present.reindex(order[5:9]) == 0).all()
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set(
        xlim=(-0.03, 1.03),
        xlabel="share of windows classified correctly (protocol C)",
        title="On bearings never seen in training",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.32), ncol=2)
    fig.tight_layout()
    _save(fig, out / "fig6_per_bearing.png")


def confusion(results: Path, out: Path) -> None:
    audit = json.loads((results / "audit.json").read_text())
    pick = [
        r
        for r in audit
        if r["task"] == "type4" and r["protocol"] == "C_unseen_bearing" and r["model"] in ("forest", "rule_fixed_band")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    for ax, r in zip(axes, pick, strict=True):
        cm = np.array(r["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        labs = [NAME[x] for x in r["labels"]]
        ax.set_xticks(range(4), labs, rotation=20)
        ax.set_yticks(range(4), labs)
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{cmn[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff" if cmn[i, j] > 0.55 else INK,
                )
        ax.set(
            xlabel="predicted",
            ylabel="true" if ax is axes[0] else "",
            title=f"{MODEL_NAME[r['model']]}\nmacro-F1 {r['mean_macro_f1']:.2f}",
        )
    fig.suptitle("Protocol C, row-normalised", x=0.02, ha="left", fontsize=9, color=INK2, y=0.02)
    fig.tight_layout()
    _save(fig, out / "fig7_confusion_C.png")


STATUS = {
    "correct": "#0ca30c",
    "undiagnosed": "#fab219",
    "wrong_type": "#ec835a",
    "missed": "#d03b3b",
    "false_alarm": "#d03b3b",
}
OUTCOME_NAME = {
    "correct": "correct",
    "undiagnosed": "undiagnosed (damage, type unclear)",
    "wrong_type": "wrong fault type",
    "missed": "missed fault",
    "false_alarm": "false alarm",
}
SYSTEM_NAME = {
    "forest": "random forest",
    "rule": "physics rule",
    "forest_abstain": "forest, may abstain",
    "physics_diagnoser": "physics diagnoser",
}


def order_spectrum(data_dir: Path, out: Path) -> None:
    """Envelope spectra of one bearing at four loads. In Hz the lines shift with
    speed; in orders (multiples of shaft speed) they line up.
    """
    ramp = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    fig, axes = plt.subplots(4, 2, figsize=(11, 6.4), sharex="col")
    bpfi = bearing.SKF_6205.multiples()["BPFI"]
    for i, load in enumerate(range(4)):
        s = loader.load(f"IR007_{load}", data_dir)
        x = dsp.lowpass(s.x, s.fs, config.BANDWIDTH_HZ)
        f, a = dsp.envelope_spectrum(x, s.fs, config.ENVELOPE_BAND_HZ)
        for col, (xs, lo, hi) in enumerate(((f, 100, 220), (f / s.shaft_hz, 3.5, 7.5))):
            ax = axes[i, col]
            w = (xs >= lo) & (xs <= hi)
            ax.plot(xs[w], a[w], color=ramp[i], lw=0.9)
            marks = [bpfi * s.shaft_hz + d * s.shaft_hz for d in (-1, 0, 1)] if col == 0 else [bpfi - 1, bpfi, bpfi + 1]
            for m, ls in zip(marks, (":", "--", ":"), strict=True):
                ax.axvline(m, color=INK, lw=0.8, ls=ls)
            ax.set_yticks([])
            ax.text(0.01, 0.8, f"{load} HP, {s.rpm:.0f} rpm", transform=ax.transAxes, fontsize=8, color=INK2)
    axes[0, 0].set_title("IR007 envelope spectrum in Hz: BPFI moves with speed")
    axes[0, 1].set_title("Same spectra in orders (x shaft speed): lines align")
    axes[-1, 0].set_xlabel("frequency (Hz)")
    axes[-1, 1].set_xlabel("order (multiples of shaft speed)")
    axes[0, 1].text(bpfi + 0.05, axes[0, 1].get_ylim()[1] * 0.55, "BPFI 5.415", fontsize=8, color=INK)
    axes[0, 1].text(bpfi + 1.05, axes[0, 1].get_ylim()[1] * 0.55, "+1 sideband", fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, out / "fig8_order_spectrum.png")


def industrial_outcomes(results: Path, out: Path) -> None:
    from .industrial import outcome

    dec = pd.read_csv(results / "industrial_recording_decisions_protocol_C.csv")
    systems = list(SYSTEM_NAME)
    faulty, healthy = dec[dec.fault != "Normal"], dec[dec.fault == "Normal"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [3, 1.3]})
    panels = (
        (
            a,
            faulty,
            ("correct", "undiagnosed", "wrong_type", "missed"),
            f"{len(faulty)} faulty recordings (unseen bearings)",
        ),
        (b, healthy, ("correct", "false_alarm"), f"{len(healthy)} healthy recordings (unseen load)"),
    )
    for ax, part, keys, title in panels:
        for yi, sysname in enumerate(systems):
            outs = [outcome(f, d) for f, d in zip(part.fault, part[sysname], strict=True)]
            left = 0
            for k in keys:
                n = outs.count(k)
                if n:
                    ax.barh(yi, n, left=left, color=STATUS[k], edgecolor=SURFACE, linewidth=2, height=0.62)
                    ax.text(
                        left + n / 2,
                        yi,
                        str(n),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="#ffffff" if k in ("correct", "missed", "false_alarm") else INK,
                    )
                left += n
        ax.set_yticks(range(len(systems)), [SYSTEM_NAME[s] for s in systems] if ax is a else [""] * len(systems))
        ax.invert_yaxis()
        ax.set_xlim(0, len(part))
        ax.set_title(title)
        ax.grid(False)
        ax.set_xlabel("recordings")
    handles = [
        matplotlib.patches.Patch(color=STATUS[k], label=OUTCOME_NAME[k])
        for k in ("correct", "undiagnosed", "wrong_type", "missed")
    ]
    handles.append(matplotlib.patches.Patch(color=STATUS["false_alarm"], label="false alarm (healthy panel)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.suptitle("One decision per recording, protocol C", x=0.01, ha="left", fontsize=11, weight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, out / "fig9_industrial_outcomes.png")


def make_all(data_dir: Path, results: Path, out: Path) -> None:
    data_dir, results, out = Path(data_dir), Path(results), Path(out)
    sampling_rate(data_dir, results, out)
    acquisition_confound(data_dir, out)
    raw_vs_envelope(data_dir, out)
    physics(results, out)
    split_audit(results, out)
    per_bearing(results, out)
    confusion(results, out)
    order_spectrum(data_dir, out)
    if (results / "industrial_summary.json").exists():
        industrial_outcomes(results, out)
