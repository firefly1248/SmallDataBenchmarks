"""es_time_publish: exploratory material for publishing the es_time results. Plan: early_stopping_time_plan.md.

Nothing here is registered. The registered output is `es_time_analyse analyse`; this script reads the stored fits and
`delta.json` and never changes them.

manifest  sha256 of analysis.txt, per_dataset.csv, stopping_set_gaps.csv and delta.json -> outputs_manifest.json
nohim     H1 to H4 (thin E2) without himalaya, and without himalaya and the low-signal datasets, with the label
          under the registered delta and under delta recomputed by the plan's A/A procedure on the same datasets
          -> publish/no_himalaya.txt
forest    publish/forest.csv (estimate, bounds, label, delta per hypothesis; all nine datasets and without himalaya)
          and publish/per_dataset_points.csv
check     every number of publish/post_facts.md against the line of its source file

Usage: python -m scripts.es_time_publish manifest | nohim | forest | check
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from scripts.es_time import DATASETS, OUT, PREPARED, ROOT, sha256
from scripts.es_time_analyse import HYP, TIE, cells, diff, flagged, label, per_dataset, pooled, split_halves

PUBLISH = OUT / "publish"
SOURCES = {"analysis.txt": OUT / "analysis.txt", "explore.txt": ROOT / "results/es_time_explore.txt",
           "no_himalaya.txt": PUBLISH / "no_himalaya.txt", "no_himalaya_explore.txt": ROOT / "results/es_time_no_himalaya.txt",
           "plots.txt": ROOT / "results/es_time_plots.txt"}
OUTPUTS = ["analysis.txt", "per_dataset.csv", "stopping_set_gaps.csv", "delta.json"]


def fits_cells():
    fits = pd.concat([joblib.load(f) for pop in DATASETS for f in sorted((OUT / pop).glob("*.joblib"))], ignore_index=True)
    return fits, cells(fits)


def sets():
    """The dataset sets of the thin population: all nine, without himalaya, without himalaya and low-signal."""
    thin = DATASETS["thin"]
    summary = pd.read_csv(PREPARED / "_summary_thin.csv").set_index("dataset").loc[thin]
    low = list(summary.index[summary.low_signal])
    no_h = [d for d in thin if d != "himalaya"]
    return {"all nine datasets (registered set)": thin, "without himalaya": no_h,
            "without himalaya and the low-signal datasets": [d for d in no_h if d not in low]}, low


def delta_for(fits, names):
    """The plan's A/A delta (root mean square of the 15 half-widths of T's seed-pair contrasts) on `names`."""
    t = fits[(fits["pop"] == "thin") & (fits.scheme == "e2") & (fits.stage == "final") & (fits.arm == "t")]
    s = t.groupby(["dataset", "seed"]).ap.mean().unstack("seed").loc[names]
    return float(np.sqrt(np.mean(np.square(split_halves(s)))))


def rows(w, names, d_reg, d_new):
    out = {}
    for h, (a, b) in HYP.items():
        per = per_dataset(diff(w, a, b), "thin", "e2").loc[names]
        m, lo, hi = pooled(per)
        out[h] = dict(estimate=m, lo=lo, hi=hi, half=(hi - lo) / 2, ahead=int((per >= TIE).sum()),
                      behind=int((per <= -TIE).sum()), tied=int(((per > -TIE) & (per < TIE)).sum()),
                      flag_reg=flagged(per, d_reg), flag_new=flagged(per, d_new),
                      label_reg=label(lo, hi, d_reg), label_new=label(lo, hi, d_new), per=per)
    return out


def manifest():
    json.dump({n: sha256(OUT / n) for n in OUTPUTS}, open(OUT / "outputs_manifest.json", "w"), indent=1)
    print(open(OUT / "outputs_manifest.json").read())


def nohim():
    fits, w = fits_cells()
    d_reg = json.load(open(OUT / "delta.json"))["delta"]
    groups, low = sets()
    assert abs(delta_for(fits, DATASETS["thin"]) - d_reg) < 1e-12, "delta.json does not reproduce"
    lines = ["Without himalaya, thin E2 (exploratory; registered estimators, 98.75 % t-interval; nothing here is a verdict)",
             f"low-signal datasets: {', '.join(low)}", f"registered delta {d_reg:.5f} (reproduced from the stored T fits)", ""]
    table = {}
    for name, names in groups.items():
        d_new = delta_for(fits, names)
        r = rows(w, names, d_reg, d_new)
        table[name] = r
        lines.append(f"{name} (n {len(names)}): delta recomputed by the A/A procedure on these datasets {d_new:.5f}")
        for h, x in r.items():
            lines += [f" {h} {HYP[h][0].upper()} - {HYP[h][1].upper()}: {x['estimate']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}], "
                      f"half-width {x['half']:.4f}; ahead {x['ahead']} / behind {x['behind']} / tied {x['tied']}",
                      f"    leave-one-out flags: registered delta {', '.join(x['flag_reg']) or 'none'}; recomputed delta "
                      f"{', '.join(x['flag_new']) or 'none'}",
                      f"    label (i) registered delta: {x['label_reg']}",
                      f"    label (ii) recomputed delta: {x['label_new']}"]
        lines.append("")
    base, nh = table["all nine datasets (registered set)"], table["without himalaya"]
    lines.append("Side by side, registered delta, thin E2: all nine against without himalaya")
    lines.append("hyp | label, nine | label, eight | sign nine / eight | flag nine | flag eight | changed")
    for h in HYP:
        a, b = base[h], nh[h]
        changed = [k for k, v in (("label", a["label_reg"] != b["label_reg"]),
                                  ("sign", np.sign(a["estimate"]) != np.sign(b["estimate"])),
                                  ("flag", a["flag_reg"] != b["flag_reg"])) if v]
        lines.append(f"{h} | {a['label_reg']} | {b['label_reg']} | {np.sign(a['estimate']):+.0f} / {np.sign(b['estimate']):+.0f} | "
                     f"{', '.join(a['flag_reg']) or 'none'} | {', '.join(b['flag_reg']) or 'none'} | {', '.join(changed) or 'nothing'}")
    PUBLISH.mkdir(exist_ok=True)
    (PUBLISH / "no_himalaya.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def forest():
    fits, w = fits_cells()
    d_reg = json.load(open(OUT / "delta.json"))["delta"]
    groups, _ = sets()
    PUBLISH.mkdir(exist_ok=True)
    forest_rows, points = [], []
    for block, names in (("all nine datasets", groups["all nine datasets (registered set)"]),
                         ("without himalaya", groups["without himalaya"])):
        for h, x in rows(w, names, d_reg, d_reg).items():
            forest_rows.append(dict(block=block, hypothesis=h, first=HYP[h][0], second=HYP[h][1], estimate=x["estimate"],
                                    lower=x["lo"], upper=x["hi"], label=x["label_reg"], delta=d_reg, n=len(names)))
            points += [dict(block=block, hypothesis=h, dataset=d, value=v) for d, v in x["per"].items()]
    pd.DataFrame(forest_rows).round(6).to_csv(PUBLISH / "forest.csv", index=False)
    pd.DataFrame(points).round(6).to_csv(PUBLISH / "per_dataset_points.csv", index=False)
    print(pd.DataFrame(forest_rows).round(4).to_string(index=False))


def check():
    """A row of post_facts.md is `| id | statement | numbers separated by ; | source |`; the source is `file:line`, a
    dataset check `DATA_SOURCES.md`, or `calc:total_hours`. Every number must occur in the cited line."""
    text = {k: v.read_text().splitlines() for k, v in SOURCES.items()}
    text["DATA_SOURCES.md"] = [(OUT / "DATA_SOURCES.md").read_text()]
    bad, n_numbers, n_rows = [], 0, 0
    for row in (PUBLISH / "post_facts.md").read_text().splitlines():
        cells_ = [c.strip() for c in row.strip().strip("|").split("|")]
        if not row.startswith("|") or len(cells_) != 4 or cells_[0] in ("id", "") or set(cells_[0]) <= set("-: "):
            continue
        fid, statement, numbers, source = cells_
        tokens = [t.strip() for t in numbers.split(";") if t.strip()]
        n_rows += 1
        if source.startswith("calc:total_hours"):
            fits = pd.concat([joblib.load(f) for pop in DATASETS for f in sorted((OUT / pop).glob("*.joblib"))])
            have = [f"{fits.sec.sum() / 3600:.2f}"]
            ok = tokens == have
            n_numbers += len(tokens)
            if not ok:
                bad.append((fid, tokens, have))
            continue
        if source == "DATA_SOURCES.md":
            for t in tokens:
                n_numbers += 1
                if t not in text[source][0]:
                    bad.append((fid, t, source))
            continue
        name, _, line = source.partition(":")
        line_text = text[name][int(line) - 1]
        for t in tokens:
            n_numbers += 1
            if t not in line_text:
                bad.append((fid, t, f"{source}: {line_text.strip()[:90]}"))
    print(f"facts sheet: {n_rows} rows, {n_numbers} numbers checked, {len(bad)} not found in the cited line")
    for b in bad:
        print("  MISMATCH", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    {"manifest": manifest, "nohim": nohim, "forest": forest, "check": check}.get(command, lambda: print(__doc__))()
