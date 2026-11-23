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
public    the outputs that hold no number derived from himalaya or f1-dnf, whose licences do not allow publishing
          them: public/no_himalaya.txt, public/forest.csv, public/per_dataset_points.csv
floor     EF fits that stop at 1 to 5 trees (seven published thin datasets, E2, final): counts per dataset and T - EF with and
          without those cells -> public/ef_floor.txt
diagnostics  capped shares, cap rule for H1, median trees, stopping-set optimism and unseen-level shares without
          himalaya (thin E2) -> public/diagnostics.txt
facts     public/post_facts.md: the rows of publish/post_facts.md that carry no himalaya mark and whose numbers all occur
          in one line of public/no_himalaya.txt (rows of other sources are kept only for datasets and calc rows)

Usage: python -m scripts.es_time_publish manifest | nohim | forest | check [sheet] | public | floor | diagnostics | facts
"""
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd

from scripts.es_time import CAPPED, DATASETS, OUT, PREPARED, ROOT, sha256
from scripts.es_time_analyse import DESCRIPTIVE, HYP, TIE, cells, diff, flagged, fmt, label, per_dataset, pooled, split_halves
from scripts.es_time_explore import EXP, cell_frame

PUBLISH = OUT / "publish"
# Not published: himalaya (licence requires written consent for analyses) and f1-dnf (Ergast terms, non-commercial only).
DROP = ("himalaya", "f1-dnf")
KEPT = "without himalaya and f1-dnf"
SOURCES = {"analysis.txt": OUT / "analysis.txt", "explore.txt": ROOT / "results/es_time_explore.txt",
           "no_himalaya.txt": PUBLISH / "no_himalaya.txt", "no_himalaya_explore.txt": ROOT / "results/es_time_no_himalaya.txt",
           "plots.txt": ROOT / "results/es_time_plots.txt", "public": OUT / "public/no_himalaya.txt",
           "public_floor": OUT / "public/ef_floor.txt", "public_diag": OUT / "public/diagnostics.txt"}
OUTPUTS = ["analysis.txt", "per_dataset.csv", "stopping_set_gaps.csv", "delta.json"]


def fits_cells():
    fits = pd.concat([joblib.load(f) for pop in DATASETS for f in sorted((OUT / pop).glob("*.joblib"))], ignore_index=True)
    return fits, cells(fits)


def sets():
    """The dataset sets of the thin population: all nine, without the two unpublished datasets, and also without the
    low-signal datasets."""
    thin = DATASETS["thin"]
    summary = pd.read_csv(PREPARED / "_summary_thin.csv").set_index("dataset").loc[thin]
    low = list(summary.index[summary.low_signal])
    kept = [d for d in thin if d not in DROP]
    return {"all nine datasets (registered set)": thin, KEPT: kept,
            f"{KEPT} and the low-signal datasets": [d for d in kept if d not in low]}, low


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
    base, nh = table["all nine datasets (registered set)"], table[KEPT]
    lines.append(f"Side by side, registered delta, thin E2: all nine against {KEPT}")
    lines.append("hyp | label, nine | label, seven | sign nine / seven | flag nine | flag seven | changed")
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
                         (KEPT, groups[KEPT])):
        for h, x in rows(w, names, d_reg, d_reg).items():
            forest_rows.append(dict(block=block, hypothesis=h, first=HYP[h][0], second=HYP[h][1], estimate=x["estimate"],
                                    lower=x["lo"], upper=x["hi"], label=x["label_reg"], delta=d_reg,
                                    delta_recomputed=delta_for(fits, names), n=len(names)))
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
    sheet = ROOT / sys.argv[2] if len(sys.argv) > 2 else PUBLISH / "post_facts.md"
    for row in sheet.read_text().splitlines():
        cells_ = [c.strip() for c in row.strip().strip("|").split("|")]
        if not row.startswith("|") or cells_[0] in ("id", "") or set(cells_[0]) <= set("-: "):
            continue
        if len(cells_) != 4:
            bad.append(("malformed row", row[:80], "expected 4 cells"))
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


def thin_e2(fits):
    return fits[(fits["pop"] == "thin") & (fits.scheme == "e2")]


def floor():
    """Final EF, EF_T and the EF fit of D_own that stop at 1 to 5 trees, and T - EF with and without those cells."""
    fits, _ = fits_cells()
    f = thin_e2(fits[~fits.dataset.isin(DROP)])
    f = f[f.stage == "final"]
    key = ["dataset", "seed", "cut"]
    best = f.pivot_table(index=key, columns="arm", values="best")
    score = f.pivot_table(index=key, columns="arm", values="ap")
    arms = {"ef": "EF", "ef_t": "EF_T", "d_own_es": "EF fit of D_own"}
    lines = ["EF fits that stop at 1 to 5 trees (best iteration at most 5), thin E2, final fits, the seven published "
             "datasets (exploratory)", f"cells: {len(best)} (seven datasets x 5 seeds x 3 cuts)", ""]
    for arm, name in arms.items():
        hit = best[arm] <= 5
        per = hit.groupby("dataset").agg(["sum", "size"])
        lines.append(f"{name}: {int(hit.sum())} of {len(hit)} fits ({hit.mean():.3f}); per dataset: "
                     + ", ".join(f"{d} {int(r['sum'])} of {int(r['size'])}" for d, r in per.iterrows()))
    hit = best.ef <= 5
    where = hit.groupby(["dataset", "cut"]).sum()
    lines += ["", "EF fits at 1 to 5 trees by dataset and cut (cut 0, 1, 2; of 5 seeds): "
              + "; ".join(f"{d} cut {c} {int(n)}" for (d, c), n in where[where > 0].items())]
    te = score.t - score.ef
    lines += ["", f"T - EF, all cells: {fmt(te.groupby('dataset').mean())}"]
    for text, keep in (("without the cells where the EF fit stops at 1 to 5 trees", ~(best.ef <= 5)),
                       ("without the cells where any of the three EF fits stops at 1 to 5 trees",
                        ~((best.ef <= 5) | (best.ef_t <= 5) | (best.d_own_es <= 5)))):
        per = te[keep].groupby("dataset").mean()
        lines.append(f"T - EF {text}: {fmt(per)}; cells kept {int(keep.sum())} of {len(keep)}")
    PUBLISH.parent.joinpath("public").mkdir(exist_ok=True)
    (OUT / "public/ef_floor.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def diagnostics():
    """The registered diagnostics of the analysis, recomputed on the seven published thin datasets."""
    fits, _ = fits_cells()
    f = thin_e2(fits[~fits.dataset.isin(DROP)])
    es = f[f.arm.isin(["er", "ef", "er_t", "ef_t", "d_own_es"])]
    final = f[f.stage == "final"]
    lines = ["Diagnostics, thin E2, the seven published datasets (the registered diagnostics recomputed)", "",
             "capped share of ES fits (best iteration above 480):",
             es.groupby(["stage", "arm"]).best.apply(lambda b: (b > CAPPED).mean()).round(3).to_string(), "",
             "capped share of final ER_T and EF_T fits per dataset (cap rule for H1: drop a dataset above 0.25):"]
    share = {a: final[final.arm == a].groupby("dataset").best.apply(lambda b: (b > CAPPED).mean()) for a in ("er_t", "ef_t")}
    lines += [f"  {d}: ER_T {share['er_t'][d]:.3f}, EF_T {share['ef_t'][d]:.3f}" for d in share["er_t"].index]
    over = [d for d in share["er_t"].index if max(share["er_t"][d], share["ef_t"][d]) > 0.25]
    h1 = per_dataset(diff(cells(fits[~fits.dataset.isin(DROP)]), "d_t", "dr_t"), "thin", "e2")
    summary = pd.read_csv(PREPARED / "_summary_thin.csv").set_index("dataset").loc[h1.index]
    lag = list(summary.index[summary.c3_ef_max > 0.05])
    lines += [f"datasets above 0.25: {', '.join(over) or 'none'}",
              f"H1 D_T - DR_T without them: {fmt(h1.drop(over))}",
              f"H1 without datasets above 5 % EF stopping-set label lag ({', '.join(lag) or 'none'}): {fmt(h1.drop(lag))}", "",
              "median trees per arm and cut (inner, final):",
              f.groupby(["stage", "arm", "cut"]).trees.median().unstack("cut").round(0).to_string(), "",
              "stopping-set optimism, final ER and EF fits (test minus stopping set), unseen-level shares, best iterations:"]
    fin = final[final.arm.isin(["er", "ef"])].assign(ll_gap=lambda d: d.tgt_ll - d.stop_ll, ap_gap=lambda d: d.ap - d.stop_ap)
    lines.append(fin.groupby("arm")[["ll_gap", "ap_gap", "unseen_stop", "unseen_tgt", "best", "best_ap_iter"]].agg(
        lambda x: x.mean() if x.name not in ("best", "best_ap_iter") else x.median()).round(4).to_string())
    lines += ["", "unseen-level shares per dataset, final ER and EF fits (stopping set, test block):",
              fin.groupby(["dataset", "arm"])[["unseen_stop", "unseen_tgt"]].mean().unstack("arm").round(3).to_string()]
    (OUT / "public").mkdir(exist_ok=True)
    (OUT / "public/diagnostics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parse_rows(path):
    for row in path.read_text().splitlines():
        c = [x.strip() for x in row.strip().strip("|").split("|")]
        if row.startswith("|") and len(c) == 4 and c[0] not in ("id", "") and not set(c[0]) <= set("-: "):
            yield c


def facts():
    """public/post_facts.md, built line by line from public/no_himalaya.txt (so every number has its line by
    construction), plus the dataset rows of the hand-built sheet and its calc row. The nine-dataset diagnostics have no
    public counterpart and are not included."""
    number = r"[+-]?\d+\.\d+"
    rows, section, last = [], "", ""
    for i, text in enumerate((OUT / "public/no_himalaya.txt").read_text().splitlines(), 1):
        if not text.strip() or text.startswith(("Published datasets", "thin:")):
            continue
        if not text.startswith(" "):
            section = text.split(":")[0].strip()
        tokens = []
        m = re.search(rf"({number}) \[({number}), ({number})\]", text)
        if m:
            tokens += list(m.groups())
        for pat in (r"half-width \d\.\d+", r"\(n \d+\)", r"ahead \d+ / behind \d+( / tied \d+)?", r"delta (recomputed.*)?(\d\.\d{5})$",
                    r"registered delta [\d.]+ \(delta.json\)"):
            q = re.search(pat, text)
            if q:
                tokens.append(q.group(0) if pat != r"delta (recomputed.*)?(\d\.\d{5})$" else q.group(2))
        if "leave-one-out flags" in text or text.strip().startswith("label"):
            tokens.append(text.strip().split(": ", 1)[1] if text.strip().startswith("label") else text.strip())
        if tokens:
            if m:
                last = text.strip().split(": ")[0]
            label_ = text.strip().split(": ")[0] if m else (f"{text.strip().split(':')[0]} for {last}" if tokens and ("flags" in text or "label" in text) else "")
            statement = section if not label_ else f"{section}: {label_}"
            rows.append((f"L{i}", statement, "; ".join(tokens), f"public:{i}"))
    for key, name, prefix in (("public_floor", "ef_floor.txt", "F"), ("public_diag", "diagnostics.txt", "D")):
        body = (OUT / "public" / name).read_text().splitlines()
        section, stage = body[0][:90], ""
        for i, text in enumerate(body, 1):
            toks = re.findall(r"[+-]?\d+\.\d+|\d+ of \d+|[A-Za-z0-9-]+ cut \d \d+", text)
            if text.rstrip().endswith(":"):
                section, stage = text.strip().rstrip(":")[:90], ""
            if not toks or text.rstrip().endswith(":"):
                continue
            if text.startswith(("final", "inner")):
                stage = text.split()[0]
            label_ = re.split(r"(?<![\w-])[+-]?\d", text.strip(), maxsplit=1)[0].strip(" :;,")[:60]
            if stage and text.startswith(" "):
                label_ = f"{stage} {label_}"
            rows.append((f"{prefix}{i}", f"{section} / {label_}", "; ".join(toks), f"{key}:{i}"))
    for fid, statement, numbers, source in parse_rows(PUBLISH / "post_facts.md"):
        if fid != "Z2" and (source == "DATA_SOURCES.md" or source.startswith("calc:")):
            rows.append((fid, statement, numbers, source))
    out = ["# es_time: facts for the post, the published datasets", "",
           "Untracked. Row format `| id | statement | numbers | source |`; `public` is `results/es_time/public/no_himalaya.txt`. "
           "Every row is exploratory or descriptive; the registered nine-dataset labels are not quoted. Check: "
           "`python -m scripts.es_time_publish check results/es_time/public/post_facts.md`.", "",
           "| id | statement | numbers | source |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    (OUT / "public/post_facts.md").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} rows")


def public():
    """Everything below is computed from the seven thin datasets without himalaya and f1-dnf and the five dense ones. The
    registered delta is not written (it was computed on all nine datasets); delta is recomputed on the seven, and the
    nine-dataset block and the side-by-side table of `nohim` are not written."""
    fits, _ = fits_cells()
    fits = fits[~fits.dataset.isin(DROP)]
    w = cells(fits)
    groups, low = sets()
    groups.pop("all nine datasets (registered set)")
    low = [d for d in low if d not in DROP]
    d7 = delta_for(fits, groups[KEPT])
    lines = ["Published datasets, without himalaya and f1-dnf (exploratory; the registered estimators, 98.75 % t-interval; "
             "nothing here is a verdict)",
             f"thin: {len(groups[KEPT])} datasets, dense: {fits[fits['pop'] == 'dense'].dataset.nunique()} datasets; "
             f"low-signal datasets: {', '.join(low)}", ""]
    forest_rows, points = [], []
    for name, names in groups.items():
        d_new = delta_for(fits, names)
        r = rows(w, names, d_new, d_new)
        lines.append(f"{name} (n {len(names)}), thin E2: delta recomputed by the A/A procedure on these datasets {d_new:.5f}")
        for h, x in r.items():
            lines += [f" {h} {HYP[h][0].upper()} - {HYP[h][1].upper()}: {x['estimate']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}], "
                      f"half-width {x['half']:.4f}; ahead {x['ahead']} / behind {x['behind']} / tied {x['tied']}",
                      f"    leave-one-out flags (recomputed delta): {', '.join(x['flag_new']) or 'none'}",
                      f"    label (recomputed delta): {x['label_new']}"]
        lines.append("")
        if name == KEPT:
            for h, x in r.items():
                forest_rows.append(dict(hypothesis=h, first=HYP[h][0], second=HYP[h][1], estimate=x["estimate"], lower=x["lo"],
                                        upper=x["hi"], label=x["label_new"], delta_recomputed=d_new, n=len(names)))
                points += [dict(hypothesis=h, dataset=d, value=v) for d, v in x["per"].items()]
    names = groups[KEPT]
    lines.append("Descriptive differences, thin E2 (no label):")
    lines += [f"  {n}: {fmt(per_dataset(diff(w, a, b), 'thin', 'e2').loc[names])}" for n, (a, b) in DESCRIPTIVE.items()]
    for pop, scheme, title in (("thin", "e1", "thin E1"), ("dense", "e2", "dense E2")):
        keep = names if pop == "thin" else [d for d in DATASETS["dense"] if d not in DROP]
        lines += ["", title]
        for h, (a, b) in HYP.items():
            per = per_dataset(diff(w, a, b), pop, scheme).loc[keep]
            lines.append(f"  {h} {a.upper()} - {b.upper()}: {fmt(per)}; ahead {int((per >= TIE).sum())} / behind "
                         f"{int((per <= -TIE).sum())}; flag: {', '.join(flagged(per, d7)) or 'none'}")
    rows_ = [r for f in sorted((EXP / "curve").glob("*.joblib")) for r in joblib.load(f) if r["dataset"] not in DROP]
    d = cell_frame(rows_, None)
    for pop in ("thin", "dense"):
        x = d.xs(pop, level="pop")
        lines += ["", f"Curve at T's pick, {pop} E2 ({x.index.get_level_values('dataset').nunique()} datasets): t = T's drawn count, "
                      "er = ER_T count, ef = EF_T count, val = count that maximises the mean AP of the three inner validation blocks"]
        for a, b in (("val", "t"), ("val", "er"), ("t", "er"), ("er", "ef"), ("val", "ef")):
            per = (x[a] - x[b]).groupby("dataset").mean()
            lines.append(f"  {a} - {b}: {fmt(per)}; ahead {int((per >= TIE).sum())} / behind {int((per <= -TIE).sum())}")
    dr = pd.concat([joblib.load(f) for f in sorted((EXP / "dr").glob("*.joblib"))], ignore_index=True)
    dr = dr[~dr.dataset.isin(DROP)]
    key = ["pop", "scheme", "dataset", "seed", "cut", "block", "cand"]
    inner = fits[fits.stage == "inner"].pivot_table(index=key, columns="arm", values="ap")
    m = inner.join(dr.set_index(key).ap.rename("dr"))
    assert m.dr.notna().all()
    lines += ["", "Inner fits, same candidate and block in both arms (refit = plain fit on all rows with the count of the ES fit)"]
    for pop, scheme in (("thin", "e2"), ("thin", "e1"), ("dense", "e2")):
        x = m.xs((pop, scheme), level=("pop", "scheme"))
        lines.append(f" {pop} {scheme} ({x.index.get_level_values('dataset').nunique()} datasets)")
        for n, (a, b) in {"d_own - dr": ("d_own", "dr"), "d_own - ef": ("d_own", "ef"), "dr - er": ("dr", "er"),
                          "ef - er": ("ef", "er")}.items():
            per = (x[a] - x[b]).groupby("dataset").mean()
            lines.append(f"  {n}: {fmt(per)}; ahead {int((per >= TIE).sum())} / behind {int((per <= -TIE).sum())}")
    target = OUT / "public"
    target.mkdir(exist_ok=True)
    (target / "no_himalaya.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(forest_rows).round(6).to_csv(target / "forest.csv", index=False)
    pd.DataFrame(points).round(6).to_csv(target / "per_dataset_points.csv", index=False)
    text = "\n".join(lines)
    assert "himalaya" not in text.replace("without himalaya", "") and "f1-dnf" not in text.replace("and f1-dnf", "")
    assert "f1-dnf" not in open(target / "per_dataset_points.csv").read()
    assert "0.00623" not in text and "0.006226" not in open(target / "forest.csv").read()
    print("\n".join(lines))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    {"manifest": manifest, "nohim": nohim, "forest": forest, "check": check, "public": public, "floor": floor, "diagnostics": diagnostics, "facts": facts}.get(command, lambda: print(__doc__))()
