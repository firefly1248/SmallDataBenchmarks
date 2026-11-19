"""es_time_prep.py: prepare the time-ordered binary datasets of early_stopping_time_plan.md and run
the pre-run checks on them.

Preparation: y = 1 is the minority class of the full prepared dataset. Rows are sorted by time, ties
broken by a seeded random key (seed 0) before any subsampling. Two populations: "thin" (datasets above
10,000 rows: 10,000 rows drawn at random, seed 0, indices sorted) and "dense" (the last 10,000 rows,
kept only when thin keeps under half of the rows). Calendar columns (CALENDAR) are dropped from a
population when its shortest E2 fit window spans under 366 days.

Geometry, in shares of n, every position snapped to the nearest row whose timestamp differs from the
previous row's: test blocks [60, 73.33), [73.33, 86.67), [86.67, 100]; three inner validation blocks of
8 % right before each cut; E2 fits on the 36 % before its block, E1 on all rows before it; the EF
stopping set is the freshest 10 % of the fitting window.

Checks (per dataset and population): 1 sorted by time; 2 minority counts per block and window;
3 share of fit-window labels not yet known at the start of the block the window feeds (at most 5 %);
4 default LightGBM on the E2 final window of each cut, time split vs random split drawn from the
window and the test block only, ROC AUC of the time split below 0.99 in every cut (low_signal: below
0.65 in every cut); 5 Spearman correlation of numeric features with row position, listed above 0.5.
Feature screens (E2 cut 1, descriptive, do not change passes): the ROC AUC of every feature alone for
the label on the test block (label_auc) and for window rows against test-block rows (drift_auc);
values above 0.8 are flagged.

Prepared files are written only for included populations that pass all checks.

Usage: python3 -I es_time_prep.py <data_dir> <repo_datasets_dir> <out_dir>
"""
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import arff
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

CAP = 10_000
CUTS = [60, 60 + 40 / 3, 60 + 80 / 3]
TEST_ENDS = [CUTS[1], CUTS[2], 100]
WINDOW, VAL = 36, 8
STOP = 0.10
INCLUDED = ["hotels", "himalaya", "wildlife", "f1-dnf", "nba", "elec", "nebraska-weather", "nycflights",
            "bank-full"]
F1_IDS = ["driverId", "constructorId", "circuitId"]
CALENDAR = {"hotels": ["arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"],
            "wildlife": ["incident_month"]}
LAG_NOT_MODELLED = {"wildlife": "report filing"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prep(data, repo):
    """Return ({name: (frame with t, y, optional label_known and features, name of y=1, name of y=0)},
    sha256 of every input file by path relative to data_dir and to the datasets dir)."""
    inputs = {"data_dir": {}, "datasets_dir": {}}

    def read_csv(root, rel, **kw):
        inputs["data_dir" if root == data else "datasets_dir"][rel] = sha256(f"{root}/{rel}")
        return pd.read_csv(f"{root}/{rel}", **kw)

    def read_arff(name):
        rel = f"{name}.arff"
        inputs["datasets_dir"][rel] = sha256(f"{repo}/{rel}")
        df = pd.DataFrame(arff.loadarff(f"{repo}/{rel}")[0])
        for c in df.columns[df.dtypes == object]:
            df[c] = df[c].str.decode("utf-8")
        return df

    out = {}

    # TidyTuesday 2020-02-11, hotel bookings (Antonio et al. 2019). Time = arrival date, so every
    # cancellation is known by then. Dropped: fields that change after booking or at check-in
    # (status, assigned room, changes, waiting list, parking, country, special requests) and the
    # arrival year. The calendar columns stay until the window rule (CALENDAR) decides. The agent and
    # company ids are categorical.
    h = read_csv(data, "hotels/hotels.csv")
    h["t"] = pd.to_datetime(h.arrival_date_year.astype(str) + "-" + h.arrival_date_month + "-"
                            + h.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
    h["y"] = h.is_canceled
    for c in ["agent", "company"]:
        h[c] = h[c].astype("Int64").astype(str).where(h[c].notna())
    out["hotels"] = (h.drop(columns=[
        "is_canceled", "reservation_status", "reservation_status_date", "assigned_room_type",
        "booking_changes", "days_in_waiting_list", "required_car_parking_spaces", "country",
        "total_of_special_requests", "arrival_date_year"]), "cancelled", "kept")

    # TidyTuesday 2020-09-22, Himalayan expeditions, one row per expedition, "Unknown" termination
    # dropped. The outcome is known at termination: time = termination date, else base camp date plus
    # the median gap, else the first day of a month of the season. A dated row's label is known at its
    # time; an imputed row's label is known one 90th-percentile gap (base camp to termination, dated
    # rows) later. Features are known at the start.
    e = read_csv(data, "himalaya/expeditions.csv")
    e = e[e.termination_reason != "Unknown"].copy()
    term = pd.to_datetime(e.termination_date, errors="coerce")
    base = pd.to_datetime(e.basecamp_date, errors="coerce")
    gap = (term - base).median()
    lag = pd.Timedelta(days=round((term - base).dt.days.quantile(0.9)))
    season = e.season.map({"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}).fillna(1)
    fallback = pd.to_datetime(e.year.astype(str) + "-" + (season * 3 + 2).astype(int).astype(str) + "-01")
    e["t"] = term.fillna(base + gap).fillna(fallback)
    e["label_known"] = e.t.where(term.notna(), e.t + lag)
    e["y"] = (~e.termination_reason.str.startswith("Success")).astype(int)
    src = np.where(term.notna(), "termination date", np.where(base.notna(), "base camp + median gap", "season"))
    print(f"himalaya time source (median gap {gap}, label lag {lag}):",
          pd.Series(src).value_counts(normalize=True).round(3).to_dict())
    out["himalaya"] = (e.drop(columns=[
        "termination_reason", "highpoint_date", "termination_date", "highpoint_metres", "member_deaths",
        "hired_staff_deaths", "oxygen_used", "expedition_id", "peak_name", "basecamp_date", "year"]),
        "no success", "success")

    # TidyTuesday 2019-01-15, orbital launches (JSR launch log). Only rows with a missing date are dropped.
    la = read_csv(data, "launches/launches.csv")
    la["t"] = pd.to_datetime(la.launch_date, errors="coerce")
    la = la[la.t.notna()].copy()
    la["y"] = (la.category == "F").astype(int)
    out["launches"] = (la.drop(columns=["category", "tag", "JD", "mission", "launch_date", "launch_year"]),
                       "failure", "success")

    # TidyTuesday 2019-07-23, FAA wildlife strikes (4 airlines). Rows without a damage code dropped;
    # every code but N counts as damage, "M?" included. In `precip`, "None" is a level (no
    # precipitation) and only "NA" is missing.
    wpath = "wildlife/wildlife_impacts.csv"
    w = read_csv(data, wpath, low_memory=False)
    precip_before = w.precip.isna()
    w["precip"] = pd.read_csv(f"{data}/{wpath}", usecols=["precip"], keep_default_na=False,
                              na_values=["NA"]).precip
    w = w[w.damage.notna()].copy()
    print(f"wildlife precip missing: {precip_before[w.index].mean():.4f} before, "
          f"{w.precip.isna().mean():.4f} after (prepared rows)")
    w["t"] = pd.to_datetime(w.incident_date)
    w["y"] = (w.damage != "N").astype(int)
    out["wildlife"] = (w.drop(columns=["damage", "cost_repairs_infl_adj", "incident_date", "airport",
                                       "species", "incident_year"]), "any damage", "no damage")

    # TidyTuesday 2021-09-07, Formula 1 (Ergast). positionText in results.csv is a number, or R
    # (retired), N (not classified), D (disqualified), E (excluded), F (failed to qualify), W
    # (withdrew). F and W never started and are dropped. Positive: a status other than "Finished" or
    # "+N Laps".
    r = read_csv(data, "f1/results.csv")
    r = r[~r.positionText.isin(["F", "W"])]
    r = r.merge(read_csv(data, "f1/races.csv")[["raceId", "year", "round", "circuitId", "date"]], on="raceId")
    r = r.merge(read_csv(data, "f1/status.csv"), on="statusId")
    r["t"] = pd.to_datetime(r.date)
    r["y"] = (~(r.status.eq("Finished") | r.status.str.startswith("+"))).astype(int)
    out["f1-dnf"] = (r[["t", "y", "grid", "driverId", "constructorId", "circuitId", "round"]],
                     "did not finish (by status)", "finished")

    # TidyTuesday 2021-05-04, water points: excluded, report_date marks upload campaigns, not process time.

    # FiveThirtyEight NBA Elo. One row per game (the non-copy row, the home team); pre-game columns
    # only. game_location is constant (home).
    n = read_csv(data, "nba/nbaallelo.csv")
    n = n[n._iscopy == 0].copy()
    n["t"] = pd.to_datetime(n.date_game, format="%m/%d/%Y")
    n["y"] = (n.game_result == "L").astype(int)
    out["nba"] = (n[["t", "y", "elo_i", "opp_elo_i", "is_playoffs", "seasongame", "fran_id", "opp_fran",
                     "lg_id"]], "home loss", "home win")

    # streaming-datasets: Electricity (NSW 1996-98) and Nebraska weather (1949-99), row order is time.
    # vicprice, vicdemand and transfer hold a constant fill in the first 37.5 % of rows (an era flag).
    el = read_csv(data, "stream/elec.csv")
    el["t"] = np.arange(len(el))
    el["y"] = el["class"].astype(int)
    out["elec"] = (el.drop(columns=["class", "vicprice", "vicdemand", "transfer"]), "price up", "price down")
    we = read_csv(data, "stream/weather.csv")
    we["t"] = np.arange(len(we))
    we["y"] = we.target.astype(int)
    out["nebraska-weather"] = (we.drop(columns=["target"]), "rain", "no rain")

    # nycflights13 via Rdatasets: arrival 15 minutes or more late. Flights without an arrival delay
    # (cancelled and diverted) are dropped. A flight's outcome is known a day after its scheduled hour
    # at the latest. One year: month and day dropped.
    f = read_csv(data, "nycflights/flights.csv", low_memory=False)
    f = f[f.arr_delay.notna()].copy()
    f["t"] = pd.to_datetime(f.time_hour)
    f["label_known"] = f.t + pd.Timedelta(days=1)
    f["y"] = (f.arr_delay >= 15).astype(int)
    out["nycflights"] = (f[["t", "y", "label_known", "sched_dep_time", "sched_arr_time", "carrier", "origin",
                            "dest", "distance", "hour"]], "arrival 15+ min late", "on time")

    # UCI bank marketing (benchmark ARFF), row order. V9 contact (an era flag: "unknown" in nearly all
    # of the first tenths of the file, nearly none at the end), V10 day, V11 month and V12 call duration
    # (known only after the call) are dropped.
    b = read_arff("bank-marketing-full")
    b.columns = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
                 "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "Class"]
    b["t"] = np.arange(len(b))
    b["y"] = (b.Class == "2").astype(int)
    out["bank-full"] = (b.drop(columns=["Class", "contact", "day", "month", "duration"]), "subscribed",
                        "not subscribed")

    # Excluded sets, kept for their counts only (no features).
    for name in ["ozone-eighthr", "ozone-onehr"]:
        d = read_arff(name)
        d["t"] = np.arange(len(d))
        d["y"] = (d.Class == "2").astype(int)
        out[name] = (d[["t", "y"]], "ozone day", "normal day")
    d = read_arff("eeg-eye-state")
    last = d.iloc[:, -1].astype(str)
    d["t"] = np.arange(len(d))
    d["y"] = (last == last.value_counts().idxmin()).astype(int)
    out["eeg-eye-state"] = (d[["t", "y"]], "minority state", "majority state")
    ch = read_csv(repo, "external/cholera-tz-result.csv", low_memory=False)
    ch["t"] = pd.to_datetime(ch.Date, format="%d-%b-%y")
    ch["y"] = (ch.result == 1).astype(int)
    out["cholera"] = (ch[["t", "y"]], "positive", "negative")
    dn = read_csv(repo, "external/dengue-mexico-deaths.csv", low_memory=False)
    dn["t"] = pd.to_datetime(dn.onset_year.astype(str) + "-" + dn.onset_month.astype(str) + "-01")
    dn["y"] = (dn.death == 1).astype(int)
    out["dengue"] = (dn[["t", "y"]], "death", "survived")
    bt = read_csv(repo, "external/biotech-y36.csv", low_memory=False)
    bt["t"] = pd.to_datetime(bt.index_date)
    bt["y"] = (bt.y36 == 1).astype(int)
    bt["label_known"] = pd.to_datetime(bt.outcome_horizon_36m_end_date, errors="coerce")
    out["biotech"] = (bt[["t", "y", "label_known"]], "y36", "not y36")
    return out, inputs


def populations(df):
    """Sort by t with a seeded random tie order, then take the thin and dense populations."""
    tn = df.t.values.astype("int64")
    df = df.iloc[np.lexsort((np.random.default_rng(0).random(len(df)), tn))].reset_index(drop=True)
    n = len(df)
    if n <= CAP:
        return {"thin": df}
    pops = {"thin": df.iloc[np.sort(np.random.default_rng(0).choice(n, CAP, replace=False))].reset_index(drop=True)}
    if CAP / n < 0.5:
        pops["dense"] = df.iloc[-CAP:].reset_index(drop=True)
    return pops


def snapper(t):
    """Return f(i): the nearest row index i' in [0, n] with t[i'] != t[i'-1] (0 and n included)."""
    bounds = np.r_[0, np.flatnonzero(t[1:] != t[:-1]) + 1, len(t)]

    def snap(i):
        k = np.searchsorted(bounds, i)
        lo, hi = bounds[max(k - 1, 0)], bounds[k]
        return int(lo if i - lo <= hi - i else hi)
    return snap


def geometry(t):
    """Test blocks, inner validation blocks, and per scheme the fit windows (inner and final) per cut."""
    n, snap = len(t), snapper(t)
    P = lambda share: snap(round(share / 100 * n))
    g = {"test": [], "val": [], "e2": [], "e1": []}
    for c, end in zip(CUTS, TEST_ENDS):
        g["test"].append((P(c), P(end)))
        g["val"] += [(P(c - 3 * VAL + j * VAL), P(c - 2 * VAL + j * VAL)) for j in range(3)]
        starts = [c - 3 * VAL, c - 2 * VAL, c - VAL, c]
        g["e2"].append([(P(s - WINDOW), P(s)) for s in starts])
        g["e1"].append([(0, P(s)) for s in starts])
    g["stop_start"] = lambda a, b: snap(b - round(STOP * (b - a)))
    return g


def drop_calendar(name, df):
    """Drop the dataset's calendar columns when its shortest E2 window spans under 366 days."""
    t = df.t.values
    days = min((t[b - 1] - t[a]) / np.timedelta64(1, "D") for cut in geometry(t)["e2"] for a, b in cut)
    dropped = days < 366
    print(f"calendar {name}: shortest E2 window {days:.0f} days, "
          f"{'dropped' if dropped else 'kept'} {CALENDAR[name]}")
    return (df.drop(columns=CALENDAR[name]) if dropped else df), \
        dict(cal_dropped=", ".join(CALENDAR[name]) if dropped else "", e2_min_days=round(float(days)))


def features(df):
    return [c for c in df.columns if c not in ("t", "y", "label_known")]


def categoricals(df):
    return [c for c in features(df) if pd.api.types.is_object_dtype(df[c]) or c in F1_IDS]


def design(df, feats, cats):
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X


def fit_auc(X, target, train, test):
    """ROC AUC on `test` of a default LightGBM (200 trees, learning rate 0.05) fitted on `train`."""
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, n_jobs=8, random_state=0)
    model.fit(X.iloc[train], target[train])
    return roc_auc_score(target[test], model.predict_proba(X.iloc[test])[:, 1])


def screen(X, y, g):
    """label_auc and drift_auc of every feature alone, on E2 cut 1: window 24-60 %, test block 60-73.3 %."""
    (fa, fb), (ta, tb) = g["e2"][0][3], g["test"][0]
    win, test = np.arange(fa, fb), np.arange(ta, tb)
    pool = np.r_[win, test]
    drift = np.r_[np.zeros(len(win), int), np.ones(len(test), int)]
    perm = np.random.default_rng(0).permutation(len(pool))
    half = len(pool) // 2
    return pd.DataFrame([dict(
        feature=f, label_auc=fit_auc(X[[f]], y, win, test),
        drift_auc=fit_auc(X[[f]].iloc[pool], drift, perm[:half], perm[half:])) for f in X.columns])


def run_checks(name, pop, df, pos_name, share_full, rows_full):
    t = df.t.values
    n = len(df)
    y = df.y.values
    g = geometry(t)
    row = dict(dataset=name, pop=pop, rows=rows_full, used=n)
    row["span"] = f"{df.t.iloc[0].date()} .. {df.t.iloc[-1].date()}" if np.issubdtype(t.dtype, np.datetime64) \
        else f"rows {t[0]}..{t[-1]}"
    row["class"] = pos_name
    row["share_full"], row["share_pop"] = round(100 * share_full, 1), round(100 * y.mean(), 1)

    # check 1
    row["c1"] = bool(np.all(t[1:] >= t[:-1]))

    # check 2
    count = lambda a, b: int(y[a:b].sum())
    row["test"] = [count(a, b) for a, b in g["test"]]
    row["val_min"] = min(count(a, b) for a, b in g["val"])
    for scheme in ("e2", "e1"):
        wins = [w for cut in g[scheme] for w in cut]
        row[f"fit_{scheme}"] = min(count(a, b) for a, b in wins)
        row[f"ef_{scheme}"] = min(count(g["stop_start"](a, b), b) for a, b in wins)
        row[f"er_{scheme}"] = min(count(a, b) // 10 for a, b in wins)
    row["c2"] = (min(row["test"]) >= 15 and row["val_min"] >= 10 and min(row["fit_e2"], row["fit_e1"]) >= 30
                 and min(row["ef_e2"], row["ef_e1"], row["er_e2"], row["er_e1"]) >= 5)

    # realized EF stopping-set share of the fit window, over all E2 and E1 windows
    ef_share = [(b - g["stop_start"](a, b)) / (b - a) for scheme in ("e2", "e1") for cut in g[scheme] for a, b in cut]
    row["ef_share_min"], row["ef_share_max"] = round(min(ef_share), 3), round(max(ef_share), 3)

    # check 3, E2 windows: share of rows whose label is known after the start of the block they feed
    if "label_known" in df:
        lk = df.label_known.values
        shares, ef_shares = [], []
        for a, b in (w for cut in g["e2"] for w in cut):
            s = g["stop_start"](a, b)
            shares.append(float((lk[a:b] > t[b]).mean()))
            ef_shares.append(float((lk[s:b] > t[b]).mean()))
        row["c3_max"], row["c3_ef_max"] = max(shares), max(ef_shares)
        row["c3"] = row["c3_max"] <= 0.05
    else:
        row["c3_max"] = row["c3_ef_max"] = np.nan
        row["c3"] = True
    row["lag_note"] = LAG_NOT_MODELLED.get(name, "")

    # check 4 on the E2 final window of every cut, and the feature screens
    feats, cats = features(df), categoricals(df)
    scr = pd.DataFrame(columns=["feature", "label_auc", "drift_auc"])
    row["flags"] = ""
    if feats:
        X = design(df, feats, cats)
        for k, ((fa, fb), (ta, tb)) in enumerate(zip((cut[3] for cut in g["e2"]), g["test"]), 1):
            win, test = np.arange(fa, fb), np.arange(ta, tb)
            row[f"auc_time_c{k}"] = fit_auc(X, y, win, test)
            pool = np.r_[win, test]
            row[f"auc_rand_c{k}"] = float(np.mean([
                fit_auc(X, y, perm[:len(win)], perm[len(win):])
                for perm in (np.random.default_rng(seed).permutation(pool) for seed in range(5))]))
        scr = screen(X, y, g)
        flagged = scr[(scr.label_auc > 0.8) | (scr.drift_auc > 0.8)]
        row["flags"] = "; ".join(f"{r.feature} label {r.label_auc:.2f} drift {r.drift_auc:.2f}"
                                 for r in flagged.itertuples())
        if row["flags"]:
            print(f"flags {name} {pop} (label_auc or drift_auc > 0.8): {row['flags']}")
    else:
        for k in (1, 2, 3):
            row[f"auc_time_c{k}"] = row[f"auc_rand_c{k}"] = np.nan
    row["auc_time"], row["auc_rand"] = row["auc_time_c1"], row["auc_rand_c1"]
    time_aucs = [row[f"auc_time_c{k}"] for k in (1, 2, 3)]
    row["c4"] = not any(a >= 0.99 for a in time_aucs)
    row["low_signal"] = bool(feats) and all(a < 0.65 for a in time_aucs)

    # check 5 (printed for the thin population), descriptive
    row["proxies"] = ""
    if pop == "thin" and feats:
        rho = {c: spearmanr(df[c].astype(float), np.arange(n), nan_policy="omit")[0] for c in feats if c not in cats}
        rho = {c: v for c, v in sorted(rho.items(), key=lambda kv: -abs(kv[1])) if not np.isnan(v)}
        print(f"check 5 {name}: " + ", ".join(f"{c} {v:.2f}" for c, v in rho.items()))
        row["proxies"] = ", ".join(f"{c} {v:.2f}" for c, v in rho.items() if abs(v) > 0.5)

    # persistence, minority share by fifth, ties
    p = y.mean()
    row["persistence"] = round(float((y[1:] == y[:-1]).mean() / (p * p + (1 - p) * (1 - p))), 2)
    row["fifths"] = [round(100 * float(x.mean()), 1) for x in np.array_split(y, 5)]
    row["ties"] = round(float(pd.Series(t).duplicated(keep=False).mean()), 2)

    reasons = []
    if not row["c1"]:
        reasons.append("c1 not sorted")
    if not row["c2"]:
        fam = dict(test=(min(row["test"]), 15), val=(row["val_min"], 10), fit=(min(row["fit_e2"], row["fit_e1"]), 30),
                   ef=(min(row["ef_e2"], row["ef_e1"]), 5), er=(min(row["er_e2"], row["er_e1"]), 5))
        reasons.append("c2 " + ",".join(f"{k} {v}<{m}" for k, (v, m) in fam.items() if v < m))
    if not row["c3"]:
        reasons.append(f"c3 max share {row['c3_max']:.2f} > 0.05")
    if not row["c4"]:
        reasons.append(f"c4 time auc {max(time_aucs):.3f} >= 0.99")
    row["included"] = name in INCLUDED
    row["passes"] = not reasons
    row["reason"] = "; ".join(reasons)
    return row, scr


def main():
    data, repo, out = sys.argv[1:4]
    out = Path(out)
    frames, inputs = prep(data, repo)
    rows, screens, prepared = [], {"thin": [], "dense": []}, {}
    for name, (df, pos_name, neg_name) in frames.items():
        if df.y.mean() > 0.5:
            df = df.assign(y=1 - df.y)
            pos_name = neg_name
        share_full = df.y.mean()
        pops = populations(df)
        if name not in INCLUDED:
            pops = {"thin": pops["thin"]}
        files = {}
        for pop, d in pops.items():
            cal = {}
            if name in CALENDAR:
                d, cal = drop_calendar(name, d)
            if pop == "thin":
                thin = d
            row, scr = run_checks(name, pop, d, pos_name, share_full, len(df))
            rows.append(row | cal)
            if len(scr):
                screens[pop].append(scr.assign(dataset=name))
            if row["included"] and row["passes"]:
                d.to_csv(out / f"{name}_{pop}.csv", index=False)
                files[f"{name}_{pop}.csv"] = sha256(out / f"{name}_{pop}.csv")
        if files:
            json.dump({"categorical": categoricals(thin), "features": features(thin), "class": pos_name,
                       "share": round(float(share_full), 4), "sha256": files},
                      open(out / f"{name}.json", "w"), indent=1)
            prepared |= files
    json.dump({"packages": {m.__name__: m.__version__ for m in (np, pd, lgb, scipy, sklearn)},
               "inputs": inputs, "prepared": prepared}, open(out / "_manifest.json", "w"), indent=1)

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 400, "display.max_colwidth", 70, "display.max_columns", 50)
    for pop in ("thin", "dense"):
        pd.concat(screens[pop])[["dataset", "feature", "label_auc", "drift_auc"]].to_csv(
            out / f"_features_{pop}.csv", index=False)
        s = summary[summary["pop"] == pop].drop(columns="pop")
        s.to_csv(out / f"_summary_{pop}.csv", index=False)
        triple = lambda prefix: s[[f"{prefix}_c{k}" for k in (1, 2, 3)]].round(3).astype(str).agg("/".join, axis=1)
        show = pd.DataFrame({
            "dataset": s.dataset, "rows": s.rows, "used": s.used, "span": s.span,
            "class (full/pop %)": s["class"] + " " + s.share_full.astype(str) + "/" + s.share_pop.astype(str),
            "test min": s.test.map(lambda x: "/".join(map(str, x))), "val": s.val_min,
            "fit": np.minimum(s.fit_e2, s.fit_e1), "ef": np.minimum(s.ef_e2, s.ef_e1),
            "er": np.minimum(s.er_e2, s.er_e1), "ef share": s.ef_share_min.astype(str) + "-" + s.ef_share_max.astype(str),
            "c3 max": s.c3_max.round(3), "c3 ef": s.c3_ef_max.round(3),
            "auc time c1/c2/c3": triple("auc_time"), "auc rand c1/c2/c3": triple("auc_rand"),
            "low": s.low_signal, "persist": s.persistence, "pass": s.passes, "reason": s.reason,
            "check 5 |rho|>0.5": s.proxies})
        print(f"\n=== {pop} ===")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
