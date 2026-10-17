"""Seasonally forced SIS model (C) for weekly influenza positivity in Italy.

Model C:
    di/dt = beta * [R(t) * (1 - i) - 1] * i + eps
    R(t) = R_b + (R_s - R_b) * exp(kappa * (cos(2 * pi * (t - phi) / P) - 1))
    phi = 26 + Delta (weeks after 1 July)

Predicted positivity is clip(THETA * i, 0, 1), fitted by binomial deviance.
The import term eps holds i at its off-season floor.

Pre-2020 seasons are fitted greedily, one after another, on an (R_s, Delta)
grid with the epidemic state carried forward. Held-out seasons (2021 onward)
get a frozen forecast: R_s = pre-2020 median + k * sd (k = -1, 0, +1) and
Delta = pre-2020 mean, with no revision. The seasonal-average baseline (SAB)
is the pooled pre-2020 positivity at each season-week.

Importing this module runs the fit and exposes D, pre_s, post_s, baseline and
results. Running it as a script also prints the report and saves two
figures: the weekly path and the held-out deviance of C against SAB.

Usage: python flu_sis_lean.py [flunet.csv] [out.png] [deviance.png]
"""

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CSV = sys.argv[1] if len(sys.argv) > 1 else "flunet_dataset.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "flu_sis.png"
OUT_DEV = sys.argv[3] if len(sys.argv) > 3 else "flu_sis_deviance.png"

# Data selection
COUNTRY = "Italy"
PRE_LAST = 2019  # last pre-2020 season (starts July 2019)
EXCLUDE = {2005}
PANDEMIC = {2009}
MIN_SPEC = 20  # minimum specimens per week
Y_MAX_OK = 0.9  # weeks with positivity at or above this are dropped
WIN_LO, WIN_HI = 14, 45  # season-week window kept
MIN_WEEKS = 15  # minimum weeks per season

# Model
BETA = 1.4  # recovery rate per week
PHI = 26.0  # forcing peak, weeks after 1 July
KAPPA = 4.0  # sharpness of the seasonal pulse
RB = 0.8  # off-season R
I_PEAK_TARGET = 0.2  # median pre-2020 peak positivity maps to i = 0.2
Y0_FLOOR = 0.003  # lower bound on the seed positivity
FLOOR_Q = 0.10  # shoulder-positivity quantile used for the import floor
STEPS = 4  # RK4 substeps per week
PERIOD = 365.25 / 7.0  # weeks per year
DGRID = np.arange(-12, 9, 1.0)  # timing shifts searched, in weeks


def load_weekly_data(path):
    """Weekly Italian positives and specimens after the quality filters."""
    cols = ["INF_ALL", "SPEC_PROCESSED_NB"]
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    df["ISO_WEEKSTARTDATE"] = pd.to_datetime(df["ISO_WEEKSTARTDATE"])
    df = df[df["COUNTRY_AREA_TERRITORY"] == COUNTRY].dropna(subset=cols)

    w = df.groupby("ISO_WEEKSTARTDATE", as_index=False)[cols].sum()
    w = w.rename(
        columns={
            "ISO_WEEKSTARTDATE": "date",
            "INF_ALL": "P",
            "SPEC_PROCESSED_NB": "spec",
        }
    )
    w = w[w["spec"] >= MIN_SPEC].sort_values("date").reset_index(drop=True)
    plausible = (w["P"] <= w["spec"]) & (w["P"] / w["spec"] < Y_MAX_OK)
    w = w[plausible].reset_index(drop=True)
    w["y"] = w["P"] / w["spec"]

    covid = (w["date"] >= pd.Timestamp("2020-03-01")) & (
        w["date"] < pd.Timestamp("2020-07-01")
    )
    return w[~covid].reset_index(drop=True)


def assign_seasons(w):
    """Add season and season-week; keep the window and usable seasons."""
    iso = w["date"].dt.isocalendar()
    week = iso.week.astype(int).clip(upper=52)
    year = iso.year.astype(int)
    w["season"] = np.where(week >= 27, year, year - 1)
    w["sw"] = ((week - 27) % 52).astype(int)

    w = w[(w["sw"] >= WIN_LO) & (w["sw"] <= WIN_HI)].reset_index(drop=True)
    counts = w.groupby("season").size()
    enough = w["season"].isin(counts[counts >= MIN_WEEKS].index)
    allowed = ~w["season"].isin(EXCLUDE | PANDEMIC)
    return w[enough & allowed].reset_index(drop=True)


def import_floor_y():
    """Lower quantile of smoothed pre-2020 shoulder-week positivity."""
    logs = []
    for s in pre_s:
        y, P, n = D[s]["y"], D[s]["P"], D[s]["n"]
        shoulder = y <= 0.25 * y.max()
        logs += list(np.log((P[shoulder] + 0.5) / (n[shoulder] + 1.0)))
    return float(np.exp(np.quantile(logs, FLOOR_Q)))


weekly = assign_seasons(load_weekly_data(CSV))
seasons = sorted(int(s) for s in weekly["season"].unique())
pre_s = [s for s in seasons if s <= PRE_LAST]
post_s = [s for s in seasons if s > PRE_LAST]

# D[season] holds that season's weekly arrays
D = {}
for s, g in weekly.groupby("season"):
    g = g.sort_values("sw")
    D[int(s)] = dict(
        sw=g["sw"].to_numpy(int),
        y=g["y"].to_numpy(float),
        P=g["P"].to_numpy(float),
        n=g["spec"].to_numpy(float),
        date=pd.DatetimeIndex(g["date"]),
    )

THETA = float(np.median([D[s]["y"].max() for s in pre_s]) / I_PEAK_TARGET)
FLOOR_Y = import_floor_y()
I_FLOOR = FLOOR_Y / THETA

# Seed for the continuous run, from the first two weeks of data
FIRST_DATE = D[seasons[0]]["date"][0]
SEED_Y = D[seasons[0]]["P"][:2].sum() / D[seasons[0]]["n"][:2].sum()
I0 = max(SEED_Y, Y0_FLOOR) / THETA


def season_time(dates):
    """Weeks elapsed since the 1 July that starts each date's season."""
    dts = pd.DatetimeIndex(dates)
    start_year = np.where(dts.month >= 7, dts.year, dts.year - 1)
    july1 = pd.DatetimeIndex([f"{int(y)}-07-01" for y in start_year])
    return np.asarray((dts - july1).days, float) / 7.0


def make_sim(rb):
    """Return an RK4 solver that integrates many (R_s, Delta) pairs at once."""
    eps = BETA * max(1.0 - rb, 0.0) * I_FLOOR

    def sim_cols(t0s, i0, Rs, dph):
        """Weekly i(t) from start times t0s, one column per (R_s, Delta).

        Returns the weekly path and the state one week after the last t0.
        """
        Rs = np.asarray(Rs, float)
        phi = PHI + np.asarray(dph, float)
        i = np.full(Rs.shape, float(i0))
        out = np.empty((len(t0s), len(Rs)))
        h = 1.0 / STEPS

        def bell(t):
            return np.exp(
                KAPPA * (np.cos(2 * np.pi * (t - phi) / PERIOD) - 1.0)
            )

        def rhs(t, i):
            R = rb + (Rs - rb) * bell(t)
            return BETA * (R * (1 - i) - 1) * i + eps

        for k, t0 in enumerate(t0s):
            out[k] = i
            t = float(t0)
            for _ in range(STEPS):
                k1 = rhs(t, i)
                k2 = rhs(t + h / 2, i + h / 2 * k1)
                k3 = rhs(t + h / 2, i + h / 2 * k2)
                k4 = rhs(t + h, i + h * k3)
                i = np.clip(i + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
                t += h
        return out, i

    return sim_cols


def xlogy(a, b):
    return np.where(a > 0, a * np.log(np.maximum(b, 1e-12)), 0.0)


def dev_cols(p, s):
    """Binomial deviance of season s; p is (weeks,) or (weeks, columns)."""
    P, n = D[s]["P"], D[s]["n"]
    y = P / n
    p = np.clip(p, 1e-6, 1 - 1e-6)
    saturated = (xlogy(P, y) + xlogy(n - P, 1 - y)).sum()
    if p.ndim > 1:
        P, n = P[:, None], n[:, None]
    loglik = (P * np.log(p) + (n - P) * np.log1p(-p)).sum(axis=0)
    return 2 * (saturated - loglik)


def obs_idx(s, start):
    """Index of season s's observed weeks on a weekly path from start."""
    return np.asarray((D[s]["date"] - start).days // 7, int)


def baseline_pred(s):
    """SAB: pooled positivity by season-week over the pre-2020 seasons.

    A pre-2020 season is scored leave-one-out: it is left out of its own pool.
    """
    pos, spec = {}, {}
    for t in pre_s:
        if t == s:
            continue
        for sw, P, n in zip(D[t]["sw"], D[t]["P"], D[t]["n"]):
            pos[sw] = pos.get(sw, 0) + P
            spec[sw] = spec.get(sw, 0) + n
    overall = sum(pos.values()) / sum(spec.values())
    return np.array(
        [pos[x] / spec[x] if x in spec else overall for x in D[s]["sw"]]
    )


baseline, baseline_pre = {}, {}
for seasons_, store in ((post_s, baseline), (pre_s, baseline_pre)):
    for s in seasons_:
        pred = baseline_pred(s)
        store[s] = dict(dev=float(dev_cols(pred, s)), pred=pred)


def run_model():
    """Fit the pre-2020 seasons and forecast the held-out seasons."""
    sim = make_sim(RB)
    RS = np.arange(RB + 0.02, 3.001, 0.02)
    nR = len(RS)
    R_grid, D_grid = np.meshgrid(RS, DGRID)
    RC, DC = R_grid.ravel(), D_grid.ravel()

    def advance(state, s, R_gap):
        """Simulate the gap before season s; return (state, gap path)."""
        start, i = state
        n_weeks = (D[s]["date"][0] - start).days // 7
        if n_weeks <= 0:
            return i, None
        dates = pd.date_range(start, periods=n_weeks, freq="7D")
        out, final = sim(season_time(dates), i, [R_gap], [0.0])
        return float(final[0]), (dates, np.clip(THETA * out[:, 0], 0, 1))

    def pack(s, out, j):
        """Weekly path of column j and its values at the observed weeks."""
        first_date = D[s]["date"][0]
        dates = pd.date_range(first_date, D[s]["date"][-1], freq="7D")
        path = np.clip(THETA * out[:, j], 0, 1)
        at_obs = np.clip(THETA * out[obs_idx(s, first_date)][:, j], 0, 1)
        return dates, path, at_obs

    def next_start(s, final_state):
        return D[s]["date"][-1] + pd.Timedelta(weeks=1), final_state

    def step_fit(s, state, history):
        """Grid-fit season s; return its result and the carried state."""
        R_gap = np.median(history) if history else 1.3
        i0, gap = advance(state, s, R_gap)
        first_date = D[s]["date"][0]
        dates = pd.date_range(first_date, D[s]["date"][-1], freq="7D")
        out, final = sim(season_time(dates), i0, RC, DC)

        p = np.clip(THETA * out[obs_idx(s, first_date)], 0, 1)
        dev = dev_cols(p, s)
        j = int(np.argmin(dev))
        dates, path, at_obs = pack(s, out, j)
        on_edge = (j % nR) in (0, nR - 1) or (j // nR) in (0, len(DGRID) - 1)
        result = dict(
            dev=float(dev[j]),
            pred=at_obs,
            R=float(RC[j]),
            D=float(DC[j]),
            edge=bool(on_edge),
            path=(dates, path),
            gap=gap,
        )
        return result, next_start(s, float(final[j]))

    def step_frozen(s, state, R_fixed, delta_fixed):
        """Forecast season s with fixed (R_s, Delta); no fitting."""
        i0, gap = advance(state, s, R_fixed)
        first_date = D[s]["date"][0]
        dates = pd.date_range(first_date, D[s]["date"][-1], freq="7D")
        out, final = sim(season_time(dates), i0, [R_fixed], [delta_fixed])
        dates, path, at_obs = pack(s, out, 0)
        result = dict(
            dev=float(dev_cols(at_obs, s)),
            pred=at_obs,
            R=R_fixed,
            D=delta_fixed,
            path=(dates, path),
            gap=gap,
        )
        return result, next_start(s, float(final[0]))

    # Pre-2020: sequential fit with the state carried forward
    state, history, fit = (FIRST_DATE, I0), [], {}
    for s in pre_s:
        fit[s], state = step_fit(s, state, history)
        history.append(fit[s]["R"])
    R_pre = np.array([fit[s]["R"] for s in pre_s])
    D_pre = np.array([fit[s]["D"] for s in pre_s])
    R_med = float(np.median(R_pre))
    R_sd = float(R_pre.std(ddof=1))
    D_mean = float(D_pre.mean())

    # Held-out: frozen forecasts for k = -1, 0, +1
    forecast = {}
    for k in (-1, 0, 1):
        fstate, results = state, {}
        for s in post_s:
            results[s], fstate = step_frozen(
                s, fstate, R_med + k * R_sd, D_mean
            )
        forecast[k] = results
    return dict(
        C=fit,
        forecast=forecast,
        R_pre=R_pre,
        D_pre=D_pre,
        R_med=R_med,
        R_sd=R_sd,
        D_mean=D_mean,
    )


def pooled_deviance(results, ss):
    """Deviance per week and per 1000 specimens, pooled over seasons ss."""
    dev = sum(results[s]["dev"] for s in ss)
    n_weeks = sum(len(D[s]["y"]) for s in ss)
    n_spec = sum(D[s]["n"].sum() for s in ss)
    return dev / n_weeks, dev / n_spec * 1e3


def shoulder_ratio(preds, ss):
    """Specimen-weighted predicted / observed positivity on shoulder weeks.

    Shoulder weeks have y <= 0.25 * the season maximum; 1 means unbiased.
    """
    pred_sum = obs_sum = 0.0
    for s in ss:
        y, p, n = D[s]["y"], preds[s], D[s]["n"]
        shoulder = y <= 0.25 * y.max()
        pred_sum += (n * p)[shoulder].sum()
        obs_sum += (n * y)[shoulder].sum()
    return float(pred_sum / obs_sum)


def print_report(res):
    fc, C = res["forecast"], res["C"]
    R_pre, D_pre = res["R_pre"], res["D_pre"]

    print(f"pre ({len(pre_s)}): {pre_s}\npost ({len(post_s)}): {post_s}")
    print(
        f"R_b={RB}  kappa={KAPPA:.0f}  THETA={THETA:.3f}  "
        f"import floor y={FLOOR_Y:.4f} (i={I_FLOOR:.5f})  seed i0={I0:.5f}"
    )

    print("\nPooled held-out deviance (lower is better)")
    rows = []
    for name, results in (
        ("C k=-1", fc[-1]),
        ("C k=0", fc[0]),
        ("C k=+1", fc[1]),
        ("SAB", baseline),
    ):
        per_week, per_1000 = pooled_deviance(results, post_s)
        rows.append(
            dict(model=name, dev_per_week=per_week, dev_per_1000_spec=per_1000)
        )
    table = pd.DataFrame(rows).set_index("model")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    c_pre = pooled_deviance(C, pre_s)[1]
    b_pre = pooled_deviance(baseline_pre, pre_s)[1]
    print(
        "\nPre-2020 reference per 1000 specimens (C in-sample, SAB "
        f"leave-one-out; not a held-out comparison): C {c_pre:.1f}, "
        f"SAB {b_pre:.1f}"
    )

    c_ratio = shoulder_ratio({s: fc[0][s]["pred"] for s in post_s}, post_s)
    b_ratio = shoulder_ratio({s: baseline[s]["pred"] for s in post_s}, post_s)
    print(
        "\nShoulder weeks, predicted / observed positivity (1 = unbiased): "
        f"C k=0 {c_ratio:.2f}, SAB {b_ratio:.2f}"
    )

    n_edge = sum(C[s]["edge"] for s in pre_s)
    print(
        f"Frozen parameters: R_s = {res['R_med']:.2f} "
        f"(pre sd {res['R_sd']:.2f}, "
        f"range [{R_pre.min():.2f}, {R_pre.max():.2f}]), "
        f"Delta = {res['D_mean']:+.1f} (pre sd {D_pre.std(ddof=1):.1f}); "
        f"pre fit hit the grid edge in {n_edge}/{len(pre_s)} seasons"
    )


def join_segments(segments):
    """Concatenate (gap path, season path) pairs into one weekly series."""
    dates, values = [], []
    for gap, path in segments:
        if gap is not None:
            dates += list(gap[0])
            values += list(gap[1])
        dates += list(path[0])
        values += list(path[1])
    return pd.DatetimeIndex(dates), np.array(values)


def make_figure(res):
    C, k0 = res["C"], res["forecast"][0]
    pre_dates, pre_vals = join_segments(
        [(C[s]["gap"], C[s]["path"]) for s in pre_s]
    )
    k0_dates, k0_vals = join_segments(
        [(k0[s]["gap"], k0[s]["path"]) for s in post_s]
    )
    # Start the forecast line at the last fitted point
    k0_dates = pre_dates[-1:].append(k0_dates)
    k0_vals = np.concatenate([pre_vals[-1:], k0_vals])

    fig, ax = plt.subplots(figsize=(22, 7))
    ax.plot(
        pre_dates,
        np.maximum(pre_vals, 1e-5),
        color="tab:blue",
        lw=1.2,
        label="C fit (pre-2020, in-sample), full weekly path incl. gaps",
    )
    ax.plot(
        k0_dates,
        np.maximum(k0_vals, 1e-5),
        color="tab:red",
        lw=1.2,
        label=(
            f"C frozen forecast k=0 (R_s={res['R_med']:.2f}, "
            f"Delta={res['D_mean']:+.1f})"
        ),
    )
    for s in seasons:
        ax.scatter(D[s]["date"], D[s]["y"], s=8, color="tab:gray", zorder=3)
    ax.set_ylabel("positivity")
    ax.legend(loc="upper left")
    ax.set_title(
        f"Seasonally forced SIS, variant C (R_b={RB}, kappa={KAPPA:.0f}); "
        "gray = observed, blue = pre-2020 fit, red = held-out forecast"
    )
    fig.tight_layout()
    return fig


def make_deviance_figure(res):
    """Deviance per 1000 specimens, C against SAB, pre-2020 and held-out.

    Bars sit at their season's year, so seasons without data leave a gap.
    Pre-2020: C is fitted in-sample and SAB is leave-one-out.
    Held-out: C is the frozen k=0 forecast and SAB uses all pre-2020 seasons.
    """
    C, k0 = res["C"], res["forecast"][0]

    def per_1000(results, ss):
        return pooled_deviance(results, ss)[1]

    c_year = {s: per_1000(C, [s]) for s in pre_s}
    c_year.update({s: per_1000(k0, [s]) for s in post_s})
    b_year = {s: per_1000(baseline_pre, [s]) for s in pre_s}
    b_year.update({s: per_1000(baseline, [s]) for s in post_s})
    years = sorted(c_year)

    first, last = years[0], years[-1]
    pooled_x = (last + 2.5, last + 4.0)  # pre pooled, held-out pooled
    pooled_c = (per_1000(C, pre_s), per_1000(k0, post_s))
    pooled_b = (per_1000(baseline_pre, pre_s), per_1000(baseline, post_s))

    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 5.5))
    xs = years + list(pooled_x)
    c_vals = [c_year[s] for s in years] + list(pooled_c)
    b_vals = [b_year[s] for s in years] + list(pooled_b)
    ax.bar(np.array(xs) - width / 2, c_vals, width, color="tab:red", label="C")
    ax.bar(
        np.array(xs) + width / 2, b_vals, width, color="tab:blue", label="SAB"
    )

    ax.axvline(PRE_LAST + 1, color="black", lw=1.0)  # pre-2020 | held-out
    ax.axvline(last + 1.75, color="tab:gray", lw=0.8, ls="--")
    ax.set_ylim(0, max(c_vals + b_vals) * 1.12)
    top = ax.get_ylim()[1] * 0.98
    ax.text(
        (first + PRE_LAST) / 2,
        top,
        "pre-2020: C in-sample, SAB leave-one-out",
        ha="center",
        va="top",
    )
    ax.text(
        (PRE_LAST + 1 + last) / 2,
        top,
        "held-out: frozen C, SAB from all pre-2020",
        ha="center",
        va="top",
    )

    ticks = list(range(first, last + 1)) + list(pooled_x)
    names = [str(y) for y in range(first, last + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(names + ["Pre pooled", "Held-out pooled"], rotation=90)
    ax.set_xlim(first - 1, pooled_x[1] + 1)
    ax.set_ylabel("deviance per 1000 specimens (lower is better)")
    ax.set_title("Seasonally forced SIS against SAB: pre-2020 and held-out")
    ax.legend(loc="center left")
    fig.tight_layout()
    return fig


results = run_model()

if __name__ == "__main__":
    print_report(results)
    make_figure(results).savefig(OUT, dpi=130)
    make_deviance_figure(results).savefig(OUT_DEV, dpi=130)
    print(f"\nSaved {OUT} and {OUT_DEV}")
