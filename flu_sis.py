"""
Seasonally forced SIS model fitted to weekly influenza positivity,
compared with a seasonal-average baseline.

Model
    di/dt = beta * [R(t) * (1 - i) - 1] * i + eps
    R(t)  = R_b + (R_s - R_b) * exp(kappa * (cos(2*pi*(t - phi)/P) - 1))
    phi   = 26 + Delta  (weeks after 1 July)
    Predicted positivity = clip(THETA * i, 0, 1), fitted by binomial
    deviance.  eps holds the off-season level at i_floor (R_b < 1 only).

Fitting (pre-2020 seasons)
    Seasons are fitted one after another on an (R_s, Delta) grid.  The
    epidemic state is carried over from the previous fitted path.  Gaps
    (summers, excluded seasons) are simulated at the running median R_s
    with Delta = 0.  The fit is greedy and in-sample.

Post-2020 seasons
    Frozen forecast: R_s = pre median + k * sd (k = -1, 0, +1) and
    Delta = pre mean, with the state carried through the gap (NPIs are
    not modelled).  A separate in-sample own-fit chain is descriptive.

Seasonal-average baseline
    Pooled positivity at each season-week over the other pre-2020
    seasons (the scored season is left out when scoring pre-2020).

Plots
    Top:    full weekly path of the fit and forecast, with observations.
    Bottom: per-season deviance per week, model vs baseline.

Usage: python flu_sis.py [flunet.csv] [out.png]
"""

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CSV = sys.argv[1] if len(sys.argv) > 1 else "flunet_dataset.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "flu_sis.png"

# Data selection
COUNTRY = "Italy"
PRE_LAST = 2019  # last pre-2020 season (starts July 2019)
# Season 2021 (Jul 2021 - Jun 2022) is excluded here; remove it from
# EXCLUDE to include it.
EXCLUDE = {2021, 2005}
PANDEMIC = {2009}
MIN_SPEC = 20  # minimum specimens per week
WIN_LO, WIN_HI = 14, 45  # season-week window kept
MIN_WEEKS = 15  # minimum weeks per season
Y_MAX_OK = 0.9  # weeks with positivity >= this are dropped

# Model
BETA = 1.4  # recovery rate (per week)
PHI = 26.0  # forcing peak, weeks after 1 July
KAPPA = 4.0  # sharpness of the seasonal pulse
RB = 0.8  # off-season R
I_PEAK_TARGET = 0.2  # median peak positivity maps to i = 0.2
Y0_FLOOR = 0.003  # lower bound on the seed positivity
FLOOR_Q = 0.10  # shoulder-positivity quantile used for the import floor
STEPS = 4  # RK4 substeps per week
PERIOD = 365.25 / 7.0  # weeks per year
DGRID = np.arange(-12, 9, 1.0)  # timing shifts searched (weeks)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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

    # COVID cut: the 2019 season ends on 2020-03-01
    covid = (w["date"] >= pd.Timestamp("2020-03-01")) & (
        w["date"] < pd.Timestamp("2020-07-01")
    )
    return w[~covid].reset_index(drop=True)


def assign_seasons(w):
    """Add season and season-week, keep the window and usable seasons."""
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

# Observation scale: the median pre-2020 peak positivity maps to
# i = I_PEAK_TARGET.
THETA = float(np.median([D[s]["y"].max() for s in pre_s]) / I_PEAK_TARGET)


def import_floor_y():
    """Lower quantile of smoothed pre-2020 shoulder-week positivity.

    Shoulder weeks are those with y <= 0.25 * the season maximum.
    """
    logs = []
    for s in pre_s:
        y, P, n = D[s]["y"], D[s]["P"], D[s]["n"]
        shoulder = y <= 0.25 * y.max()
        logs += list(np.log((P[shoulder] + 0.5) / (n[shoulder] + 1.0)))
    return float(np.exp(np.quantile(logs, FLOOR_Q)))


FLOOR_Y = import_floor_y()
I_FLOOR = FLOOR_Y / THETA

# Start of the continuous run and its seed from the first two weeks
first = seasons[0]
d0 = D[first]["date"][0]
seed_y = D[first]["P"][:2].sum() / D[first]["n"][:2].sum()
i0_cont = max(seed_y, Y0_FLOOR) / THETA

print(f"pre ({len(pre_s)}): {pre_s}\npost ({len(post_s)}): {post_s}")
print(
    f"R_b={RB}  kappa={KAPPA:.0f}  THETA={THETA:.3f}  "
    f"import floor y={FLOOR_Y:.4f} (i={I_FLOOR:.5f})  "
    f"seed i0={i0_cont:.5f}"
)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def season_time(dates):
    """Weeks elapsed since the 1 July that starts each date's season."""
    dts = pd.DatetimeIndex(dates)
    start_year = np.where(dts.month >= 7, dts.year, dts.year - 1)
    july1 = pd.DatetimeIndex([f"{int(y)}-07-01" for y in start_year])
    return np.asarray((dts - july1).days, float) / 7.0


def make_sim(rb):
    """Return sim_cols, an RK4 solver for many (R_s, Delta) pairs at once."""
    eps = BETA * max(1.0 - rb, 0.0) * I_FLOOR

    def sim_cols(t0s, i0, Rs, dph):
        """Weekly i(t) from start times t0s; one column per (R_s, Delta).

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


# ---------------------------------------------------------------------------
# Deviance and baseline
# ---------------------------------------------------------------------------
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
    """Weekly index of season s's observed weeks on a path from start."""
    return np.asarray((D[s]["date"] - start).days // 7, int)


def baseline_pred(s):
    """Pooled positivity by season-week over the other pre-2020 seasons."""
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


baseline = {}
for s in seasons:
    pred = baseline_pred(s)
    baseline[s] = dict(dev=float(dev_cols(pred, s)), pred=pred)


# ---------------------------------------------------------------------------
# The SIS fit
# ---------------------------------------------------------------------------
def run_model():
    """Fit the pre-2020 seasons and forecast the post-2020 seasons."""
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
        """Weekly path of column j, and its values at the observed weeks."""
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
    state, history, fit = (d0, i0_cont), [], {}
    for s in pre_s:
        fit[s], state = step_fit(s, state, history)
        history.append(fit[s]["R"])
    R_pre = np.array([fit[s]["R"] for s in pre_s])
    D_pre = np.array([fit[s]["D"] for s in pre_s])
    R_med = float(np.median(R_pre))
    R_sd = float(R_pre.std(ddof=1))
    D_mean = float(D_pre.mean())

    # Post-2020: descriptive in-sample own-fit chain
    own_state, own_history, own_fit = state, list(history), {}
    for s in post_s:
        own_fit[s], own_state = step_fit(s, own_state, own_history)
        own_history.append(own_fit[s]["R"])

    # Post-2020: frozen forecasts for k = -1, 0, +1
    forecast = {}
    for k in (-1, 0, 1):
        fstate, results = state, {}
        for s in post_s:
            results[s], fstate = step_frozen(
                s, fstate, R_med + k * R_sd, D_mean
            )
        forecast[k] = results
        if k == 0:
            k0_state = fstate

    # Plot only: continue the k = 0 forecast past the last observed week,
    # cut at its trough (nothing is scored on this part).
    dates = pd.date_range(k0_state[0], periods=30, freq="7D")
    out, _ = sim(season_time(dates), k0_state[1], [R_med], [D_mean])
    values = np.clip(THETA * out[:, 0], 0, 1)
    cut = int(np.argmin(values))
    return dict(
        ext=(dates[: cut + 1], values[: cut + 1]),
        C=fit,
        own=own_fit,
        forecast=forecast,
        R_pre=R_pre,
        D_pre=D_pre,
        R_med=R_med,
        R_sd=R_sd,
        D_mean=D_mean,
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def pooled_deviance(results, ss):
    """Deviance per week and per 1000 specimens, pooled over seasons ss."""
    dev = sum(results[s]["dev"] for s in ss)
    n_weeks = sum(len(D[s]["y"]) for s in ss)
    n_spec = sum(D[s]["n"].sum() for s in ss)
    return dev / n_weeks, dev / n_spec * 1e3


def signed_bias(preds, ss):
    """Peak bias, shoulder bias, RMSE and shoulder predicted/observed.

    Bias is obs - pred (positive = underprediction), specimen-weighted.
    Peak weeks have y >= 0.7 * season max; shoulder weeks y <= 0.25 * max.
    """
    peak_bias, shoulder_bias = [], []
    sq_err = weight = shoulder_pred = shoulder_obs = 0.0
    for s in ss:
        y, p, n = D[s]["y"], preds[s], D[s]["n"]
        top = y.max()
        peak, shoulder = y >= 0.7 * top, y <= 0.25 * top
        peak_bias.append(((y - p)[peak] * n[peak]).sum() / n[peak].sum())
        shoulder_bias.append(
            ((y - p)[shoulder] * n[shoulder]).sum() / max(n[shoulder].sum(), 1)
        )
        sq_err += (n * (y - p) ** 2).sum()
        weight += n.sum()
        shoulder_pred += (n * p)[shoulder].sum()
        shoulder_obs += (n * y)[shoulder].sum()
    return (
        float(np.mean(peak_bias)),
        float(np.mean(shoulder_bias)),
        float(np.sqrt(sq_err / weight)),
        float(shoulder_pred / shoulder_obs),
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------
def print_report(res):
    C, own, fc = res["C"], res["own"], res["forecast"]
    k0 = fc[0]
    R_pre, D_pre = res["R_pre"], res["D_pre"]
    pd.set_option(
        "display.width",
        220,
        "display.max_columns",
        None,
        "display.float_format",
        lambda x: f"{x:.3f}",
    )

    rows = []
    for s in seasons:
        pre = s <= PRE_LAST
        r = C[s] if pre else k0[s]
        rows.append(
            dict(
                season=s,
                period="pre" if pre else "post",
                R_s=r["R"],
                Delta=r["D"],
                edge=r.get("edge", ""),
                own_R=np.nan if pre else own[s]["R"],
                own_D=np.nan if pre else own[s]["D"],
                C_dev_wk=r["dev"] / len(D[s]["y"]),
                baseline_dev_wk=baseline[s]["dev"] / len(D[s]["y"]),
            )
        )
    print(
        "\n=== Per season (post rows: R_s and Delta are the FROZEN k=0 "
        "values; own_R / own_D are the in-sample own-fit, descriptive) ==="
    )
    print(pd.DataFrame(rows).set_index("season"))

    print(
        "\n=== Pooled deviance (lower = better); "
        "C pre in-sample, C post frozen forecast ==="
    )
    models = (
        ("C pre (fit)", C, pre_s),
        ("C post k=-1", fc[-1], post_s),
        ("C post k=0", k0, post_s),
        ("C post k=+1", fc[1], post_s),
        ("baseline pre", baseline, pre_s),
        ("baseline post", baseline, post_s),
    )
    table = []
    for name, results, ss in models:
        per_week, per_1000 = pooled_deviance(results, ss)
        table.append(
            dict(model=name, dev_per_week=per_week, dev_per_1000_spec=per_1000)
        )
    print(pd.DataFrame(table).set_index("model"))

    print(
        "\n=== Signed bias (obs - pred; positive = underprediction); "
        "shoulder_pred_over_obs = predicted/observed on shoulder weeks "
        "(1 = unbiased) ==="
    )
    models = (
        ("C pre", C, pre_s),
        ("C post k=0", k0, post_s),
        ("baseline pre", baseline, pre_s),
        ("baseline post", baseline, post_s),
    )
    table = []
    for name, results, ss in models:
        peak, shoulder, rmse, ratio = signed_bias(
            {s: results[s]["pred"] for s in ss}, ss
        )
        table.append(
            dict(
                model=name,
                peak_bias=peak,
                shoulder_bias=shoulder,
                shoulder_pred_over_obs=ratio,
                rmse=rmse,
            )
        )
    print(pd.DataFrame(table).set_index("model"))

    n_edge = sum(C[s]["edge"] for s in pre_s)
    print(
        f"\nfrozen post parameters: R_s = {res['R_med']:.2f} "
        f"(pre sd {res['R_sd']:.2f}, "
        f"range [{R_pre.min():.2f}, {R_pre.max():.2f}]), "
        f"Delta = {res['D_mean']:+.1f} (pre sd {D_pre.std(ddof=1):.1f}); "
        f"pre edge hits {n_edge}/{len(pre_s)}"
    )
    z_R = ", ".join(
        f"{s}:{(own[s]['R'] - R_pre.mean()) / R_pre.std(ddof=1):+.1f}"
        for s in post_s
    )
    z_D = ", ".join(
        f"{s}:{(own[s]['D'] - D_pre.mean()) / D_pre.std(ddof=1):+.1f}"
        for s in post_s
    )
    print(f"post own-fit z vs pre: R_s {z_R} | Delta {z_D}")
    print("Delta at the grid edge (-12 or +8) is a BOUND, not an estimate.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
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
    C, own, k0 = res["C"], res["own"], res["forecast"][0]
    R_med, D_mean = res["R_med"], res["D_mean"]

    pre_dates, pre_vals = join_segments(
        [(C[s]["gap"], C[s]["path"]) for s in pre_s]
    )
    k0_dates, k0_vals = join_segments(
        [(k0[s]["gap"], k0[s]["path"]) for s in post_s]
    )
    # Start the red line at the last blue point and extend it to the trough
    k0_dates = pre_dates[-1:].append(k0_dates).append(res["ext"][0])
    k0_vals = np.concatenate([pre_vals[-1:], k0_vals, res["ext"][1]])

    fig, (ax_top, ax_bar) = plt.subplots(
        2, 1, figsize=(22, 13), gridspec_kw=dict(height_ratios=[2, 1])
    )

    # Top panel: weekly path
    ax_top.plot(
        pre_dates,
        np.maximum(pre_vals, 1e-5),
        color="tab:blue",
        lw=1.2,
        label="C fit (pre, in-sample), full weekly path incl. gaps",
    )
    ax_top.plot(
        k0_dates,
        np.maximum(k0_vals, 1e-5),
        color="tab:red",
        lw=1.2,
        label=f"C frozen forecast k=0 (R_s={R_med:.2f}, Delta={D_mean:+.1f})",
    )
    for j, s in enumerate(post_s):
        dates, values = own[s]["path"]
        ax_top.plot(
            dates,
            np.maximum(values, 1e-5),
            color="tab:green",
            ls="--",
            lw=1,
            label="post own-fit (in-sample)" if j == 0 else None,
        )
    for s in seasons:
        ax_top.scatter(D[s]["date"], D[s]["y"], s=8, color="tab:gray", zorder=3)
    ax_top.set_ylabel("positivity")
    ax_top.legend(loc="upper left", ncol=2)
    ax_top.set_title(
        f"Continuous SIS, variant C (R_b={RB}, kappa={KAPPA:.0f}); "
        "gray pre, red post; full weekly path with state carried"
    )

    # Bottom panel: deviance per week, model vs baseline
    colours = ["tab:red" if s > PRE_LAST else "tab:blue" for s in seasons]
    x = np.arange(len(seasons))
    model_dev = [
        (C[s]["dev"] if s <= PRE_LAST else k0[s]["dev"]) / len(D[s]["y"])
        for s in seasons
    ]
    base_dev = [baseline[s]["dev"] / len(D[s]["y"]) for s in seasons]
    ax_bar.bar(
        x - 0.2, model_dev, 0.4, color=colours, label="C (post: frozen k=0)"
    )
    ax_bar.bar(
        x + 0.2,
        base_dev,
        0.4,
        color="tab:orange",
        label="seasonal-average baseline",
    )
    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(seasons, rotation=60)
    ax_bar.set_ylabel("deviance / week")
    ax_bar.legend()
    ax_bar.set_title("per-season deviance per week")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
results = run_model()
print_report(results)
make_figure(results).savefig(OUT, dpi=130)
print(f"\nSaved {OUT}")
