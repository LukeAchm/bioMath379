"""
C vs seasonal-average baseline (SAB) on held-out post-2020 seasons.

Imports flu_sis.py (same folder), which runs the model once. Both models are
built from pre-2020 data only and scored on held-out seasons without revision.
  C   = frozen forecast, k = 0 (primary); k = -1, +1 as sensitivity
  SAB = flu_sis.baseline (pooled pre-2020 positivity by season-week)

HYPOTHESIS TEST (Part 1)
  H0: pooled held-out d <= 0 (C no better than SAB); H1: d > 0.
  d_s = dev_SAB - dev_C, binomial deviance per 1000 specimens.
  Test: paired one-sided sign-flip over held-out seasons (min p = 1/2^n).
  Also per season: skill = 1 - dev_C/dev_SAB, peak-week error (pred - obs),
  peak-height ratio (pred/obs).

SUPPORTING CHECKS (Part 2, context only, not part of the test)
  Observed peak week per season against the pre-2020 range; variance-ratio
  permutation test on peak week (post vs pre, one-sided), with and without
  2021; median weekly specimen volume pre vs held-out.

Usage: python sab_benchmark_lean.py [flunet.csv] [out_prefix]
Outputs: <prefix>.png, <prefix>_metrics.csv
"""

import itertools
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "flunet_dataset.csv"
PFX = sys.argv[2] if len(sys.argv) > 2 else "sab_benchmark"
sys.argv = ["flu_sis.py", CSV, f"{PFX}_flu_sis.png"]
import flu_sis as F  # noqa: E402

D, pre_s, post_s = F.D, F.pre_s, F.post_s
MODELS = {
    "C k=0": F.results["forecast"][0],
    "C k=-1": F.results["forecast"][-1],
    "C k=+1": F.results["forecast"][1],
}
RED, GRY, BLU, ORG = "tab:red", "tab:gray", "tab:blue", "tab:orange"
pd.set_option("display.width", 200, "display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# ------------------------------ helpers -------------------------------------
def per_1000(dev, s):
    return float(dev / D[s]["n"].sum() * 1000)


def pooled_obs(s):
    """+-1 season-week pooled observed positivity at each observed week."""
    sw, P, n = D[s]["sw"], D[s]["P"], D[s]["n"]
    m = [np.abs(sw - x) <= 1 for x in sw]
    return np.array([P[i].sum() / n[i].sum() for i in m])


def obs_peak(s):
    """Observed peak season-week and height (linear fill across gaps)."""
    sw = D[s]["sw"]
    grid = np.arange(sw.min(), sw.max() + 1)
    gy = np.interp(grid, sw, pooled_obs(s))
    return float(grid[int(np.argmax(gy))]), float(gy.max())


def pred_peak(pred, s):
    return float(D[s]["sw"][int(np.argmax(pred))]), float(np.max(pred))


def signflip(d):
    d = np.asarray(d, float)
    n, obs = len(d), d.mean()
    S = np.array(list(itertools.product([-1, 1], repeat=n)))
    return float(obs), float(((S * d).mean(1) >= obs - 1e-12).mean())


def season_row(s, res):
    c, b = res[s], F.baseline[s]
    o_sw, o_pk = obs_peak(s)
    c_sw, c_pk = pred_peak(c["pred"], s)
    b_sw, b_pk = pred_peak(b["pred"], s)
    dc, db = per_1000(c["dev"], s), per_1000(b["dev"], s)
    return dict(
        season=s,
        dev_C=dc,
        dev_SAB=db,
        d=db - dc,
        skill=1 - dc / db,
        err_sw_C=c_sw - o_sw,
        err_sw_SAB=b_sw - o_sw,
        pk_ratio_C=c_pk / o_pk,
        pk_ratio_SAB=b_pk / o_pk,
    )


def pooled(res, ss):
    dc = sum(res[s]["dev"] for s in ss)
    db = sum(F.baseline[s]["dev"] for s in ss)
    ns = sum(D[s]["n"].sum() for s in ss)
    return dc / ns * 1000, db / ns * 1000


# ============================ PART 1: the test ===============================
POST = pd.DataFrame([season_row(s, MODELS["C k=0"]) for s in post_s]).set_index(
    "season"
)
print(f"held out: {post_s}   (pre-2020 training: {len(pre_s)} seasons)")
print("\n=== Held-out seasons, C (frozen k=0) vs SAB ===")
print(POST)
dpool_c, dpool_b = pooled(MODELS["C k=0"], post_s)
o, p = signflip(POST["d"])
print(
    f"\nPOOLED held-out deviance/1000: C {dpool_c:.1f} vs SAB {dpool_b:.1f} "
    f"(skill {1 - dpool_c / dpool_b:+.3f}); C wins {(POST['d'] > 0).sum()}/"
    f"{len(POST)}; mean d {o:+.1f}; sign-flip p={p:.4f} "
    f"(min possible {1 / 2 ** len(POST):.4f})"
)
print("\nSensitivity to the frozen R_s (pooled held-out deviance/1000):")
for name, res in MODELS.items():
    dc, db = pooled(res, post_s)
    wins = sum(res[s]["dev"] < F.baseline[s]["dev"] for s in post_s)
    print(f"  {name}: C {dc:.1f} vs SAB {db:.1f}, C wins {wins}/{len(post_s)}")
POST.to_csv(f"{PFX}_metrics.csv")

# ======================== PART 2: supporting checks ==========================
print("\n" + "=" * 70 + "\nSupporting checks (context, not part of the test)")
peaks = pd.Series({s: obs_peak(s)[0] for s in pre_s + post_s})
pre_pk = peaks[pre_s]
print(
    f"Pre-2020 peak week: median {pre_pk.median():.1f}, range "
    f"{pre_pk.min():.0f} to {pre_pk.max():.0f}, sd {pre_pk.std(ddof=1):.2f}"
)
print("Held-out peak week:", {int(s): float(peaks[s]) for s in post_s})


def var_ratio_test(pre, post):
    """One-sided exact permutation test of var(post)/var(pre), post larger."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    obs = post.var(ddof=1) / pre.var(ddof=1)
    a = np.concatenate([pre, post])
    a = a - a.mean()
    n, k = len(a), len(post)
    m = n - k
    x = a[np.array(list(itertools.combinations(range(n), k)))]
    sk, sk2 = x.sum(1), (x**2).sum(1)
    T, T2 = a.sum(), (a**2).sum()
    vr = ((sk2 - sk**2 / k) / (k - 1)) / (
        ((T2 - sk2) - (T - sk) ** 2 / m) / (m - 1)
    )
    return float(obs), float((vr >= obs - 1e-12).mean())


print(
    "\nPeak-week variance ratio, held-out vs pre (one-sided p; single "
    "statistic, no multiplicity correction):"
)
VR = {}
for lab, ss in [
    ("held-out as configured", post_s),
    ("without 2021", [s for s in post_s if s != 2021]),
]:
    if lab == "without 2021" and 2021 not in post_s:
        continue
    if len(ss) < 3:
        continue
    o_vr, p_vr = var_ratio_test(pre_pk.to_numpy(), peaks[ss].to_numpy())
    VR[lab] = (o_vr, p_vr)
    print(
        f"  {lab} (n_post={len(ss)}): sd post {peaks[ss].std(ddof=1):.2f} vs "
        f"pre {pre_pk.std(ddof=1):.2f}, variance ratio {o_vr:.2f}, "
        f"p={p_vr:.4f}"
    )

spec = pd.Series({s: float(D[s]["n"].mean()) for s in pre_s + post_s})
print(
    f"\nMedian weekly specimens: pre {spec[pre_s].median():.0f}, held-out "
    f"{spec[post_s].median():.0f} (x{spec[post_s].median() / spec[pre_s].median():.1f})"
)

# ------------------------------- plot ---------------------------------------
x = np.arange(len(post_s))
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
lo = [min(MODELS[k][s]["dev"] for k in MODELS) for s in post_s]
hi = [max(MODELS[k][s]["dev"] for k in MODELS) for s in post_s]
yc = POST["dev_C"].to_numpy()
errs = np.array(
    [
        [yc[i] - per_1000(lo[i], s), per_1000(hi[i], s) - yc[i]]
        for i, s in enumerate(post_s)
    ]
).T
ax[0].bar(
    x - 0.2, yc, 0.4, yerr=errs, color=RED, label="C k=0 (bars: k=-1..+1)"
)
ax[0].bar(x + 0.2, POST["dev_SAB"], 0.4, color=ORG, label="SAB")
ax[0].set_xticks(x)
ax[0].set_xticklabels(post_s)
ax[0].set_ylabel("deviance / 1000 specimens")
ax[0].legend(fontsize=8)
ax[0].set_title(f"held-out deviance; sign-flip p={p:.3f}", fontsize=9)

ax[1].axhspan(pre_pk.min(), pre_pk.max(), color="0.9", zorder=0)
ax[1].axhline(pre_pk.median(), color="0.5", ls="--", lw=1)
ax[1].scatter(pre_s, pre_pk, color=GRY, s=28, label="pre-2020 observed")
ax[1].scatter(
    post_s, peaks[post_s], color=RED, s=40, zorder=3, label="held-out observed"
)
ax[1].scatter(
    post_s,
    [pred_peak(MODELS["C k=0"][s]["pred"], s)[0] for s in post_s],
    marker="x",
    color=BLU,
    s=45,
    zorder=4,
    label="C predicted",
)
ax[1].scatter(
    post_s,
    [pred_peak(F.baseline[s]["pred"], s)[0] for s in post_s],
    marker="+",
    color=ORG,
    s=60,
    zorder=4,
    label="SAB predicted",
)
ax[1].set_ylabel("peak week (season-week, 0 = ISO wk 27)")
ax[1].legend(fontsize=7)
vt = VR.get("held-out as configured")
ax[1].set_title(
    "peak week vs pre-2020 range"
    + (f"; variance ratio {vt[0]:.1f}, p={vt[1]:.3f}" if vt else ""),
    fontsize=9,
)

ax[2].bar(
    [str(s) for s in spec.index],
    spec.to_numpy(),
    color=[RED if s > F.PRE_LAST else GRY for s in spec.index],
)
ax[2].set_yscale("log")
ax[2].tick_params(axis="x", rotation=90, labelsize=7)
ax[2].set_title("mean weekly specimens (log)", fontsize=9)
fig.suptitle("C (frozen) vs seasonal-average baseline, held-out seasons")
fig.tight_layout()
fig.savefig(f"{PFX}.png", dpi=130)
print(f"\nSaved {PFX}.png, {PFX}_metrics.csv")
