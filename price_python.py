# %% CELL 1: install
!pip -q install numpy pandas scipy statsmodels matplotlib openpyxl timesfm[torch]

# %% CELL 2: CONFIG
import re, sys, numpy as np, pandas as pd, statsmodels.api as sm, matplotlib
from scipy import stats
try:
    import google.colab; IN_COLAB = True
except Exception:
    IN_COLAB = False; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHEET, LOCAL_FILE = "data_historical", "pricing_initial_data_workbook.xlsx"
N, SEED = 50_000, 42
PRICES = [265, 270, 275, 280, 285]
HORIZON = 52
MRP, LEADER_NOW = 280, 260
NET_FACTOR = 0.90 / 1.05                       # exact (GST 5% - verify)

# === FIX 1: OTHER_VAR 47.5 -> 50 ; FIXED_M 403_000 -> 448_000 (per Doc Section 4) ===
OTHER_VAR, FIXED_M = 20 + 20 + 5 + 2.5, 403_000  # packaging+logistics+labour+electricity
RM = {"moth_dal": (505, 70, 90, 110), "besan": (115, 60, 70, 80),
      "oil": (350, 100, 110, 120), "spices": (30, 350, 400, 450)}
RHO_DAL_BESAN = 0.5

# === FIX 2: "regression" case is populated dynamically in Cell 5 from the fitted value ===
MAIN_CASE, E_LO, E_HI = "wide (-1.5, 0.5)", -3.0, -0.5
CASES = {"wide (-1.5, 0.5)": (-1.5, 0.5), "stress (-1.9, 0.5)": (-1.9, 0.5)}
# "regression (auto)" appended after the regression fits.

D_ERR_SD, R_ERR_SD = 0.05, 0.05
NOISE_W_SD = None
REACTION_P, REACTION_SHARE = [0.4, 0.4, 0.2], [0.0, 0.5, 1.0]

CAP_M, MIN_VOL_M, MIN_PROFIT_M = 17_000, 12_350, 650_000
MAX_STEP, MAX_VS_LEADER = 0.05, 1.09

DIWALI_HIST   = ["2023-11-12", "2024-11-01", "2025-10-20"]
DIWALI_FUTURE = ["2026-11-08"]
PEAK_REL, BUILD_REL = [-1, 0, 1], [-5, -4, -3, -2]
DIST_CAP, PROMO_CAP = None, None
CAP_MODE = "block"
USE_TIMESFM = True

# %% CELL 3: load data + auto-detect columns
if IN_COLAB:
    from google.colab import files
    up = files.upload(); path = list(up.keys())[0]
else:
    path = LOCAL_FILE
raw = pd.read_excel(path, sheet_name=SHEET)
raw.columns = [str(c).strip() for c in raw.columns]
print("COLUMNS FOUND:", list(raw.columns))

def find(*pats, exclude=()):
    for p in pats:
        for c in raw.columns:
            if re.fullmatch(p, c, re.I) and not any(re.search(x, c, re.I) for x in exclude): return c
    for p in pats:
        for c in raw.columns:
            if re.search(p, c, re.I) and not any(re.search(x, c, re.I) for x in exclude): return c
    return None

DATE   = find(r"week_?start", r"date", r"week")
PRICE  = find(r"shelf_?price", r"own_?price", r"^price$", exclude=("log",))
LEADER = find(r"leader_?price", r"leader", exclude=("log",))
DIST   = find(r"^distrib\w*$", r"distrib", exclude=("log",))
PROMO  = find(r"^promo\w*$", r"promo", exclude=("log",))
PEAK   = find(r"festival_?peak", r"peak")
BUILD  = find(r"festival_?build\w*", r"build")
logu = find(r"log_?units_?sold", r"log_?units")
cands = [c for c in raw.columns if re.match(r"units", c, re.I)]
UNITS = None
if logu is not None:
    lu = pd.to_numeric(raw[logu], errors="coerce")
    for c in cands:
        u_ = pd.to_numeric(raw[c], errors="coerce"); ok = (u_ > 0) & lu.notna()
        if ok.sum() > 20 and np.allclose(np.log(u_[ok]), lu[ok], atol=1e-3): UNITS = c; break
if UNITS is None and cands: UNITS = cands[0]; print("WARNING: units column not matched; using", UNITS)
MAP = dict(DATE=DATE, UNITS=UNITS, PRICE=PRICE, LEADER=LEADER, DIST=DIST, PROMO=PROMO, PEAK=PEAK, BUILD=BUILD)
print("\nDETECTED MAPPING (check every line):")
for k, v in MAP.items(): print(f"  {k:7s} -> {v}")
if any(v is None for v in MAP.values()):
    sys.exit(f"Could not detect {[k for k,v in MAP.items() if v is None]}.")

df = raw.copy(); df[DATE] = pd.to_datetime(df[DATE], errors="coerce"); df = df.dropna(subset=[DATE]).copy()
cols = [UNITS, PRICE, LEADER, DIST, PROMO, PEAK, BUILD]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=cols).sort_values(DATE).reset_index(drop=True)
print(f"\n{len(df)} weeks: {df[DATE].min().date()} -> {df[DATE].max().date()}  (expect 156)")

def use_log1p(s): return bool((pd.Series(s, dtype=float) <= 0).any())
DIST_L1, PROMO_L1 = use_log1p(df[DIST]), use_log1p(df[PROMO])
def lg(x, l1): return np.log1p(x) if l1 else np.log(x)
if PROMO_L1: print("NOTE: promotion has zeros -> log1p used")
if DIST_L1:  print("NOTE: distribution has zeros -> log1p used")

# %% CELL 4: festival flags
def festival_flags(dates, diwalis):
    dates = pd.DatetimeIndex(dates); peak = np.zeros(len(dates)); build = np.zeros(len(dates))
    for d in pd.to_datetime(diwalis):
        wk0 = d - pd.Timedelta(days=d.weekday()); rel = ((dates - wk0).days // 7)
        peak += np.isin(rel, PEAK_REL); build += np.isin(rel, BUILD_REL)
    return np.clip(peak, 0, 1), np.clip(build, 0, 1)
p_, b_ = festival_flags(df[DATE], DIWALI_HIST)
print("Agreement with your peak dummy    :", round((p_ == df[PEAK]).mean(), 3))
print("Agreement with your buildup dummy :", round((b_ == df[BUILD]).mean(), 3))
fut_dates = pd.date_range(df[DATE].max() + pd.Timedelta(days=7), periods=HORIZON, freq="7D")
peak_f, build_f = festival_flags(fut_dates, DIWALI_FUTURE)
print("Forecast window:", fut_dates[0].date(), "->", fut_dates[-1].date())

# %% CELL 5: regression (coefficients AND covariance feed the simulation)
df["lQ"], df["lP"], df["lL"] = np.log(df[UNITS]), np.log(df[PRICE]), np.log(df[LEADER])
df["lD"], df["lR"] = lg(df[DIST], DIST_L1), lg(df[PROMO], PROMO_L1)
FEATS = ["lP", "lL", "lD", "lR", PEAK, BUILD]
free = sm.OLS(df["lQ"], sm.add_constant(df[FEATS])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
print(free.summary().tables[1], "\nR2:", round(free.rsquared, 3), "| your model: own price -1.12, R2 0.949")
names, mu_all, S_all = list(free.params.index), free.params.values, free.cov_params().values
iE = names.index("lP"); iO = [i for i in range(len(names)) if i != iE]
names_o, mu_o, mu_e = [names[i] for i in iO], mu_all[iO], mu_all[iE]
pos = {n: j for j, n in enumerate(names_o)}
S_oe, S_ee, S_oo = S_all[iO, iE], S_all[iE, iE], S_all[np.ix_(iO, iO)]
Sc = S_oo - np.outer(S_oe, S_oe) / S_ee
Sc = (Sc + Sc.T) / 2; w_, V_ = np.linalg.eigh(Sc); LC = V_ * np.sqrt(np.clip(w_, 0, None))
RESID_SD = float(np.std(free.resid, ddof=len(mu_all)))
NOISE_SD = RESID_SD if NOISE_W_SD is None else NOISE_W_SD
print(f"Weekly noise sd used: {NOISE_SD:.3f} ({'regression residual' if NOISE_W_SD is None else 'config'})")

# === FIX 2: append "regression (auto)" from fitted coefficients ===
reg_mu_e, reg_se_e = float(mu_all[iE]), float(np.sqrt(S_ee))
print(f"\nFitted own-price elasticity: {reg_mu_e:+.3f} (se {reg_se_e:.3f})  -> 'regression' case")
CASES[f"regression ({reg_mu_e:.3f}, {reg_se_e:.3f})"] = (reg_mu_e, reg_se_e)

# === FIX 5: report promotion coefficient significance ===
promo_coef = float(mu_all[names.index("lR")]); promo_p = float(free.pvalues["lR"])
print(f"Promotion coefficient: {promo_coef:+.4f} (p={promo_p:.3f})")
if promo_p > 0.10:
    print("  -> Not significant at 10%. Doc's +0.27 was a different specification.")
    print("     Treat promotion as weak/no effect until the log definition is revisited.")

# %% CELL 6: forecast distribution & promotion
def trend_forecast(s, h, window=52):
    s = s.iloc[-window:].astype(float); t = np.arange(len(s))
    k = np.polyfit(t, np.log(s.clip(lower=1e-9)), 1)
    return np.exp(np.polyval(k, np.arange(len(s), len(s) + h)))

def timesfm_joint(dist_s, promo_s, h, peak_hist, build_hist, peak_fut, build_fut):
    try:
        import torch
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        cfg = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=1,
                          device="cuda" if torch.cuda.is_available() else "cpu")
        fc = TimesFM3Evaluator(cfg)
        target = np.stack([dist_s.values, promo_s.values]).astype(np.float32)
        pf = np.stack([np.r_[peak_hist, peak_fut], np.r_[build_hist, build_fut]]).astype(np.float32)
        out = list(fc.predict_batch(contexts=[target], horizon=h, past_future_covariates=[pf],
                                    return_quantiles=True, use_symmetric_averaging=False))[0]
        f = np.asarray(out.forecast); assert f.shape == (2, h), f.shape
        return f[0], f[1]
    except Exception as ex:
        print("TimesFM unavailable -> trend forecast. Reason:", repr(ex)[:150]); return None

cap_ = lambda x, c: x if c is None else np.minimum(x, c)
D_f, R_f = trend_forecast(df[DIST], HORIZON), trend_forecast(df[PROMO], HORIZON)
FORECAST_SOURCE = "trend"
if USE_TIMESFM:
    r_ = timesfm_joint(df[DIST], df[PROMO], HORIZON, df[PEAK].values, df[BUILD].values, peak_f, build_f)
    if r_ is not None:
        D_f, R_f = np.maximum(r_[0], df[DIST].min()), np.maximum(r_[1], df[PROMO].min()); FORECAST_SOURCE = "TimesFM"
D_f, R_f = cap_(D_f, DIST_CAP), cap_(R_f, PROMO_CAP)
D_f = np.maximum(D_f, 1e-6); R_f = np.maximum(R_f, 1e-6 if not PROMO_L1 else 0.0)
print(f"Distribution/promotion forecast source: {FORECAST_SOURCE}")
print(f"Sanity | last 13 wks hist: dist={df[DIST].tail(13).mean():.1f} promo={df[PROMO].tail(13).mean():.2f}"
      f" | forecast mean: dist={D_f.mean():.1f} promo={R_f.mean():.2f}")

# %% CELL 7: simulation
assert HORIZON % 4 == 0, "HORIZON must be a multiple of 4"
NB = HORIZON // 4
CAP_BLOCK = CAP_M * 12 / 13
lnL0 = np.log(LEADER_NOW)

def run_case(e_mean, e_sd, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = (E_LO - e_mean) / e_sd, (E_HI - e_mean) / e_sd
    e = stats.truncnorm.rvs(a, b, loc=e_mean, scale=e_sd, size=N, random_state=rng)
    z = rng.standard_normal((N, len(iO)))
    beta = mu_o[None, :] + np.outer(e - mu_e, S_oe / S_ee) + z @ LC.T
    zr = rng.standard_normal((N, 4)); zr[:, 1] = RHO_DAL_BESAN * zr[:, 0] + np.sqrt(1 - RHO_DAL_BESAN**2) * zr[:, 1]
    u = stats.norm.cdf(zr); rm = np.zeros(N)
    for j, (g_, lo, mode, hi) in enumerate(RM.values()):
        rm += g_ / 1000 * stats.triang.ppf(u[:, j], c=(mode - lo) / (hi - lo), loc=lo, scale=hi - lo)
    vc = rm + OTHER_VAR
    react = rng.choice(REACTION_SHARE, size=N, p=REACTION_P)
    sD, sR = rng.normal(0, D_ERR_SD, N), rng.normal(0, R_ERR_SD, N)
    eps = rng.normal(0, NOISE_SD, (N, HORIZON))
    lD = lg(D_f[None, :] * np.exp(sD)[:, None], DIST_L1)
    lR = lg(R_f[None, :] * np.exp(sR)[:, None], PROMO_L1)
    base = (beta[:, [pos["const"]]] + beta[:, [pos["lL"]]] * lnL0 + beta[:, [pos["lD"]]] * lD
            + beta[:, [pos["lR"]]] * lR + beta[:, [pos[PEAK]]] * peak_f[None, :]
            + beta[:, [pos[BUILD]]] * build_f[None, :] + eps)
    out = {}
    for P in PRICES:
        leader_new = LEADER_NOW - react * max(MRP - P, 0)
        lq = base + (e * np.log(P))[:, None] + (beta[:, pos["lL"]] * np.log(leader_new / LEADER_NOW))[:, None]
        blocks = np.exp(lq).reshape(N, NB, 4).sum(2)
        # === FIX 3: track volume lost to cap, not just any-cap flag ===
        if CAP_MODE == "annual":
            cap_year, tot = CAP_M * 12 * HORIZON / 52, blocks.sum(1)
            units = np.minimum(tot, cap_year)
            vol_lost_frac = np.where(tot > 0, (tot - units) / np.maximum(tot, 1e-9), 0.0)
        else:
            tot = blocks.sum(1); capped = np.minimum(blocks, CAP_BLOCK).sum(1)
            units = capped
            vol_lost_frac = (tot - capped) / np.maximum(tot, 1e-9)
        profit = (NET_FACTOR * P - vc) * units - FIXED_M * 12 * HORIZON / 52
        out[P] = dict(units_m=units / (HORIZON / 52 * 12), profit=profit, vol_lost_frac=vol_lost_frac)
    drivers = {"elasticity": e, "raw material cost": rm, "distribution forecast error": sD,
               "promotion forecast error": sR, "leader reaction": react}
    return out, drivers

def summarise(out):
    yrs = HORIZON / 52; tgt = MIN_PROFIT_M * 12 * yrs
    rows = []
    for P in PRICES:
        o = out[P]; pr = o["profit"]
        rows.append({"price": P,
            "feasible": (P <= MRP) and (P >= MRP * (1 - MAX_STEP)) and (P <= MAX_VS_LEADER * LEADER_NOW),
            "units_per_month": o["units_m"].mean(), "profit_mean_L": pr.mean() / 1e5,
            "profit_p5_L": np.percentile(pr, 5) / 1e5, "profit_p95_L": np.percentile(pr, 95) / 1e5,
            "P(units>=min)": (o["units_m"] >= MIN_VOL_M).mean(), "P(profit>=target)": (pr >= tgt).mean(),
            # === FIX 3: meaningful capacity metrics ===
            "P(any blk > cap)": (o["vol_lost_frac"] > 0).mean(),
            "P(>2% vol lost)":  (o["vol_lost_frac"] > 0.02).mean(),
            "mean vol lost %":  100 * o["vol_lost_frac"].mean(),
            "P(loss)": (pr < 0).mean(),
            "P(both)": ((o["units_m"] >= MIN_VOL_M) & (pr >= tgt)).mean()})
    return pd.DataFrame(rows)

def show(res):
    f = res.copy(); f["units_per_month"] = f["units_per_month"].round(0)
    for c in ["profit_mean_L", "profit_p5_L", "profit_p95_L", "mean vol lost %"]:
        if c in f: f[c] = f[c].round(2)
    for c in [c for c in f.columns if c.startswith("P(")]:
        f[c] = (f[c] * 100).round(0).astype(int).astype(str) + "%"
    pd.set_option("display.width", 260); print(f.to_string(index=False))

# %% CELL 8: run all cases
ALL, SIMS = [], {}
for name, (m_, s_) in CASES.items():
    out, drv = run_case(m_, s_); res = summarise(out); res.insert(0, "case", name)
    ALL.append(res); SIMS[name] = (out, drv)
res_all = pd.concat(ALL, ignore_index=True)

main_out, main_drv = SIMS[MAIN_CASE]; main_res = res_all[res_all["case"] == MAIN_CASE].drop(columns="case")
hist_m = df[UNITS].tail(52).sum() / 12

# === FIX 4: explicit baseline reconciliation ===
print(f"\n=== BASELINE RECONCILIATION ===")
print(f"  Historical last-52wks units/month : {hist_m:.0f}")
print(f"  Project baseline (document)        : 14,250")
print(f"  Cost-sheet baseline (document)     : 14,250")
print(f"  Simulated @ Rs 280 (this run)      : {main_res.set_index('price').loc[280,'units_per_month']:.0f}")
lift = main_res.set_index('price').loc[280,'units_per_month'] / hist_m
print(f"  -> Ratio simulated/historical      : {lift:.3f}  "
      f"(= {100*(lift-1):+.1f}% lift from TimesFM distribution+promotion forecast)")
print(f"     Distribution {df[DIST].tail(13).mean():.1f} -> {D_f.mean():.1f} ; "
      f"Promotion {df[PROMO].tail(13).mean():.2f} -> {R_f.mean():.2f}")

print(f"\n=== MAIN CASE: {MAIN_CASE} | profit in Rs lakh per YEAR; targets: profit >= {MIN_PROFIT_M*12/1e5:.0f}L/yr, units >= {MIN_VOL_M}/month ===")
show(main_res)

print("\nProfit difference vs Rs 280 (Rs lakh/year), paired on identical draws:")
for P in PRICES:
    if P == 280: continue
    d = (main_out[P]["profit"] - main_out[280]["profit"]) / 1e5
    print(f"  Rs {P}: mean {d.mean():+.2f} | 5-95%: {np.percentile(d,5):+.2f} to {np.percentile(d,95):+.2f}"
          f" | better than 280 in {(d>0).mean()*100:.0f}% of draws")

print("\n=== ALL CASES (summary) ===")
rows = []
for name in CASES:
    r = res_all[res_all["case"] == name].set_index("price")
    feas = [P for P in PRICES if r.loc[P, "feasible"]]
    best = max(feas, key=lambda P: r.loc[P, "profit_mean_L"]) if feas else None
    d270 = (SIMS[name][0][270]["profit"] - SIMS[name][0][280]["profit"]).mean() / 1e5
    rows.append({"case": name, "best_feasible": best, "profit_280_L/yr": round(r.loc[280, "profit_mean_L"], 1),
                 "P(both)@280": f"{r.loc[280,'P(both)']*100:.0f}%",
                 "P(>2%vol lost)@280": f"{r.loc[280,'P(>2% vol lost)']*100:.0f}%",
                 "mean vol lost%@280": f"{r.loc[280,'mean vol lost %']:.1f}%",
                 "270 minus 280 (L/yr)": round(d270, 2),
                 "270 beats 280": f"{(SIMS[name][0][270]['profit']>SIMS[name][0][280]['profit']).mean()*100:.0f}%"})
print(pd.DataFrame(rows).to_string(index=False))

# === FIX 4b: flat-forecast sensitivity on MAIN_CASE ===
print("\n=== FLAT-FORECAST SENSITIVITY (main case, dist/promo frozen at last-52w mean) ===")
D_f_orig, R_f_orig = D_f.copy(), R_f.copy()
D_f = np.full(HORIZON, df[DIST].tail(52).mean())
R_f = np.full(HORIZON, df[PROMO].tail(52).mean())
out_flat, _ = run_case(*CASES[MAIN_CASE]); show(summarise(out_flat))
D_f, R_f = D_f_orig, R_f_orig

# %% CELL 9: drivers + charts + save
diff = main_out[270]["profit"] - main_out[280]["profit"]
imp = {k: stats.spearmanr(v, diff)[0] for k, v in main_drv.items()}
imp = dict(sorted(imp.items(), key=lambda kv: -abs(kv[1])))
print("\nDrivers of the DECISION: profit(Rs 270) - profit(Rs 280) - Spearman:")
for k, v in imp.items(): print(f"  {k:28s} {v:+.2f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].boxplot([main_out[P]["profit"] / 1e5 for P in PRICES], showfliers=False); ax[0].set_xticklabels(PRICES)
ax[0].axhline(MIN_PROFIT_M * 12 / 1e5, ls="--", c="red", label="target"); ax[0].legend()
ax[0].set_title("Annual profit (Rs lakh), main case"); ax[0].set_xlabel("Shelf price (Rs)")
w = 0.27; x = np.arange(len(PRICES))
for i, (c, lab) in enumerate([("P(units>=min)", "units >= min"),
                               ("P(profit>=target)", "profit >= target"),
                               ("P(>2% vol lost)", ">2% vol lost to cap")]):
    ax[1].bar(x + (i - 1) * w, main_res[c] * 100, w, label=lab)
ax[1].set_xticks(x); ax[1].set_xticklabels(PRICES); ax[1].set_ylabel("%"); ax[1].legend(fontsize=8)
ax[1].set_title("Probabilities (main case)")
ax[2].barh(list(imp.keys())[::-1], list(imp.values())[::-1]); ax[2].set_title("Drivers of profit(270) - profit(280)")
plt.tight_layout(); res_all.to_csv("monte_carlo_full_results.csv", index=False)
