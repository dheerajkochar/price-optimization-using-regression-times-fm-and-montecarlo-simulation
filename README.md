# Price Optimisation Model: Bikaneri Bhujia 1 kg

A personal / academic project that chooses the shelf price of a 1 kg packaged snack
(Bishanlal Babulal, Bikaner; follower brand, leader Bikaji at Rs 260).

**Read this first:** there is no real sales history. The 156 weeks of demand data were
generated from assumed rules. The project demonstrates a method and the logic of a pricing
decision. It is not real-world evidence, and no profit gain, deployment or TimesFM result is claimed.

## Method

1. **Unit economics (Excel):** price to net revenue, variable and fixed cost, profit equation.
2. **Elasticity regression:** weekly log-log regression with HAC standard errors.
3. **Demand forecast and sensitivity:** 52-week forecast for Rs 265 to 285 under several elasticities.
4. **Monte Carlo:** 50,000 scenarios of profit under business constraints, to get probabilities of meeting targets.

Profit equation used throughout:

```
Profit = (0.857 x Price - Variable cost) x Units - Fixed cost
```

- Net factor 0.857 = 0.90 / 1.05 (GST 5%, to be verified; retailer margin 10% of pre-tax price).
- Variable cost Rs 151.5 per unit without retailer commission. Fixed cost Rs 4,03,000 a month.

## Files

| File | What it does |
|---|---|
| `demand_profit_colab.ipynb` / `demand_profit.py` | Reads the workbook, fits the regression, forecasts distribution and promotion, and computes demand and annual profit for each price and elasticity. Route A uses a trend forecast; Route B uses TimesFM 3.0 if it loads, otherwise it falls back to Route A and labels itself `B_fallback`. |
| `monte_carlo_pricing.py` / `monte_carlo_colab.ipynb` | Simple Monte Carlo. Needs no data upload. Uses a fixed monthly baseline of 13,000 units at Rs 280. Good for explaining the method. |
| `monte_carlo_full_colab.ipynb` / `monte_carlo_full.py` | Fuller Monte Carlo. Weekly demand from the regression with distribution, promotion and festival terms; coefficients drawn jointly; capacity checks; three elasticity cases. Use this for the final analysis. |
| `pricing_workbook_fixed.xlsx` and related sheets | Master workbook with Checks and Change_Log sheets, constraints sheet and regression sheet (kept outside this folder if not included). |

## Input data

Excel sheet `data_historical` in `pricing_initial_data_workbook.xlsx`, with weekly rows.
The scripts detect these columns by name pattern and print the mapping they chose:

| Role | Example column name |
|---|---|
| Date | `week_start` |
| Own price | `shelf_price` |
| Leader price | `leader_price` |
| Units | `units_sold` (the one whose log equals `log_units_sold`) |
| Distribution | `distribution` |
| Promotion | `promotion` |
| Festival peak and buildup flags | `festival_peak`, `festival_buildup` |

Rows below the data (for example pasted regression text) are dropped automatically.
If a column is not found, the script stops and names it; set it by hand in the config.

## How to run (Google Colab)

1. Open a notebook (`demand_profit_colab.ipynb` or `monte_carlo_full_colab.ipynb`) in Colab.
2. Run all cells. When asked, upload `pricing_initial_data_workbook.xlsx`.
3. Check the printed column mapping, the row count (expect 156) and the festival agreement numbers (should be close to 1.0).
4. For TimesFM only: choose Runtime > Change runtime type > GPU, set `USE_TIMESFM = True` (in `monte_carlo_full`) and install per the official README (`pip install timesfm[torch]`; if `import timesfm3` then fails, clone the repository and `pip install -e .[torch]`).

Python packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl` (preinstalled in Colab except possibly `statsmodels`/`openpyxl`).
The scripts were tested on mock data shaped like the workbook, not on the real workbook.

## Key settings (edit in the config cell)

| Setting | Meaning |
|---|---|
| `PRICES` | Candidate shelf prices (265 to 285) |
| `CASES` | Elasticity cases as (mean, sd): (-1.5, 0.5) main, (-1.116, 0.268) regression, (-1.9, 0.5) stress |
| `NET_FACTOR` | 0.90 / 1.05 (owner's rule of thumb is 0.85) |
| `PEAK_REL`, `BUILD_REL`, festival date lists | Diwali peak = week 0; buildup = weeks -4 to -1; Holi and Raksha weeks also count as peak (verify dates) |
| `D_ERR_SD`, `R_ERR_SD` | Assumed forecast error of distribution and promotion (5% each) |
| `CAP_MODE` | `"block"`: capacity per 4-week block (conservative); `"annual"`: yearly total only (optimistic) |
| `USE_TIMESFM` | Off by default; the script falls back to a trend forecast if TimesFM fails |

## Constraints

- Shelf price must not exceed the printed MRP, Rs 280 (legal limit).
- At most 5% change in one step, so the floor is Rs 266 (Rs 265 needs a two-step reduction).
- At most 109% of the leader's price.
- Capacity 17,000 units a month; minimum volume 12,350 a month; profit target Rs 6.5 lakh a month (checked on simulated outcomes, not used as a price limit).
- A raised MRP (for example Rs 285) is a scenario only.

## Results so far (synthetic data, simple Monte Carlo, monthly)

| Shelf price | Feasible | Mean profit (Rs lakh) | P(profit >= 6.5L) | P(units >= 12,350) |
|---|---|---|---|---|
| 265 | No | 6.44 | 46% | 88% |
| 270 | Yes | 6.81 | 58% | 83% |
| 275 | Yes | 7.17 | 70% | 77% |
| **280** | Yes | **7.52** | 79% | 70% |
| 285 | No (above MRP) | 7.84 | 85% | 63% |

Rs 280 had the highest expected profit among feasible prices. Each Rs 5 cut cost about Rs 0.35 lakh a month.
A cut only pays if elasticity is below about -2.8.
Add the results of the fuller Monte Carlo here after running it on the real workbook.

## Assumptions and limitations

- **Synthetic data.** Coefficients reflect the rules used to generate it.
- **Weak identification.** Only 6 own-price changes in 3 years; elasticity is treated as a range (-1.1 central, -1.9 and -3 as stress). The distribution coefficient (about 1.3) probably absorbs growth.
- **Assumed inputs:** leader reaction mix (40% none / 40% half / 20% full match), perception scores, raw material most-likely values, and the 5% to 8% forecast errors.
- **Volume baseline is unresolved:** 13,000 (demand data), 14,250 (cost sheet) or 15,000 produced. The two Monte Carlo versions give different profit levels for this reason and should not be compared directly.
- **GST rate (5%) is unverified** and feeds the net factor.
- **TimesFM** was not run in the reported results (Route B used the fallback). The TimesFM call follows the official README but is unverified on the real weights.
- No segment, channel or online-versus-offline evidence exists.

## Legal and governance

- Never price above the printed MRP (Legal Metrology rules).
- Verify the GST rate on this product.
- Use only public competitor prices; do not coordinate with competitors (Competition Act, 2002).
- Follow marketplace terms; handle any personal data under the DPDP Act, 2023.
- TimesFM 3.0 weights are licensed for non-commercial, non-production use. Source code is Apache-2.0.

## Next steps

1. Settle one volume baseline and rerun everything on it.
2. Run the elasticity stress case and report whether the conclusion changes.
3. Get TimesFM running on a GPU, or drop it from the claims.
4. Decide whether Rs 265 for competitive markets stays in, with a stated goal.
5. Replace synthetic data with real sales and estimate elasticity by segment.
6. Run a 6 to 8 week price pilot with test and control markets and a pre-registered metric.

## Resume wording

**Price Optimisation Model for a Packaged-Food SKU (personal / academic project)**

- Built unit economics in Excel and a weekly log-log elasticity regression on synthetic data (central estimate about -1.1).
- Wrote a Python Monte Carlo (50,000 scenarios) to compare prices under cost, demand and competitor-reaction uncertainty.

Do not claim real data, profit gains, deployment, Double ML or TimesFM results.
