"""
COVID-19 Outcomes, Testing, and Vaccination: Exploratory Analysis
Reproduces the statistics and figures in report/COVID19_Analysis_Report.docx

Run from the project folder:  python analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

os.makedirs("figures", exist_ok=True)

CASES, DEATHS, TESTS = "Tot Cases/ 1M pop", "Deaths/ 1M pop", "Tests/ 1M pop"

# 1. U.S. DATA
us = pd.read_csv("data/USA_Covid_Data.csv")
print("U.S. dataset:", us.shape)
print("Duplicate rows:", us.duplicated().sum(), "| duplicate states:", us["USA State"].duplicated().sum())
print("Missing values by column:\n", us.isnull().sum())
print("Negative values:", int((us.select_dtypes('number') < 0).sum().sum()))
print("States with cases > population:", int((us["Total Cases"] > us["Population"]).sum()))
# Total Recovered / Active Cases are missing for 5 states; left as missing, not analyzed.

print("\nDescriptive statistics:")
print(us[[CASES, DEATHS, TESTS]].agg(["mean", "median", "std", "min", "max"]).round(0))

# 2. CORRELATIONS
print("\nCorrelations (n = 50):")
for x, y in [(CASES, DEATHS), (TESTS, CASES), (TESTS, DEATHS)]:
    r, p = stats.pearsonr(us[x], us[y])
    rho, _ = stats.spearmanr(us[x], us[y])
    se = 1 / np.sqrt(len(us) - 3)                       # Fisher z confidence interval
    lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
    print(f"  {x} vs {y}: r={r:.3f} (95% CI {lo:.3f} to {hi:.3f}), p={p:.4f}, Spearman={rho:.3f}")

# 3. REGRESSION
us["cases_10k"] = us[CASES] / 10_000       # rescale: per 10,000 cases per million
us["tests_M"] = us[TESTS] / 1e6            # rescale: per million tests per million

m1 = sm.OLS(us[DEATHS], sm.add_constant(us[["cases_10k"]])).fit()
m2 = sm.OLS(us[DEATHS], sm.add_constant(us[["cases_10k"]])).fit(cov_type="HC3")
print("\nModel 1: OLS, deaths ~ cases")
print(m1.summary().tables[1], "R-squared:", round(m1.rsquared, 3))
print("\nModel 2: same, robust (HC3) standard errors")
print(m2.summary().tables[1])

# Outlier / influence diagnostics
infl = m1.get_influence()
us["stud_resid"] = infl.resid_studentized_external
us["cooks_d"] = infl.cooks_distance[0]
flagged = us[us["stud_resid"].abs() > 2]
print("\nOutliers (|studentized residual| > 2):")
print(flagged[["USA State", CASES, DEATHS, "stud_resid", "cooks_d"]].round(3).to_string(index=False))

# Sensitivity check: refit without flagged states
sub = us[~us["USA State"].isin(flagged["USA State"])]
m3 = sm.OLS(sub[DEATHS], sm.add_constant(sub[["cases_10k"]])).fit()
print(f"\nModel 3: excluding {', '.join(flagged['USA State'])} (n={int(m3.nobs)})")
print(m3.summary().tables[1], "R-squared:", round(m3.rsquared, 3))

m4 = sm.OLS(us[DEATHS], sm.add_constant(us[["cases_10k", "tests_M"]])).fit(cov_type="HC3")
print("\nModel 4: cases + tests, robust SEs")
print(m4.summary().tables[1], "R-squared:", round(m4.rsquared, 3))

# 4. U.S. FIGURES
slope, intercept = m1.params["cases_10k"] / 10_000, m1.params["const"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(us[CASES] / 1000, us[DEATHS], alpha=0.85)
xs = np.linspace(us[CASES].min(), us[CASES].max(), 50)
ax.plot(xs / 1000, intercept + slope * xs, color="red")
for _, row in flagged.iterrows():
    ax.annotate(row["USA State"], (row[CASES] / 1000, row[DEATHS]), xytext=(5, 5), textcoords="offset points")
ax.set(xlabel="Reported cases per million (thousands)", ylabel="Reported deaths per million",
       title="Case rates vs. death rates across 50 U.S. states")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/case_vs_death_rates.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(us[TESTS] / 1e6, us[CASES] / 1000, alpha=0.85)
ax.set(xlabel="Tests per million (millions)", ylabel="Reported cases per million (thousands)",
       title="Testing rates vs. case rates across 50 U.S. states")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figures/tests_vs_case_rates.png", dpi=200); plt.close()

# ---------------------------------------------------------------- 5. VACCINATION DATA
vax = pd.read_csv("data/Worldwide_Vaccine_Data.csv")
vax.columns = ["country", "doses_per_100", "total_doses", "pct_vax", "pct_full"]
print("\nVaccination dataset:", vax.shape, "| missing:", int(vax.isnull().sum().sum()),
      "| duplicates:", int(vax.duplicated().sum()))
print(vax[["doses_per_100", "pct_vax", "pct_full"]].describe().round(1))

vax["gap"] = vax["pct_vax"] - vax["pct_full"]
print(f"\nMean gap (>=1 dose minus fully vaccinated): {vax['gap'].mean():.1f} points")
print(vax.nlargest(5, "gap")[["country", "pct_vax", "pct_full", "gap"]].to_string(index=False))
print("\nHighest % fully vaccinated:\n", vax.nlargest(5, "pct_full")[["country", "pct_full"]].to_string(index=False))
print("\nLowest % fully vaccinated:\n", vax.nsmallest(5, "pct_full")[["country", "pct_full"]].to_string(index=False))
print("\nDoses/100 vs % fully vaccinated: r = %.3f" % stats.pearsonr(vax["doses_per_100"], vax["pct_full"])[0])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(vax["pct_full"], bins=20, edgecolor="white")
ax.set(xlabel="% of population fully vaccinated", ylabel="Number of countries",
       title="Distribution of full vaccination coverage (187 countries)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("figures/vaccination_distribution.png", dpi=200); plt.close()
print("\nDone. Figures saved to figures/")
