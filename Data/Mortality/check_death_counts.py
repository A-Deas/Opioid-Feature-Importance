import pandas as pd
from pathlib import Path
import numpy as np

RAW_DIR = Path("Data/Mortality/Raw Files")
INTERIM_DIR = Path("Data/Mortality/Interim Files")
FINAL_RATES_PATH = Path("Data/Mortality/Final Files/Mortality_final_rates.csv")
MISSING_VALUE = -9.0

def check_imputed_deaths():
    # Load final mortality rates (wide format: FIPS, 2010 MR, ..., 2022 MR)
    final_df = pd.read_csv(FINAL_RATES_PATH, dtype={"FIPS": str})
    final_df["FIPS"] = final_df["FIPS"].str.zfill(5)

    years = range(2010, 2023)

    for year in years:
        year_col = f"{year} MR"
        raw_path = RAW_DIR / f"{year}_cdc_wonder_raw_mortality.csv"
        interim_path = INTERIM_DIR / f"{year}_mortality_interim.csv"

        raw_df = pd.read_csv(raw_path, dtype={"FIPS": str})
        interim_df = pd.read_csv(interim_path, dtype={"FIPS": str})

        raw_df["FIPS"] = raw_df["FIPS"].str.zfill(5)
        interim_df["FIPS"] = interim_df["FIPS"].str.zfill(5)

        suppressed = raw_df[raw_df["Crude Rate"] == "Suppressed"][["FIPS"]]

        suppressed = suppressed.merge(
            interim_df[["FIPS", "Population"]],
            on="FIPS",
            how="left"
        )

        suppressed = suppressed.merge(
            final_df[["FIPS", year_col]].rename(columns={year_col: "final_rate"}),
            on="FIPS",
            how="left"
        )

        suppressed["implied_deaths"] = (
            suppressed["final_rate"] * suppressed["Population"] / 100000.0
        )

        # Summary statistics
        total = len(suppressed)
        over_9 = (suppressed["implied_deaths"] > 9).sum()
        pct = 100 * over_9 / total if total > 0 else 0.0

        summarize_implied_deaths(suppressed, year)

        print(
            f"{year}: {over_9} of {total} suppressed counties have implied deaths > 9: {pct:.2f}%"
        )

def summarize_implied_deaths(suppressed, year):
    vals = suppressed["implied_deaths"]

    summary = {
        "min": vals.min(),
        "max": vals.max(),
        "mean": vals.mean(),
        "median": vals.median(),
        "std": vals.std(),
        "p90": vals.quantile(0.90),
        "p95": vals.quantile(0.95),
        "p99": vals.quantile(0.99),
    }

    print(f"\n{year} implied deaths summary:")
    for k, v in summary.items():
        print(f"  {k:>6}: {v:6.2f}")

if __name__ == "__main__":
    check_imputed_deaths()
