import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# File paths
real_path = "Revisions/hepvu_mort_rates_real.csv"
censored_path = "Revisions/hepvu_mort_rates_censored.csv"
imputed_path = "Revisions/hepvu_mort_rates_imputed.csv"

# Load datasets
real_df = pd.read_csv(real_path)
censored_df = pd.read_csv(censored_path)
imputed_df = pd.read_csv(imputed_path)

# Store metrics per year
results = []

# Loop through each year and compute absolute errors where censored
for year in range(2014, 2021):
    col = f"{year} OD"
    
    # Boolean mask for censored values (-9.0)
    mask = censored_df[col] == -9.0

    # Get real and imputed values only where they were censored
    real_values = real_df.loc[mask, col]
    imputed_values = imputed_df.loc[mask, col]

    # Mean of the actual (ground truth) values that were censored
    mean_real = real_values.mean()
    
    # Compute absolute error
    abs_error = (real_values - imputed_values).abs()

    # Ensure they're aligned
    assert len(real_values) == len(imputed_values)

    mae = mean_absolute_error(real_values, imputed_values)
    rmse = root_mean_squared_error(real_values, imputed_values)

    results.append((year, mae, rmse, mean_real))
    print(f"{year}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, Real Mean = {mean_real:.2f}, MAE / Real Mean = {mae / mean_real:.4f}")
    
    # Plot histogram
    plt.figure()
    plt.hist(abs_error, bins=30, edgecolor='black')
    plt.title(f"Absolute Error Histogram: {year}")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Optional: Convert to DataFrame for summary or plotting
results_df = pd.DataFrame(results, columns=["Year", "MAE", "RMSE", "Real Mean"])
