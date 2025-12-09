import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# File paths
real_path = "Revisions/Hepvu Data/hepvu_mort_rates_real.csv"
censored_path = "Revisions/Hepvu Data/hepvu_mort_rates_censored.csv"
imputed_paths = {
    'global': "Revisions/Hepvu Data/Imputed Data/hepvu_mort_rates_global_imputation.csv",
    'state': "Revisions/Hepvu Data/Imputed Data/hepvu_mort_rates_state_imputation.csv",
    'neighbors': "Revisions/Hepvu Data/Imputed Data/hepvu_mort_rates_neighbor_imputation.csv",
    'idw': "Revisions/Hepvu Data/Imputed Data/hepvu_mort_rates_idw_imputation.csv"
}

# Load base datasets
real_df = pd.read_csv(real_path)
censored_df = pd.read_csv(censored_path)

# Preload imputed dataframes
imputed_dfs = {method: pd.read_csv(path) for method, path in imputed_paths.items()}

# Store metrics
results = []

# Loop through each year
for year in range(2014, 2021):
    col = f"{year} OD"
    mask = (censored_df[col] == -9.0) & (real_df[col] != -9.0)

    plt.figure()
    plt.title(f"Absolute Error Comparison: {year}")
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")

    for method, imputed_df in imputed_dfs.items():
        real_values = real_df.loc[mask, col]
        imputed_values = imputed_df.loc[mask, col]

        # NaN check
        if real_values.isna().any() or imputed_values.isna().any():
            print(f"⚠️ NaNs detected in {year}, method: {method}")
            bad_rows = pd.DataFrame({
                'FIPS': real_df.loc[mask, 'FIPS'],
                'Real': real_values,
                'Imputed': imputed_values
            })
            print(bad_rows[bad_rows.isna().any(axis=1)])
            continue

        abs_error = (real_values - imputed_values).abs()
        mae = mean_absolute_error(real_values, imputed_values)
        rmse = root_mean_squared_error(real_values, imputed_values)
        mean_real = real_values.mean()

        epsilon = 0.01  # Threshold to avoid division by zero

        # Filter out small real values for percentage calculations
        valid_mask = real_values > epsilon
        filtered_real = real_values[valid_mask]
        filtered_imputed = imputed_values[valid_mask]

        mape = (np.abs((filtered_real - filtered_imputed) / filtered_real)).mean() * 100
        mpe = ((filtered_real - filtered_imputed) / filtered_real).mean() * 100

        results.append((year, method, mae, rmse, mpe, mape, mae/mean_real))
        print(f"{year} [{method}]: MAE = {mae:.2f}, RMSE = {rmse:.2f}, MPE = {mpe:.2f}%, MAPE = {mape:.2f}%")
            #   Real Mean = {mean_real:.2f}, MAE/Mean = {mae/mean_real:.4f}")

        # Plot this method's histogram
        plt.hist(abs_error, bins=30, edgecolor='black', alpha=0.75, label=method)

    print()
    plt.legend()
    plt.tight_layout()
    plt.show()

# Convert to dataframe for review
results_df = pd.DataFrame(results, columns=["Year", "Method", "MAE", "RMSE", "MPE", "MAPE", "Normalized Real Mean"])
