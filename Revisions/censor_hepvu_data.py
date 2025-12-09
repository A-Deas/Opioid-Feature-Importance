import pandas as pd
import numpy as np
import random

# Set a seed for reproducibility
np.random.seed(42)

# Load your full dataset (modify path as needed)
df = pd.read_csv("Revisions/hepvu_mort_rates_real.csv")
df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)
for col in df.columns:
    if col != 'FIPS':
        df[col] = df[col].astype(float)

# Make a copy to apply censoring
censored_df = df.copy()

# Loop through each year and randomly censor about half of the entries
for year in range(2014, 2021):
    col_name = f"{year} OD"

    # Get the indices of all valid (non-missing) entries
    valid_indices = df[df[col_name] != -9.0].index.tolist()
    
    # Randomly sample half of the valid indices to censor
    n_to_censor = len(valid_indices) // 2
    censored_indices = np.random.choice(valid_indices, n_to_censor, replace=False)
    
    # Set those values to -9.0 (simulate missingness)
    censored_df.loc[censored_indices, col_name] = -9.0

# Save the censored dataset to a new file
censored_df.to_csv("Revisions/hepvu_mort_rates_censored.csv", index=False)

print("Censored dataset saved as 'hepvu_censored_half_per_year.csv'.")
