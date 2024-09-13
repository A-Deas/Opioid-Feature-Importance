import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
TAIL = 3  # Tails for anomaly detection

def load_mort_rates():
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)  # Pad FIPS codes with leading zeros
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def count_zero_values(mort_df, year):
    zero_count = (mort_df[f'{year} Mortality Rates'] == 0).sum()
    print(f'Year {year}: {zero_count} counties have zero mortality rates.')

def fit_distribution(mort_df, year):
    # Get the mortality data for the selected year
    mort_rates = mort_df[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]  # Ignore zero values for lognormal fit

    # Fit the lognormal distribution to the non-zero mortality rates
    log_shape, loc, scale = lognorm.fit(non_zero_mort_rates)
    
    # Generate points for the fitted lognormal distribution
    x_vals = np.linspace(non_zero_mort_rates.min(), non_zero_mort_rates.max(), 1000)
    pdf_vals = lognorm.pdf(x_vals, log_shape, loc=loc, scale=scale)
    
    # Plot the histogram of the actual data
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_mort_rates, bins=30, density=True, alpha=0.6, color='b', label='Mortality Data')
    
    # Plot the fitted lognormal distribution
    plt.plot(x_vals, pdf_vals, 'r-', lw=3, label=f'Fitted Lognormal\nShape: {log_shape:.2f}, Loc: {loc:.2f}, Scale: {scale:.2f}')
    
    # Add labels and title
    plt.title(f'{year} Mortality Rates and Fitted Lognormal Distribution', size=16)
    plt.xlabel('Mortality Rate', size=14)
    plt.ylabel('Density', size=14)
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Save the plot
    plt.savefig(f'Anomalies/Fitted Distributions/{year}_fitted_distribution.png')

def main():
    mort_df = load_mort_rates()

    for year in range(2010, 2023):
        count_zero_values(mort_df, year)
        fit_distribution(mort_df, year)

if __name__ == "__main__":
    main()
