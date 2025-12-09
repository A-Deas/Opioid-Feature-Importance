import pandas as pd
import numpy as np
import logging

# Constants
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
LOW_THRESH = 2
HIGH_TRESH = 98

# Set up logging
log_file = 'Log Files/empirical_anomaly_investigation.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),
    logging.StreamHandler()
])

def load_mort_rates():
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def compute_empirical_anomalies(mort_df, year):
    """Flags anomalies based on empirical distribution percentiles including 0s."""
    col = f'{year} Mortality Rates'
    rates = mort_df[col].values

    # Compute and print percentiles from 0% to 100% in 5% increments
    percentiles = np.arange(0, 105, 5)
    values = np.percentile(rates, percentiles)

    logging.info(f"\n📊 Percentile Summary for {year}:")
    for p, val in zip(percentiles, values):
        logging.info(f"  {p:3d}th percentile = {val:.2f}")

    low_cutoff = np.percentile(rates, LOW_THRESH)
    high_cutoff = np.percentile(rates, HIGH_TRESH)

    anomaly_labels = np.zeros(len(rates), dtype=int)
    anomaly_labels[rates <= low_cutoff] = -1  # cold anomaly
    anomaly_labels[rates >= high_cutoff] = 1   # hot anomaly

    mort_df[f'{year} Anomaly Label'] = anomaly_labels

    logging.info(f"Year {year}: Low cutoff is {low_cutoff:.2f}, high cutoff is {high_cutoff:.2f}")
    logging.info(f"  Cold anomalies: {(anomaly_labels == -1).sum()}")
    logging.info(f"  Hot anomalies:  {(anomaly_labels == 1).sum()}")
    return mort_df

def main():
    mort_df = load_mort_rates()

    for year in range(2010, 2023):
        mort_df = compute_empirical_anomalies(mort_df, year)

    logging.info("Anomaly detection completed and saved.")

if __name__ == "__main__":
    main()