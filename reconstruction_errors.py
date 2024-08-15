from scipy import stats
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
MORTALITY_PATH = 'Clean Data/Mortality rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{yr} Mortality rates' for yr in range(2014, 2021)]
DECODINGS_PATH = '/home/p5d/volume1/NN_Practice/Hybrid Model/Decodings/raw_mortality_decodings.csv'
DECODINGS_NAMES = [f'{yr} Raw Mortality Decodings' for yr in range(2015, 2021)]

def load_mortality(data_path, data_names):
    mort_df = pd.read_csv(data_path, header=0, names=data_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[data_names[1:]] = mort_df[data_names[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def load_raw_decodings(decodings_path, decodings_names):
    dec_df = pd.read_csv(decodings_path, header=0, names=decodings_names)
    dec_df[decodings_names] = dec_df[decodings_names].astype(float)
    dec_df = dec_df.reset_index(drop=True)
    return dec_df

def reconstruction_dataframe(mort_df, dec_df):
    recon_columns = [f'{year} Mortality Reconstruction Error' for year in range(2015,2021)]
    recon_df = pd.DataFrame(columns=recon_columns)
    
    for year in range(2015,2021):
        recon_df[f'{year} Mortality Reconstruction Error'] = abs( dec_df[f'{year} Raw Mortality Decodings'] - mort_df[f'{year} Mortality rates'] )
        recon_df[f'{year} Mortality Reconstruction Error'] = recon_df[f'{year} Mortality Reconstruction Error'].round(2)

    # Save to CSV
    recon_df.to_csv('/home/p5d/volume1/NN_Practice/Hybrid Model/Anomalies/mortality_reconstruction_errors.csv', index=False)
    print("Reconstruction errors saved to CSV --------------------------------")
    return recon_df

def distribution_of_errors(recon_df):
    for year in range(2015, 2021):
        year_col = f'{year} Mortality Reconstruction Error'
        mu = recon_df[year_col].mean()
        sigma = recon_df[year_col].std()
        print(f'{year} Reconstruction Errors:  mean = {mu:.2f} and std = {sigma:.2f}')

def detect_anomalies(recon_df):
    mu_sigma_dict = {}

    for year in range(2015, 2021): 
        year_col = f'{year} Mortality Reconstruction Error'
        mu = recon_df[year_col].mean()
        sigma = recon_df[year_col].std()
        mu_sigma_dict[year] = (mu, sigma)

    # Define anomaly threshold as mean + 3*std
    thresholds = {year: mu + 3*sigma for year, (mu, sigma) in mu_sigma_dict.items()}

    anomaly_columns = [f'{year} Anomalies' for year in range(2015,2021)]
    anomaly_df = pd.DataFrame(columns=anomaly_columns)
    
    # Add columns for each year's anomalies
    for year in range(2015, 2021):
        yearly_threshold = thresholds[year]
        anomaly_df[f'{year} Anomalies'] = (recon_df[f'{year} Mortality Reconstruction Error'] > yearly_threshold).astype(int)

    # Save anomalies to CSV
    anomaly_df.to_csv('/home/p5d/volume1/NN_Practice/Hybrid Model/Anomalies/anomalies.csv', index=False)
    print()
    print("Anomalies saved to CSV --------------------------------")

def main():
    mort_df = load_mortality(MORTALITY_PATH, MORTALITY_NAMES)
    dec_df = load_raw_decodings(DECODINGS_PATH, DECODINGS_NAMES)
    recon_df = reconstruction_dataframe(mort_df, dec_df)
    distribution_of_errors(recon_df)
    detect_anomalies(recon_df)

if __name__ == "__main__":
    main()