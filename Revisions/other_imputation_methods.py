import pandas as pd
import numpy as np

def load_yearly_mortality(year):
    input_path = 'Revisions/Hepvu Data/hepvu_mort_rates_censored.csv'
    mort_names = ['FIPS', '2014 OD', '2015 OD', '2016 OD', '2017 OD', '2018 OD', '2019 OD', '2020 OD']
    cols_to_keep = ['FIPS', f'{year} OD']

    mort_df = pd.read_csv(input_path, header=0, names=mort_names)
    mort_df = mort_df[cols_to_keep]
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[f'{year} OD'] = mort_df[f'{year} OD'].astype(float)
    return mort_df

def impute_global_mean(yearly_df, year):
    mask = yearly_df[f'{year} OD'] != -9
    global_mean = yearly_df.loc[mask, f'{year} OD'].mean()
    
    yearly_df[f'{year} OD'] = yearly_df[f'{year} OD'].replace(-9, global_mean)
    yearly_df[f'{year} OD'] = yearly_df[f'{year} OD'].round(2)
    return yearly_df

def impute_state_mean(yearly_df, year): 
    yearly_df['State'] = yearly_df['FIPS'].str[:2]
    state_list = yearly_df['State'].unique()

    # Compute global mean (for fallback)
    global_mask = yearly_df[f'{year} OD'] != -9
    global_mean = yearly_df.loc[global_mask, f'{year} OD'].mean()

    for state in state_list:
        # Create a mask for non-missing values in this state
        state_mask = (yearly_df['State'] == state) & (yearly_df[f'{year} OD'] != -9)
        state_mean = yearly_df.loc[state_mask, f'{year} OD'].mean()

        # Fallback to global mean if state_mean is NaN
        if pd.isna(state_mean):
            state_mean = global_mean
        
        # Create a mask for missing values in this state
        missing_mask = (yearly_df['State'] == state) & (yearly_df[f'{year} OD'] == -9)
        yearly_df.loc[missing_mask, f'{year} OD'] = state_mean

    # Clean up
    yearly_df.drop(columns='State', inplace=True)
    yearly_df[f'{year} OD'] = yearly_df[f'{year} OD'].round(2)
    return yearly_df

def main():
    combined_df = pd.DataFrame()
    impute_method = 'global'  # Choose 'global' or 'state'

    for year in range(2014, 2021):
        yearly_df = load_yearly_mortality(year)

        if impute_method == 'global':
            yearly_df = impute_global_mean(yearly_df, year)
        elif impute_method == 'state':
            yearly_df = impute_state_mean(yearly_df, year)
        else:
            raise ValueError("Invalid imputation method. Choose 'global' or 'state'.")

        if combined_df.empty:
            combined_df = yearly_df
        else:
            combined_df = pd.merge(combined_df, yearly_df, on='FIPS', how='outer')

    output_path = f'Revisions/Hepvu Data/Imputed Data/hepvu_mort_rates_{impute_method}_imputation.csv'
    combined_df.to_csv(output_path, index=False)
    print('Final mortality rates saved.')

if __name__ == "__main__":
    main()
