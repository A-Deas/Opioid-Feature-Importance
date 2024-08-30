import pandas as pd

MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
PREDICTIONS_PATH = 'Autoencoder/Predictions/ae_mortality_predictions.csv'
PREDICTIONS_NAMES = [f'{year} Preds' for year in range(2011, 2023)]

def load_mortality(mort_path, mort_names):
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def load_predictions(preds_path, preds_names):
    preds_df = pd.read_csv(preds_path, header=0, names=preds_names)
    preds_df[preds_names] = preds_df[preds_names].astype(float)

    # Initialize dictionaries to store the predicted means and standard deviations
    predicted_shapes = {}
    predicted_locs = {}
    predicted_scales = {}
    start_year = 2011

    # Extract the last three rows (shape, location, scale)
    for i, col in enumerate(preds_names):
        year = start_year + i
        predicted_shapes[year] = preds_df[col].iloc[-3]
        predicted_locs[year] = preds_df[col].iloc[-2]
        predicted_scales[year] = preds_df[col].iloc[-1]

    # Drop the last three rows from the predicted rates
    preds_df = preds_df.iloc[:-3].reset_index(drop=True)
    return preds_df, predicted_shapes, predicted_locs, predicted_scales

def strip_fips(mort_df):
    fips_df = mort_df['FIPS'].copy()
    return fips_df

def attach_fips_to_predictions(fips_df, preds_df):
    preds_df['FIPS'] = fips_df
    cols = ['FIPS'] + [col for col in preds_df.columns if col != 'FIPS']
    preds_df = preds_df[cols]
    return preds_df

def merge_preds_with_data(preds_df, mort_df):
    preds_df = preds_df.merge(mort_df, on='FIPS', how='left')
    return preds_df

def calculate_errors_for_year(preds_df, year):
    actual_col = f'{year} Mortality Rates'
    pred_col = f'{year} Preds'
    error_col = f'{year} Error'

    # Calculate the absolute error for the specified year
    preds_df[error_col] = (preds_df[actual_col] - preds_df[pred_col]).abs()

    # Get the top 10 rows with the highest error for the specified year
    top10_df = preds_df.nlargest(10, error_col)

    # Keep only the FIPS, prediction, and error columns
    top10_df = top10_df[['FIPS', pred_col, actual_col, error_col]]

    # Reset the index and drop the old index
    top10_df = top10_df.reset_index(drop=True)

    return top10_df


def main():
    mort_df = load_mortality(MORTALITY_PATH, MORTALITY_NAMES)
    preds_df, _, _, _ = load_predictions(PREDICTIONS_PATH, PREDICTIONS_NAMES)
    fips_df = strip_fips(mort_df)

    preds_df = attach_fips_to_predictions(fips_df, preds_df)
    preds_df = merge_preds_with_data(preds_df, mort_df)

    year = 2021
    top10_df = calculate_errors_for_year(preds_df, year)

    print(f"Top 10 errors for {year}:")
    print(top10_df)

if __name__ == "__main__":
    main()
