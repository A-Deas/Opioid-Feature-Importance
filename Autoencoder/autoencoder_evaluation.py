from scipy import stats
from scipy.stats import lognorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
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

def calculate_err_acc(mort_df, preds_df):
    acc_df = mort_df[['FIPS']].copy()
    metrics = {'Year': [], 'Avg Error': [], 'Max Error': [], 'Avg Accuracy': [], 
               'MSE': [], 'R2': [], 'MedAE': []}

    for year in range(2011, 2023):
        absolute_errors = abs(preds_df[f'{year} Preds'] - mort_df[f'{year} Mortality Rates'])
        acc_df[f'{year} Absolute Errors'] = absolute_errors
        avg_err = np.mean(absolute_errors)
        max_err = absolute_errors.max()
        mse = np.mean(absolute_errors ** 2)
        r2 = 1 - (np.sum((mort_df[f'{year} Mortality Rates'] - preds_df[f'{year} Preds']) ** 2) / np.sum((mort_df[f'{year} Mortality Rates'] - np.mean(mort_df[f'{year} Mortality Rates'])) ** 2))
        medae = np.median(absolute_errors)

        # Adjusting accuracy calculation
        if max_err == 0:  # Perfect match scenario
            acc_df[f'{year} Accuracy'] = 0.9999
        else:
            acc_df[f'{year} Accuracy'] = 1 - (absolute_errors / max_err)
            acc_df[f'{year} Accuracy'] = acc_df[f'{year} Accuracy'].apply(lambda x: 0.9999 if x == 1 else (0.0001 if x == 0 else x))
        
        avg_acc = np.mean(acc_df[f'{year} Accuracy'])
        
        metrics['Year'].append(year)
        metrics['Avg Error'].append(avg_err)
        metrics['Max Error'].append(max_err)
        metrics['Avg Accuracy'].append(avg_acc)
        metrics['MSE'].append(mse)
        metrics['R2'].append(r2)
        metrics['MedAE'].append(medae)
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def kl_divergence_lognorm(shape1, loc1, scale1, shape2, loc2, scale2):
    sigma1, mu1 = shape1, np.log(scale1)
    sigma2, mu2 = shape2, np.log(scale2)
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

def wasserstein_distance_lognorm(shape1, loc1, scale1, shape2, loc2, scale2):
    sigma1, mu1 = shape1, np.log(scale1)
    sigma2, mu2 = shape2, np.log(scale2)
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

def compare_distributions(mort_df, predicted_shapes, predicted_locs, predicted_scales):
    for year in range(2011, 2023):
        mort_rates = mort_df[f'{year} Mortality Rates'].values
        mort_rates = mort_rates + 1e-5 # add a small values to avoid log(0) problems

        # Fit a log-normal distribution to the actual mortality rates
        params_lognorm = lognorm.fit(mort_rates, floc=0)
        shape, loc, scale = params_lognorm
    
        predicted_shape = predicted_shapes[year]
        predicted_loc = predicted_locs[year]
        predicted_scale = predicted_scales[year]

        kl_div = kl_divergence_lognorm(shape, loc, scale, predicted_shape, predicted_loc, predicted_scale)
        w_dist = wasserstein_distance_lognorm(shape, loc, scale, predicted_shape, predicted_loc, predicted_scale)

        # Round values to two decimal places
        kl_div = round(kl_div, 4)
        w_dist = round(w_dist, 2)

        print(f'Year: {year}')
        print(f'KL Divergence: {kl_div}')
        print(f'Wasserstein Distance: {w_dist}\n')

        # Visual comparison of the distributions
        plt.figure(figsize=(10, 5))

        # Plot actual distribution
        x = np.linspace(min(mort_rates), max(mort_rates), 100)
        plt.plot(x, lognorm.pdf(x, shape, loc, scale), label='Target Distribution', color='blue')

        # Plot predicted distribution
        plt.plot(x, lognorm.pdf(x, predicted_shape, predicted_loc, predicted_scale), label='Predicted Distribution', color='red', linestyle='dashed')

        plt.title(f'Distribution Comparison for Year {year}')
        plt.xlabel('Mortality Rates')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Autoencoder/Autoencoder Predictions/Yearly Distributions/{year}_distribution_comparison.png')
        plt.close()

def main():
    mort_df = load_mortality(MORTALITY_PATH, MORTALITY_NAMES)
    preds_df, predicted_shapes, predicted_locs, predicted_scales = load_predictions(PREDICTIONS_PATH, PREDICTIONS_NAMES)
    metrics_df = calculate_err_acc(mort_df, preds_df)
    metrics_df = metrics_df.round(4)
    print(metrics_df)

    compare_distributions(mort_df, predicted_shapes, predicted_locs, predicted_scales)

if __name__ == "__main__":
    main()