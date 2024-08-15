from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
DATA_PATH = 'Clean Data/Mortality rates.csv'
PREDICTIONS_PATH = '/home/p5d/volume1/NN_Practice/Hybrid Model/Decodings/final_mortality_predictions.csv'
DATA_NAMES = ['FIPS'] + [f'{yr} Data' for yr in range(2014, 2021)]
PREDICTIONS_NAMES = [f'{yr} Preds' for yr in range(2015, 2021)]

def load_data(data_path, data_names, predictions_path, predictions_names):
    data_df = pd.read_csv(data_path, header=0, names=data_names)
    data_df['FIPS'] = data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    data_df[data_names[1:]] = data_df[data_names[1:]].astype(float).clip(lower=0)

    preds_df = pd.read_csv(predictions_path, header=0, names=predictions_names)
    preds_df[predictions_names] = preds_df[predictions_names].astype(float)
    return data_df, preds_df

def calculate_err_acc(data_df, preds_df):
    acc_df = data_df[['FIPS']].copy()
    metrics = {'Year': [], 'Avg Error': [], 'Max Error': [], 'Avg Accuracy': [], 
               'MSE': [], 'R2': [], 'MedAE': []}

    for year in range(2015, 2021):
        absolute_errors = abs(preds_df[f'{year} Preds'] - data_df[f'{year} Data'])
        acc_df[f'{year} Absolute Errors'] = absolute_errors
        avg_err = np.mean(absolute_errors)
        max_err = absolute_errors.max()
        mse = np.mean(absolute_errors ** 2)
        r2 = 1 - (np.sum((data_df[f'{year} Data'] - preds_df[f'{year} Preds']) ** 2) / np.sum((data_df[f'{year} Data'] - np.mean(data_df[f'{year} Data'])) ** 2))
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

def plot_pred_vs_actual(data_df, preds_df):
    for year in range(2015, 2021):
        plt.figure(figsize=(10, 6))
        actual_values = data_df[f'{year} Data']
        predicted_values = preds_df[f'{year} Preds']
        plt.scatter(actual_values, predicted_values)
        plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], color='r', linestyle='--')  # y=x line
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs. Actual Values for {year}')
        plt.tight_layout()
        # plt.savefig(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Residuals/{year}_residual_plot.png')
        plt.show()
        plt.close()
        # print(f'Predicted vs. actual plot for {year} saved.')

def main():
    data_df, preds_df = load_data(DATA_PATH, DATA_NAMES, PREDICTIONS_PATH, PREDICTIONS_NAMES)
    metrics_df = calculate_err_acc(data_df, preds_df)
    metrics_df = metrics_df.round(4)
    print(metrics_df)
    # plot_pred_vs_actual(data_df, preds_df)

if __name__ == "__main__":
    main()
