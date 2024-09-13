import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import random
from scipy.stats import lognorm
import warnings
import logging

# Constants
FEATURES = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
            # 'Disability', 
            'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
            'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
            'Single-Parent Household', 'Unemployed']
NUM_VARIABLES = len(FEATURES)
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]  
DATA_YEARS = range(2010, 2022) # Can't use data in 2022 as we are not making 2023 predictions
NUM_COUNTIES = 3144

# Set up logging
log_file = 'Log Files/shapley.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in FEATURES:
        variable_path = f'Data/SVI/Final Files/{variable}_final_rates.csv'
        variable_names = ['FIPS'] + [f'{year} {variable} Rates' for year in range(2010, 2023)]
        variable_df = pd.read_csv(variable_path, header=0, names=variable_names)
        variable_df['FIPS'] = variable_df['FIPS'].astype(str).str.zfill(5)
        variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float)

        if data_df.empty:
            data_df = variable_df
        else:
            data_df = pd.merge(data_df, variable_df, on='FIPS', how='outer')

    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

def construct_mort_df(mort_path, mort_names):
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).str.zfill(5)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

class Tensors(Dataset):
    def __init__(self, data_df, mort_df, years=DATA_YEARS):
        self.data_df = data_df
        self.mort_df = mort_df
        self.years = years
        self.tensor_storage = list(range(len(self.years))) # I want one data vector for each year

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]
        variable_list = []
        for variable in FEATURES:
            yearly_var_rates = self.data_df[f'{year} {variable} Rates'].values
            variable_list.append(yearly_var_rates)
        yearly_data_array = np.array(variable_list)
        yearly_data_tensor = torch.tensor(yearly_data_array, dtype=torch.float32)

        # Append the lognormal parameters to the mortality rates
        mort_rates = self.mort_df[f'{year+1} Mortality Rates'].values
        non_zero_mort_rates = mort_rates[mort_rates > 0]
        lognorm_params = lognorm.fit(non_zero_mort_rates)
        shape, loc, scale = lognorm_params
        mort_rates = np.append(mort_rates, [shape, loc, scale])

        mort_rates_array = np.array(mort_rates)
        mort_rates_tensor = torch.tensor(mort_rates_array, dtype=torch.float32)
        return yearly_data_tensor, mort_rates_tensor
    
class Autoencoder_model(nn.Module):
    def __init__(self):
        super(Autoencoder_model, self).__init__()                   

        self.conv1d = nn.Conv1d(in_channels=NUM_VARIABLES, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NUM_COUNTIES, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000) ) ])
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, NUM_COUNTIES + 3) ) ])

    def forward(self, x):
        x = self.conv1d(x).squeeze(1)  # Remove the channel dimension after conv1d
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        x = x.squeeze(0) # remove the batch dimension
        return x

def explain_model_with_shap(model, tensor_loader):
    yearly_shap_values = []
    model.eval()

    for i, (input_tensor, _) in enumerate(tensor_loader):
        logging.info(f"Input tensor shape: {input_tensor.shape}")

        def shapley_forward(x):
            with torch.no_grad(): 
                output = model(x)
            return output.numpy()

        explainer = shap.GradientExplainer((model, model.conv1d), input_tensor)

        # Explain the model's predictions for the inputs
        shap_values = explainer.shap_values(input_tensor)
        logging.info(f"shap_values shape: {shap_values[0].shape}\n")

        # Aggregate SHAP values across all counties (take the mean absolute value for each feature)
        aggregated_shap_values = []
        for feature in range(len(FEATURES)):
            county_values = shap_values[0][feature, :]
            feature_value = np.mean(np.abs(county_values))  # Take the mean absolute SHAP value
            aggregated_shap_values.append(feature_value)
        
        aggregated_shap_values = np.array(aggregated_shap_values)
        yearly_shap_values.append(aggregated_shap_values)
    
    return yearly_shap_values

def plot_feature_importance(yearly_shap_values):
    # Convert the SHAP values into a DataFrame for plotting
    yearly_shap_values_df = pd.DataFrame(yearly_shap_values, index=[f"{year+1}" for year in DATA_YEARS], columns=FEATURES)

    # Calculate the average SHAP value across all years for each feature
    yearly_shap_values_df.loc['Average'] = yearly_shap_values_df.mean(axis=0)

    # Transpose the DataFrame so that features are on the Y-axis and years are the columns
    yearly_shap_values_df = yearly_shap_values_df.T

    # Sort the DataFrame by the average SHAP value
    yearly_shap_values_df = yearly_shap_values_df.sort_values(by='Average', ascending=True)

    # Color the bars on the importance plot
    num_years = len(DATA_YEARS)
    colors = list(plt.cm.tab20.colors[:num_years]) + ['black']  # Add black for 'Average'

    # Plot the SHAP feature importance over the years
    ax = yearly_shap_values_df.plot(kind='barh', figsize=(12, 8), legend=True, color=colors)

    plt.title('Shapley Feature Importance', fontweight='bold')
    plt.xlabel('Mean |SHAP Value|', fontweight='bold')
    plt.legend(title='Year', bbox_to_anchor=(1, 0), loc='lower right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Feature Importance/shapley_feature_importance.png', bbox_inches='tight')
    plt.close()

    # Log the average SHAP value for each variable
    logging.info("Average SHAP Value for each variable:")
    for feature, avg_shap in yearly_shap_values_df['Average'].items():
        logging.info(f"{feature}: {avg_shap:.4f}")

def main():
    data_df = construct_data_df()
    mort_df = construct_mort_df(MORTALITY_PATH, MORTALITY_NAMES)
    tensors = Tensors(data_df, mort_df)

    model = Autoencoder_model()
    model.load_state_dict(torch.load('PyTorch Models/autoencoder_model.pth'))
    tensor_loader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)

    # Explain the model with SHAP
    yearly_shap_values = explain_model_with_shap(model, tensor_loader)
    plot_feature_importance(yearly_shap_values)

if __name__ == "__main__":
    main()