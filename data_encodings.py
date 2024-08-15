import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, for example
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Constants
YEARS = range(2014,2021)
NUM_COUNTIES = 3143
NUM_VARIABLES = 16 # 14 SVI + Dispensing + Mortality
FEATURES = ['Mortality', 'Dispensing', 
            'SVI Aged 17 or Younger', 'SVI Aged 65 or Older', 'SVI Below Poverty',
            'SVI Crowding', 'SVI Disability', 'SVI Group Quarters', 'SVI Limited English Ability',
            'SVI Minority Status', 'SVI Mobile Homes', 'SVI Multi-Unit Structures', 'SVI No High School Diploma',
            'SVI No Vehicle', 'SVI Single-Parent Household', 'SVI Unemployed']

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in FEATURES:
        variable_path = f'Clean Data/{variable} rates.csv'
        variable_names = ['FIPS'] + [f'{yr} {variable} rates' for yr in range(2014, 2021)]
        variable_df = pd.read_csv(variable_path, header=0, names=variable_names)
        variable_df['FIPS'] = variable_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
        variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float).clip(lower=0)

        if data_df.empty:
            data_df = variable_df
        else:
            data_df = pd.merge(data_df, variable_df, on='FIPS', how='outer')

    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

class Data_Tensors(Dataset):
    def __init__(self, data_df, years=YEARS):
        self.data_df = data_df
        self.years = years
        self.tensor_storage = list(range(len(self.years))) # I want one data vector for each year

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]
        # logging.info(year)
        variable_list = []
        for variable in FEATURES:
            yearly_var_rates = self.data_df[f'{year} {variable} rates'].values
            variable_list.append(yearly_var_rates)
        yearly_data_array = np.array(variable_list)
        yearly_tensor = torch.tensor(yearly_data_array, dtype=torch.float32)
        # logging.info(yearly_data_vector.shape)
        # logging.info(yearly_data_vector)
        return yearly_tensor, yearly_tensor

class MainAutoencoder(nn.Module):
    def __init__(self):
        super(MainAutoencoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(NUM_COUNTIES, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000)
            ) for _ in range(NUM_VARIABLES)
        ])
        
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, NUM_COUNTIES)
            ) for _ in range(NUM_VARIABLES)
        ])
        
    def encode(self, x):
        variable_tensors = [x[var, :].unsqueeze(0) for var in range(NUM_VARIABLES)]
        encoded_vars = [encoder(var) for var, encoder in zip(variable_tensors, self.encoder_layers)]
        return encoded_vars
    
def data_encodings(data_loader):
    model = MainAutoencoder()
    model.load_state_dict(torch.load('/home/p5d/volume1/NN_Practice/Hybrid Model/PyTorch Models/main_autoencoder.pth'))
    model.eval()

    # Create a dictionary to hold DataFrames for each variable
    encodings_dfs = {variable: pd.DataFrame(columns=[f'{year} {variable} Encoding' for year in YEARS]) for variable in FEATURES}
    
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.squeeze(0)
            year = YEARS[idx]
            encoded_vars = model.encode(inputs)
            for var_idx, variable in enumerate(FEATURES):
                encodings_dfs[variable][f'{year} {variable} Encoding'] = encoded_vars[var_idx].numpy().flatten()

    # Save each DataFrame to a separate CSV file
    for variable, df in encodings_dfs.items():
        df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Data Encodings/{variable}_encoded_rates.csv', index=False)
        print(f"Encodings for {variable} saved to CSV")

def main():
    data_df = construct_data_df()
    data_tensors = Data_Tensors(data_df)
    data_loader = DataLoader(data_tensors, batch_size=1, shuffle=False)
    data_encodings(data_loader)

if __name__ == "__main__":
    main()