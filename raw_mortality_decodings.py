import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, for example
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Constants
YEARS = range(2014, 2021)
NUM_COUNTIES = 3143
NUM_VARIABLES = 16  # 14 SVI + Dispensing + Mortality
FEATURES = ['Mortality', 'Dispensing',
            'SVI Aged 17 or Younger', 'SVI Aged 65 or Older', 'SVI Below Poverty',
            'SVI Crowding', 'SVI Disability', 'SVI Group Quarters', 'SVI Limited English Ability',
            'SVI Minority Status', 'SVI Mobile Homes', 'SVI Multi-Unit Structures', 'SVI No High School Diploma',
            'SVI No Vehicle', 'SVI Single-Parent Household', 'SVI Unemployed']

def construct_decodings_df():
    mort_preds_path = '/home/p5d/volume1/NN_Practice/Hybrid Model/Data Encodings/Mortality_encoded_rates.csv'
    mort_preds_names = [f'{yr} Mortality Encoding' for yr in range(2014,2021)]
    dec_df = pd.read_csv(mort_preds_path, names=mort_preds_names, header=0)
    
    for variable in FEATURES:
        if variable != 'Mortality':
            # create a zero dataframe
            empty_variable_df = pd.DataFrame(0, index=range(1000), columns=[f'{yr} {variable} Encoding' for yr in YEARS])
            dec_df = pd.concat([dec_df, empty_variable_df], axis=1)

    dec_df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Debugging Folder/final_raw_mort_dec_df.csv', index=False)
    return dec_df

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in FEATURES:
        variable_path = f'/home/p5d/volume1/NN_Practice/Clean Data/{variable} rates.csv'
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

class Encodings_Tensors(Dataset):
    def __init__(self, dec_df, data_df, years=YEARS):
        self.dec_df = dec_df
        self.data_df = data_df
        self.years = years
        self.tensor_storage = list(range(len(self.years)))  # One data vector for each year

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]
        enc_list = []
        data_list = []
        for variable in FEATURES:
            if variable == 'Mortality':
                yearly_var_enc = self.dec_df[f'{year} Mortality Encoding'].values
                enc_list.append(yearly_var_enc)

                yearly_var_data = self.data_df[f'{year} {variable} rates'].values
                data_list.append(yearly_var_data)
    
            else:
                yearly_var_enc = np.zeros(1000)  # 0 vector of size 1000
                enc_list.append(yearly_var_enc)

                yearly_var_data = np.zeros(NUM_COUNTIES)  # 0 vector of size 3143
                data_list.append(yearly_var_data)

        yearly_enc_array = np.array(enc_list)
        yearly_enc_tensor = torch.tensor(yearly_enc_array, dtype=torch.float32)

        yearly_data_array = np.array(data_list)
        yearly_data_tensor = torch.tensor(yearly_data_array, dtype=torch.float32)

        return yearly_enc_tensor, yearly_data_tensor

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
        
    def decode(self, x):
        encoded_vars = [x[var, :].unsqueeze(0) for var in range(NUM_VARIABLES)]
        decoded_vars = [decoder(enc_var) for enc_var, decoder in zip(encoded_vars, self.decoder_layers)]
        reconstructed_x = torch.cat(decoded_vars, dim=0)
        return reconstructed_x

def data_decodings(data_loader):
    model = MainAutoencoder()
    model.load_state_dict(torch.load('/home/p5d/volume1/NN_Practice/Hybrid Model/PyTorch Models/main_autoencoder.pth'))
    model.eval()

    # Create a dictionary to hold DataFrames for each variable
    decodings_dfs = {variable: pd.DataFrame(columns=[f'{year} {variable} Decoding' for year in YEARS]) for variable in FEATURES}
    
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.squeeze(0)
            year = YEARS[idx]
            decoded_vars = model.decode(inputs)
            for var_idx, variable in enumerate(FEATURES):
                decodings_dfs[variable][f'{year} {variable} Decoding'] = decoded_vars[var_idx].numpy().flatten()

    # Save each DataFrame to a separate CSV file
    for variable, df in decodings_dfs.items():
        if variable == 'Mortality':
            df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Decodings/raw_mortality_decodings.csv', index=False)
            print(f"Mortality decodings saved to CSV")
        else:
            break

def main():
    dec_df = construct_decodings_df()
    data_df = construct_data_df()
    encs_tensors = Encodings_Tensors(dec_df, data_df)
    tensors_loader = DataLoader(encs_tensors, batch_size=1, shuffle=False)
    data_decodings(tensors_loader)

if __name__ == "__main__":
    main()
