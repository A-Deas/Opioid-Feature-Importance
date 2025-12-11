import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold
import random
from scipy.stats import lognorm
from pathlib import Path

# Constants
INTERIM_DIR = Path("Data/Mortality/Interim Files")
MISSING_VALUE = -9.0

FEATURES = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
            # 'Disability', 
            'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
            'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
            'Single-Parent Household', 'Unemployment']
NUM_VARIABLES = len(FEATURES)
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
# LOSS_FUNCTION = nn.L1Loss() # PyTorch's built-in loss function for MAE, measures the absolute difference between the predicted values and the actual values     
DATA_YEARS = range(2010, 2022) # Can't use data in 2022 as we are not making 2023 predictions
NUM_COUNTIES = 3144
NUM_EPOCHS = 100
PATIENCE = 5

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set up logging
log_file = 'Log Files/autoencoder_NMD_model.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def construct_masks(years, num_counties=NUM_COUNTIES):
    """
    Build a dict: year -> mask vector of length (NUM_COUNTIES + 3)
    Mask = 1 for counties with observed (non-missing) mortality in year+1,
           0 for missing/imputed;
    Last 3 entries (lognormal params) are always included (mask=1).
    """
    masks = {}

    for year in years:
        target_year = year + 1  # we predict mortality in year+1
        year_col = f"{target_year} MR"
        interim_path = INTERIM_DIR / f"{target_year}_mortality_interim.csv"

        interim_df = pd.read_csv(interim_path, dtype={"FIPS": str})
        interim_df["FIPS"] = interim_df["FIPS"].str.zfill(5)
        interim_df = interim_df.sort_values("FIPS").reset_index(drop=True)

        # Counties with *observed* mortality (not -9.0)
        county_mask = (interim_df[year_col].values != MISSING_VALUE).astype(float)

        if len(county_mask) != num_counties:
            raise ValueError(
                f"Expected {num_counties} counties, got {len(county_mask)} in {interim_path}"
            )

        # Append 3 ones for the lognormal params (we always include them in the loss)
        full_mask = np.concatenate([county_mask, np.ones(3, dtype=float)])
        masks[year] = full_mask

    return masks

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
    def __init__(self, data_df, mort_df, masks, years=DATA_YEARS):
        self.data_df = data_df
        self.mort_df = mort_df
        self.years = list(years)
        self.masks = masks  # dict: year -> full_mask (NUM_COUNTIES+3)
        self.tensor_storage = list(range(len(self.years))) # I want one data vector for each year

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]

        # Build SVI input tensor: shape [NUM_VARIABLES, NUM_COUNTIES]
        variable_list = []
        for variable in FEATURES:
            yearly_var_rates = self.data_df[f'{year} {variable} Rates'].values
            variable_list.append(yearly_var_rates)
        yearly_data_array = np.array(variable_list)
        yearly_data_tensor = torch.tensor(yearly_data_array, dtype=torch.float32)

        # Mortality rates for year+1, plus lognormal params
        mort_rates = self.mort_df[f'{year+1} Mortality Rates'].values
        non_zero_mort_rates = mort_rates[mort_rates > 0]
        shape, loc, scale = lognorm.fit(non_zero_mort_rates)
        mort_rates = np.append(mort_rates, [shape, loc, scale])

        mort_rates_tensor = torch.tensor(mort_rates, dtype=torch.float32)

        # Mask for this year (NUM_COUNTIES+3)
        mask_np = self.masks[year]
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

        return yearly_data_tensor, mort_rates_tensor, mask_tensor
    
def masked_l1_loss(outputs, targets, mask):
    """
    Compute MAE only over entries where mask == 1.
    outputs, targets, mask are 1D tensors of the same length.
    """
    diff = torch.abs(outputs - targets) * mask
    # avoid dividing by total length; divide by number of active elements
    return diff.sum() / mask.sum()
    
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

def train_model(train_loader, val_loader, model, optimizer, scheduler,
                num_epochs=NUM_EPOCHS, patience=PATIENCE):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets, mask in train_loader:
            optimizer.zero_grad()
            inputs = inputs.squeeze(dim=0)   # [NUM_VARIABLES, NUM_COUNTIES]
            targets = targets.squeeze(dim=0) # [NUM_COUNTIES + 3]
            mask = mask.squeeze(dim=0)       # [NUM_COUNTIES + 3]

            outputs = model(inputs)          # [NUM_COUNTIES + 3]
            loss = masked_l1_loss(outputs, targets, mask)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        average_epoch_loss = round(epoch_loss / num_batches, 4)
        logging.info(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {average_epoch_loss}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, val_mask in val_loader:
                val_inputs = val_inputs.squeeze(dim=0)
                val_targets = val_targets.squeeze(dim=0)
                val_mask = val_mask.squeeze(dim=0)

                val_outputs = model(val_inputs)
                val_loss += masked_l1_loss(val_outputs, val_targets, val_mask).item()

        average_val_loss = round(val_loss / len(val_loader), 4)
        logging.info(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Loss: {average_val_loss}\n')

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1} with best validation loss: {best_val_loss}")
            break

    return best_val_loss, best_model_state

def evaluate_model(test_loader, model):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs = inputs.squeeze(dim=0)
            targets = targets.squeeze(dim=0)
            mask = mask.squeeze(dim=0)

            outputs = model(inputs)
            loss = masked_l1_loss(outputs, targets, mask)

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss

def predict_mortality_rates(predictions_loader):
    model = Autoencoder_model()
    model.load_state_dict(torch.load('PyTorch Models/auto_NMD_model.pth'))
    model.eval()
    
    years = [f'{year+1} AE Preds' for year in DATA_YEARS]
    predictions_df = pd.DataFrame(columns=years)
    year_counter = 2010
    
    with torch.no_grad():
        for inputs, _, _ in predictions_loader:  # ignore targets, mask
            inputs = inputs.squeeze(dim=0)
            year_counter += 1
            outputs = model(inputs)
            outputs_np = outputs.numpy()
            outputs_np = np.round(outputs_np, 2)
            predictions_df[f'{year_counter} AE Preds'] = outputs_np.flatten()

    predictions_df.to_csv('Autoencoder/Auto_NMD_Predictions/auto_NMD_preds.csv', index=False)
    print("Predictions saved to CSV --------------------------------")

def main():
    data_df = construct_data_df()
    mort_df = construct_mort_df(MORTALITY_PATH, MORTALITY_NAMES)

    # Build per-year masks (based on which counties had observed mortality)
    masks = construct_masks(DATA_YEARS)

    tensors = Tensors(data_df, mort_df, masks=masks, years=DATA_YEARS)

    train_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    val_indices =  [5]
    test_indices = [11]

    train_set = Subset(tensors, train_indices)
    val_set = Subset(tensors, val_indices)
    test_set = Subset(tensors, test_indices)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    logging.info("Training model --------------------------------\n")
    model = Autoencoder_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.00001, max_lr=.001, step_size_up=10, mode='triangular2'
    )
    best_validation_loss, best_model_state = train_model(
        train_loader, val_loader, model, optimizer, scheduler
    )
    torch.save(best_model_state, 'PyTorch Models/auto_NMD_model.pth')
    logging.info("Model training complete and saved --------------------------------\n")

    logging.info("Testing model --------------------------------\n")
    model = Autoencoder_model()
    model.load_state_dict(best_model_state)
    test_loss = evaluate_model(test_loader, model)
    logging.info(f"Test loss for 2022 predictions: {test_loss:.4f}")
    logging.info("Model testing complete --------------------------------\n")

    predictions_loader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)
    predict_mortality_rates(predictions_loader)

if __name__ == "__main__":
    main()