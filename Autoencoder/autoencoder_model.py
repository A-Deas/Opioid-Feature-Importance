import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import KFold, train_test_split
import random
from scipy.stats import lognorm

# Constants
FEATURES = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
            # 'Disability', 
            'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
            'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
            'Single-Parent Household', 'Unemployed']
NUM_VARIABLES = len(FEATURES)
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} Mortality Rates' for year in range(2010, 2023)]
LOSS_FUNCTION = nn.L1Loss() # PyTorch's built-in loss function for MAE, measures the absolute difference between the predicted values and the actual values     
DATA_YEARS = range(2010, 2022) # Can't use data in 2022 as we are not making 2023 predictions
NUM_COUNTIES = 3144
KFOLDS = len(DATA_YEARS)  # Use as many folds as we have training years of data
NUM_EPOCHS = 500
PATIENCE = 10

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set up logging
log_file = 'Log Files/autoencoder_model.log'
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

def train_model(train_loader, val_loader, model, loss_function, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    best_validation_loss = float('inf')  # Track the best validation loss
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):  # Epoch loop
        epoch_train_loss = 0.0  # Reset epoch train loss
        num_batches_train = 0  # Batch counter for training

        model.train()  # Set the model to training mode
        for inputs, targets in train_loader:  # For each training batch
            optimizer.zero_grad()  # Reset gradients
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            targets = targets.squeeze(dim=0)  # Remove the batch dimension
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update the scheduler at the end of each batch
            epoch_train_loss += loss.item()  # Accumulate loss
            num_batches_train += 1  # Increment batch counter

        # Compute average train loss
        average_epoch_train_loss = epoch_train_loss / num_batches_train
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_epoch_train_loss:.4f}')

        # Validation step
        model.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0.0
        num_batches_val = 0

        with torch.no_grad():  # Disable gradients for validation
            for inputs, targets in val_loader:  # For each validation batch
                inputs = inputs.squeeze(dim=0)
                targets = targets.squeeze(dim=0)
                outputs = model(inputs)
                val_loss = loss_function(outputs, targets)
                epoch_val_loss += val_loss.item()
                num_batches_val += 1

        average_epoch_val_loss = epoch_val_loss / num_batches_val
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_epoch_val_loss:.4f}\n')

        # Early stopping based on validation loss
        if average_epoch_val_loss < best_validation_loss:
            best_validation_loss = average_epoch_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1} with best validation loss: {best_validation_loss:.4f}")
            break

    return best_validation_loss, best_model_state

def evaluate_model(test_loader, model, loss_function):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0  # Initialize/reset loss for this evaluation
    num_batches = 0 

    with torch.no_grad():  # Disable gradients for evaluation
        for inputs, targets in test_loader:  # Batch loop
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            targets = targets.squeeze(dim=0)  # Remove the batch dimension
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            total_loss += loss.item()  # Accumulate loss
            num_batches += 1  # Increment batch counter

    average_loss = total_loss / num_batches
    return average_loss

def predict_mortality_rates(predictions_loader):
    model = Autoencoder_model()
    model.load_state_dict(torch.load('PyTorch Models/autoencoder_model.pth'))
    model.eval()
    
    # Initialize an empty DataFrame with the desired column names
    years = [f'{year+1} AE Preds' for year in DATA_YEARS]
    predictions_df = pd.DataFrame(columns=years)
    year_counter = 2010
    
    with torch.no_grad():
        for inputs, _ in predictions_loader:
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            year_counter += 1 # 1st input is for 2015 predictions, 2nd is for 2016 and so on
            outputs = model(inputs)
            outputs_np = outputs.numpy()  # Convert tensor to numpy array
            outputs_np = np.round(outputs_np, 2)
            predictions_df[f'{year_counter} AE Preds'] = outputs_np.flatten() # Place the column vector in the appropriate column of the DataFrame

    print(predictions_df.head())
    # Save to CSV
    predictions_df.to_csv('Autoencoder/Predictions/ae_mortality_predictions.csv', index=False)
    print("Predictions saved to CSV --------------------------------")

def main():
    data_df = construct_data_df()
    mort_df = construct_mort_df(MORTALITY_PATH, MORTALITY_NAMES)
    tensors = Tensors(data_df, mort_df, years=DATA_YEARS)

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    best_fold_test_loss = float('inf')
    best_fold = -1
    best_train_indices = None
    best_test_indices = None
    best_model_state = None

    for fold, (train_indices, test_indices) in enumerate(kf.split(tensors)):
        logging.info(f'Fold {fold + 1}/{KFOLDS}: --------------------------------\n')

        # Split train indices into actual training and validation indices
        # I want to validate on only a single year to optimize training, I have 11 years of training
        # data so I need to take 1/11 or 0.0909 percent of the training data for validation
        train_indices, val_indices = train_test_split(train_indices, test_size=0.0909, random_state=42)

        train_set = Subset(tensors, train_indices)
        val_set = Subset(tensors, val_indices)
        test_set = Subset(tensors, test_indices)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

        # Train the model
        logging.info("Training model --------------------------------\n")
        model = Autoencoder_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=.001, step_size_up=10, mode='triangular2')
        best_validation_loss, best_fold_model_state = train_model(train_loader, val_loader, model, LOSS_FUNCTION, optimizer, scheduler)
        logging.info(f"Model training complete with best validation loss: {best_validation_loss:.4f} --------------------------------\n")

        # Test the model
        logging.info("Testing model --------------------------------\n")
        model = Autoencoder_model()
        model.load_state_dict(best_fold_model_state)  # Load the best model state from this fold
        test_loss = evaluate_model(test_loader, model, LOSS_FUNCTION)
        logging.info(f"Test loss on 2020 reconstructions: {test_loss:.4f}")

        # Save the best model based on test loss
        if test_loss < best_fold_test_loss:
            best_fold_test_loss = test_loss
            best_model_state = best_fold_model_state  # Save the best model state
            best_fold = fold
            best_train_indices = train_indices
            best_test_indices = test_indices
            torch.save(best_model_state, 'PyTorch Models/autoencoder_model.pth')
            logging.info("Best model updated from this fold based on test loss.")

    # Log the best fold details
    logging.info(f'Best Fold: {best_fold + 1}')
    logging.info(f'Best Fold Training Indices: {best_train_indices}')
    logging.info(f'Best Fold Testing Indices: {best_test_indices}')
    logging.info(f'Best Fold Test Loss: {best_fold_test_loss}')

    predictions_loader = DataLoader(tensors, batch_size=1, shuffle=False, num_workers=0)
    predict_mortality_rates(predictions_loader)


if __name__ == "__main__":
    main()