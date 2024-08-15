import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0, for example
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import logging
from sklearn.model_selection import KFold
import numpy as np

# Constants
LOSS_FUNCTION = nn.L1Loss()  # MAE loss
YEARS = range(2014,2021)
NUM_EPOCHS = 50 ############################### Could def up this !!!
NUM_COUNTIES = 3143
NUM_VARIABLES = 16 # 14 SVI + Dispensing + Mortality
NOISE_FACTOR = 0.05  # Noise factor for denoising autoencoder
KFOLDS = 7  # Number of folds for cross-validation
PATIENCE = 5
FEATURES = ['Mortality', 'Dispensing', 
            'SVI Aged 17 or Younger', 'SVI Aged 65 or Older', 'SVI Below Poverty',
            'SVI Crowding', 'SVI Disability', 'SVI Group Quarters', 'SVI Limited English Ability',
            'SVI Minority Status', 'SVI Mobile Homes', 'SVI Multi-Unit Structures', 'SVI No High School Diploma',
            'SVI No Vehicle', 'SVI Single-Parent Household', 'SVI Unemployed']

# Set up logging
log_file = '/home/p5d/volume1/NN_Practice/Hybrid Model/Log Files/main_autoencoder.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

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
        # noisy_tensor = yearly_tensor + NOISE_FACTOR * torch.randn(yearly_tensor.size())
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
        
    def forward(self, x):
        variable_tensors = [x[var, :].unsqueeze(0) for var in range(NUM_VARIABLES)]
        # logging.info(variable_tensors)

        # Encode then decode each variable independently
        encoded_vars = [encoder(var) for var, encoder in zip(variable_tensors, self.encoder_layers)]
        decoded_vars = [decoder(enc_var) for enc_var, decoder in zip(encoded_vars, self.decoder_layers)]
        
        # Concatenate decoded variables back together
        reconstructed_x = torch.cat(decoded_vars, dim=0)
        # logging.info(reconstructed_x .shape)
        # logging.info(reconstructed_x)
        return reconstructed_x

def train_model(train_loader, model, loss_function, optimizer, scheduler, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    best_training_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):  # Epoch loop
        epoch_loss = 0.0  # Reset epoch loss
        num_batches = 0  # Batch counter

        model.train()  # Set the model to training mode
        for inputs, targets in train_loader:  # For each batch
            optimizer.zero_grad()  # Reset gradients
            inputs = inputs.squeeze(dim=0)  # Remove the batch dimension
            targets = targets.squeeze(dim=0)  # Remove the batch dimension
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            scheduler.step() # Update the scheduler at the end of each batch
            epoch_loss += loss.item()  # Accumulate loss
            num_batches += 1  # Increment batch counter

        average_epoch_loss = round(epoch_loss / num_batches, 4)  # Compute average epoch loss
        logging.info(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {average_epoch_loss}')  # Log epoch loss

        # Early stopping check
        if average_epoch_loss < best_training_loss:
            best_training_loss = average_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1} with best training loss: {best_training_loss}")
            break

    return best_training_loss, best_model_state

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

def main():
    # Construct the tensors
    data_df = construct_data_df()
    data_tensors = Data_Tensors(data_df)

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    best_fold_test_loss = float('inf')
    best_fold = -1
    best_train_indices = None
    best_test_indices = None
    best_model_state = None

    for fold, (train_indices, test_indices) in enumerate(kf.split(data_tensors)):
        logging.info(f'Fold {fold + 1}/{KFOLDS}: --------------------------------\n')

        train_set = Subset(data_tensors, train_indices)
        test_set = Subset(data_tensors, test_indices)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

        # Train the model 
        logging.info("Training model --------------------------------\n")
        model = MainAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # initial LR, but will be adjusted by the scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, step_size_up=5, mode='triangular2')
        best_training_loss, best_fold_model_state = train_model(train_loader, model, LOSS_FUNCTION, optimizer, scheduler)
        logging.info("Model training complete and best model saved --------------------------------\n")

        # Test the model
        logging.info("Testing model --------------------------------\n")
        model = MainAutoencoder()
        model.load_state_dict(best_fold_model_state)  # Load the best model state from this fold
        test_loss = evaluate_model(test_loader, model, LOSS_FUNCTION)
        logging.info(f"Test loss on 2020 reconstructions: {test_loss:.4f}")
        logging.info("Model testing complete --------------------------------\n")

        # Check if this fold's model is the best one based on test loss
        if test_loss < best_fold_test_loss:
            best_fold_test_loss = test_loss
            best_model_state = best_fold_model_state  # Save the best model state
            best_fold = fold
            best_train_indices = train_indices
            best_test_indices = test_indices
            torch.save(best_model_state, '/home/p5d/volume1/NN_Practice/Hybrid Model/PyTorch Models/main_autoencoder.pth')
            logging.info("Best model updated from this fold based on test loss.")
            
        logging.info("Model testing complete --------------------------------\n")

    # Log the best fold details
    logging.info(f'Best Fold: {best_fold + 1}')
    logging.info(f'Best Fold Training Indices: {best_train_indices}')
    logging.info(f'Best Fold Testing Indices: {best_test_indices}')
    logging.info(f'Best Fold Test Loss: {best_fold_test_loss}')

if __name__ == "__main__":
    main()