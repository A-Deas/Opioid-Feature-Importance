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
XGB_PATH = '/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Predictions/final_xgb_predicted_mortality_encodings.csv'
XGB_NAMES = [f'{yr} XGB Predicted Mortality Encoding' for yr in range(2015, 2021)]
MORTALITY_PATH = '/home/p5d/volume1/NN_Practice/Clean Data/Mortality rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{yr} Mortality rates' for yr in range(2014, 2021)]
LOSS_FUNCTION = nn.L1Loss()  # MAE loss
YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
NUM_EPOCHS = 200 ############################ I can increase this even further !!!!!!!!
NUM_COUNTIES = 3143
KFOLDS = 6  # Number of folds for cross-validation
PATIENCE = 5  # Early stopping patience

# Set up logging
log_file = '/home/p5d/volume1/NN_Practice/Hybrid Model/Log Files/xgb_decoder.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def grab_xgb_predicted_encodings():
    xgb_df = pd.read_csv(XGB_PATH, header=0, names=XGB_NAMES)
    xgb_df[XGB_NAMES[1:]] = xgb_df[XGB_NAMES[1:]].astype(float).clip(lower=0)
    xgb_df = xgb_df.reset_index(drop=True)
    return xgb_df

def grab_mortality_rates():
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    mort_df = mort_df.drop(columns=['FIPS'])
    mort_df = mort_df.drop(columns=['2014 Mortality rates'])
    return mort_df

class Yearly_Tensors(Dataset):
    def __init__(self, xgb_df, mort_df, years=YEARS):
        self.xgb_df = xgb_df
        self.mort_df = mort_df
        self.years = years
        self.tensor_storage = list(range(len(self.years)))

    def __len__(self):
        return len(self.tensor_storage)
    
    def __getitem__(self, idx):
        year = self.years[idx]
        xgb_predicted_enc = self.xgb_df[f'{year} XGB Predicted Mortality Encoding'].values
        xgb_tensor = torch.tensor(xgb_predicted_enc, dtype=torch.float32)

        mort_rates = self.mort_df[f'{year} Mortality rates'].values
        mort_tensor = torch.tensor(mort_rates, dtype=torch.float32)
        return xgb_tensor, mort_tensor

class Xgb_Decoder(nn.Module):
    def __init__(self):
        super(Xgb_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, NUM_COUNTIES))
        
    def forward(self, x):
        x = self.decoder(x)
        return x

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

def predict_mortality_rates(data_loader):
    model = Xgb_Decoder()
    model.load_state_dict(torch.load('/home/p5d/volume1/NN_Practice/Hybrid Model/PyTorch Models/xgb_decoder.pth'))
    model.eval()
    
    # Initialize an empty DataFrame with the desired column names
    years = [f'{year} Final Mortality Predictions' for year in YEARS]
    predictions_df = pd.DataFrame(columns=years)
    year_counter = 2014
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            year_counter += 1 # 1st input is for 2015 predictions, 2nd is for 2016 and so on
            outputs = model(inputs)
            outputs_np = outputs.numpy()  # Convert tensor to numpy array
            outputs_np = np.round(outputs_np, 2)
            predictions_df[f'{year_counter} Final Mortality Predictions'] = outputs_np.flatten() # Place the column vector in the appropriate column of the DataFrame

    print(predictions_df.head())
    # Save to CSV
    predictions_df.to_csv('/home/p5d/volume1/NN_Practice/Hybrid Model/Decodings/final_mortality_predictions.csv', index=False)
    print("Predictions saved to CSV --------------------------------")

def main():
    # Construct the tensors
    xgb_df = grab_xgb_predicted_encodings()
    mort_df = grab_mortality_rates()
    yearly_tensors = Yearly_Tensors(xgb_df, mort_df)
    
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    best_fold_test_loss = float('inf')
    best_fold = -1
    best_train_indices = None
    best_test_indices = None
    best_model_state = None

    for fold, (train_indices, test_indices) in enumerate(kf.split(yearly_tensors)):
        logging.info(f'Fold {fold + 1}/{KFOLDS}: --------------------------------\n')
        
        train_set = Subset(yearly_tensors, train_indices)
        test_set = Subset(yearly_tensors, test_indices)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

        # Train the model 
        logging.info("Training model --------------------------------")
        model = Xgb_Decoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # initial LR, but will be adjusted by the scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001, step_size_up=5, mode='triangular2')
        best_training_loss, best_fold_model_state = train_model(train_loader, model, LOSS_FUNCTION, optimizer, scheduler)
        logging.info("Model training complete and best model saved --------------------------------\n")

        # Test the model
        logging.info("Testing model --------------------------------")
        model.load_state_dict(best_fold_model_state)  # Load the best model state from this fold
        test_loss = evaluate_model(test_loader, model, LOSS_FUNCTION)
        logging.info(f"Test Loss on this fold: {test_loss:.4f}")

        # Check if this fold's model is the best one based on test loss
        if test_loss < best_fold_test_loss:
            best_fold_test_loss = test_loss
            best_model_state = best_fold_model_state  # Save the best model state
            best_fold = fold
            best_train_indices = train_indices
            best_test_indices = test_indices
            torch.save(best_model_state, '/home/p5d/volume1/NN_Practice/Hybrid Model/PyTorch Models/xgb_decoder.pth')
            logging.info("Best model updated from this fold based on test loss.")
            
        logging.info("Model testing complete --------------------------------\n")

    # Log the best fold details
    logging.info(f'Best Fold: {best_fold + 1}')
    logging.info(f'Best Fold Training Indices: {best_train_indices}')
    logging.info(f'Best Fold Testing Indices: {best_test_indices}')
    logging.info(f'Best Fold Test Loss: {best_fold_test_loss}')

    tensor_loader = DataLoader(yearly_tensors, batch_size=1, shuffle=False, num_workers=0)
    predict_mortality_rates(tensor_loader)

if __name__ == "__main__":
    main()
