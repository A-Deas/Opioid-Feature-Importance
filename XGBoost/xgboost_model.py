import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xgboost as xgb
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)

# Constants
OPTIMIZED_XGBOOST = xgb.XGBRegressor(
    colsample_bytree=0.9,
    gamma=0,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    subsample=0.8,
    random_state=42)
KF = KFold(n_splits=5, shuffle=True, random_state=42)
DATA = ['Mortality',
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployed']

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in DATA:
        variable_path = f'Data/Clean/{variable}_rates.csv'
        variable_names = ['FIPS'] + [f'{year} {variable} Rates' for year in range(2014, 2021)]
        variable_df = pd.read_csv(variable_path, header=0, names=variable_names)
        variable_df['FIPS'] = variable_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
        variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float)

        if data_df.empty:
            data_df = variable_df
        else:
            data_df = pd.merge(data_df, variable_df, on='FIPS', how='outer')

    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

def strip_fips(data_df):
    fips_codes = data_df['FIPS']
    return fips_codes

def features_targets(data_df, year):
    targets = data_df[f'{year} Mortality Rates']
    columns_to_keep = [f'{year-1} {feature} Rates' for feature in DATA if feature != 'Mortality']
    features = data_df[columns_to_keep]
    return features, targets

def run_xgboost(features, targets, fips_codes):
    xgb_predictions = []
    all_test_fips = []  # Collect FIPS codes for all test sets
    feature_importances = np.zeros(features.shape[1])
    
    for train_index, test_index in KF.split(features):
        # Splitting the data for this fold
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_targets, test_targets = targets.iloc[train_index], targets.iloc[test_index]
        test_fips = fips_codes.iloc[test_index]  # Get FIPS codes for the test set

        # Training the XGBoost model
        OPTIMIZED_XGBOOST.fit(train_features, train_targets)
        predictions = OPTIMIZED_XGBOOST.predict(test_features)
        
        # Update with the results from the current fold
        feature_importances += OPTIMIZED_XGBOOST.feature_importances_
        xgb_predictions.extend(predictions)
        all_test_fips.extend(test_fips)  # accumulate FIPS codes in the order they are used to test

    feature_importances = feature_importances / KF.get_n_splits()

    return feature_importances, xgb_predictions, all_test_fips

def save_predictions(year, xgb_predictions, test_fips):
    saving_df = pd.DataFrame({
        'FIPS': test_fips,
        f'{year} XGBoost Predictions': xgb_predictions,
    })
    saving_df[f'{year} XGBoost Predictions'] = saving_df[f'{year} XGBoost Predictions'].round(2)
    saving_df = saving_df.sort_values(by='FIPS').reset_index(drop=True)
    saving_df.to_csv(f'XGBoost/XGBoost Predictions/{year}_xgboost_predictions.csv', index=False)

def update_total_importance(feature_importances, total_importance):
    total_importance += feature_importances
    return total_importance

def plot_feature_importance(feature_importance_df):
    # Number of features and number of years
    num_features = feature_importance_df.shape[0]
    num_years = feature_importance_df.shape[1]

    # Create a bar width
    bar_width = 1 / (num_years + 1)  # add some space between sets

    # Set position of bar on X axis
    r = np.arange(num_features)
    positions = [r + bar_width*i for i in range(num_years)]
    
    # Make the plot
    plt.figure(figsize=(12, 8))
    
    for i, year in enumerate(feature_importance_df.columns):
        if i == len(feature_importance_df.columns) - 1:  # Check if it's the last item
            plt.barh(positions[i], feature_importance_df[year], height=bar_width, label=str(year), color='black')
        else:
            plt.barh(positions[i], feature_importance_df[year], height=bar_width, label=str(year))

    # Add xticks on the middle of the group bars
    plt.ylabel('Features', fontweight='bold')
    plt.xlabel('Feature Importance (Gain)', fontweight='bold')
    plt.title('XGBoost Feature Importance', fontweight='bold')
    plt.yticks(r + bar_width, feature_importance_df.index)
    plt.legend(title='Year', loc='lower right')
    
    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    # plt.savefig('/home/p5d/volume1/NN_Practice/XGBoost/cumulative XGBoost feature importances.png')
    plt.show()
    plt.close()

def main():
    yearly_importance_dict = {yr: [] for yr in range(2015, 2021)}
    num_features = len(DATA) - 1
    total_importance = np.zeros(num_features)
    data_df = construct_data_df()
    fips_codes = strip_fips(data_df)
    for year in range(2015, 2021): # We start predicting for 2015, not 2014
        features, targets = features_targets(data_df, year)
        feature_importances, xgb_predictions, test_fips = run_xgboost(features, targets, fips_codes)
        save_predictions(year, xgb_predictions, test_fips)
        yearly_importance_dict[year] = feature_importances.tolist()
        total_importance = update_total_importance(feature_importances, total_importance)
    total_importance = total_importance / 6

    # Final overall importance plot
    feature_names = [feature for feature in DATA if feature != 'Mortality']
    feature_importance_df = pd.DataFrame(yearly_importance_dict, index=feature_names)
    feature_importance_df['Average'] = total_importance
    feature_importance_df = feature_importance_df.sort_values('Average', ascending=True)
    plot_feature_importance(feature_importance_df)

if __name__ == "__main__":
    main()