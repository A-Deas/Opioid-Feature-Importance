import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Constants ###################### I could reoptimize XGBoost here !!!!!
xgb_optimal = xgb.XGBRegressor( 
    colsample_bytree=0.9,
    gamma=0,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
FEATURES = ['Mortality', 'Dispensing', 
            'SVI Aged 17 or Younger', 'SVI Aged 65 or Older', 'SVI Below Poverty',
            'SVI Crowding', 'SVI Disability', 'SVI Group Quarters', 'SVI Limited English Ability',
            'SVI Minority Status', 'SVI Mobile Homes', 'SVI Multi-Unit Structures', 'SVI No High School Diploma',
            'SVI No Vehicle', 'SVI Single-Parent Household', 'SVI Unemployed']

def construct_dataframe():
    final_df = pd.DataFrame()
    for variable in FEATURES:
        var_enc_path = f'/home/p5d/volume1/NN_Practice/Hybrid Model/Data Encodings/{variable}_encoded_rates.csv'
        var_enc_names = [f'{yr} {variable} Encoding' for yr in range(2014, 2021)]
        var_enc_df = pd.read_csv(var_enc_path, header=0, names=var_enc_names)
        var_enc_df[var_enc_names[1:]] = var_enc_df[var_enc_names[1:]].astype(float)
        var_enc_df = var_enc_df.reset_index(drop=True)

        var_data_path = f'/home/p5d/volume1/NN_Practice/Clean Data/{variable} rates.csv'
        var_data_names = ['FIPS'] + [f'{yr} {variable} rates' for yr in range(2014, 2021)]
        var_data_df = pd.read_csv(var_data_path, header=0, names=var_data_names)
        var_data_df['FIPS'] = var_data_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
        var_data_df[var_data_names[1:]] = var_data_df[var_data_names[1:]].astype(float).clip(lower=0)
        var_data_df = var_data_df.sort_values(by='FIPS').reset_index(drop=True)

        variable_df = pd.DataFrame() # reset for each variable

        for yr in range(2014, 2021):
            yearly_list = []

            data = var_data_df[f'{yr} {variable} rates'].values
            yearly_list.append(data)

            encodings = var_enc_df[f'{yr} {variable} Encoding'].values
            yearly_list.append(encodings)

            yearly_column = np.concatenate(yearly_list)
            temp_df = pd.DataFrame(yearly_column, columns=[f'{yr} {variable} Data and Encoding'])

            variable_df = pd.concat([variable_df, temp_df], axis=1) # append each year to the variable_df
            variable_df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Debugging Folder/{variable}_df.csv', index=False)

        if final_df.empty:
            final_df = variable_df
        else:
            final_df = pd.concat([final_df, variable_df], axis=1) # append each variable to the final_df

    final_df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/Debugging Folder/final_df.csv', index=False)
    return final_df

def features_targets(final_df, year):
    targets = final_df[f'{year+1} Mortality Data and Encoding']
    feature_columns = [f'{year} {variable} Data and Encoding' for variable in FEATURES if variable != 'Mortality']
    features = final_df[feature_columns]
    return features, targets

def folded_xgboost(xgb_optimal, kf, features, targets):
    folded_xgb_predictions = []
    folded_actuals = []
    feature_importances = np.zeros(features.shape[1])
    
    for train_index, test_index in kf.split(features):
        # Splitting the data for this fold
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_targets, test_targets = targets.iloc[train_index], targets.iloc[test_index]

        # Training the RandomForest model
        xgb_optimal.fit(train_features, train_targets)
        predictions = xgb_optimal.predict(test_features)
        
        # Update with the results from the current fold
        feature_importances += xgb_optimal.feature_importances_
        folded_xgb_predictions.extend(predictions)
        folded_actuals.extend(test_targets)

    feature_importances = feature_importances / kf.get_n_splits()

    return feature_importances, folded_xgb_predictions, folded_actuals

def save_predictions(year, folded_xgb_predictions):
    saving_df = pd.DataFrame({f'{year+1} XGBoost Predicted Mortality Rates and Encoding': folded_xgb_predictions})
    saving_df[f'{year+1} XGBoost Predicted Mortality Rates and Encoding'] = saving_df[f'{year+1} XGBoost Predicted Mortality Rates and Encoding'].round(2)
    saving_df = saving_df.reset_index(drop=True)
    saving_df.to_csv(f'/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Predictions/{year+1} XGBoost Predicted Mortality Rates and Encoding.csv', index=False)

def update_total_importance(feature_importances, total_importance):
    total_importance += feature_importances
    return total_importance

def cumulative_feature_importance_plot(feature_importance_df):
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
        plt.barh(positions[i], feature_importance_df[year], height=bar_width, label=str(year))
    
    # Add xticks on the middle of the group bars
    plt.ylabel('Features', fontweight='bold')
    plt.xlabel('Gain Relative Importance', fontweight='bold')
    plt.title('XGBoost Feature Importances', fontweight='bold')
    plt.yticks(r + bar_width, feature_importance_df.index)
    plt.legend(title='Year', loc='upper right')
    
    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Feature Importance/cumulative_xgb_feature_importance.png', dpi=300)
    print('Cumulative importance plot printed and saved.')
    plt.close()

def plot_average_shap_values(shap_values_dict, features):
    # Combine SHAP values from all years
    combined_shap_values = np.mean([shap_values for shap_values in shap_values_dict.values()], axis=0)
    
    # Create a DataFrame for easier plotting
    shap_values_df = pd.DataFrame(combined_shap_values, columns=features.columns)
    
    # Plot the average SHAP values
    shap.summary_plot(shap_values_df.values, features, plot_type="bar", show=False)
    plt.title('Average SHAP Values Across All Years')
    plt.tight_layout()
    plt.savefig('/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Feature Importance/average_shap_values.png')
    plt.close()
    print('Average SHAP values plot saved.')

def final_xgboost_predictions():
    result_df = pd.DataFrame()

    for year in range(2015, 2021):
        path = f'/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Predictions/{year} XGBoost Predicted Mortality Rates and Encoding.csv'

        df = pd.read_csv(path)
        processed_df = df.iloc[3143:].reset_index(drop=True) # Remove the raw mortality predictions
        processed_df.columns = [f'{year} XGB Predicted Mortality Encoding'] # Rename the column
        
        if result_df.empty:
            result_df = processed_df
        else:
            result_df = pd.concat([result_df, processed_df], axis=1)

    # Save the final DataFrame
    result_df.to_csv('/home/p5d/volume1/NN_Practice/Hybrid Model/XGBoost Predictions/final_xgb_predicted_mortality_encodings.csv', index=False)
    print('Final result saved.')

def main():
    yearly_importance_dict = {yr: [] for yr in range(2015, 2021)}
    num_features = len(FEATURES) - 1 # need to substract mortality from features
    total_importance = np.zeros(num_features)
    final_df = construct_dataframe()
    shap_values_dict = {yr: [] for yr in range(2015, 2021)}

    for year in range(2014, 2020): # We start predicting for 2015, not 2014
        features, targets = features_targets(final_df, year)
        feature_importances, folded_xgb_predictions, folded_actuals = folded_xgboost(xgb_optimal, kf, features, targets)
        save_predictions(year, folded_xgb_predictions)
        yearly_importance_dict[year+1] = feature_importances.tolist()
        total_importance = update_total_importance(feature_importances, total_importance)

        # SHAP explainer
        explainer = shap.Explainer(xgb_optimal)
        shap_values = explainer(features)
        shap_values_dict[year+1] = shap_values.values

    total_importance = total_importance / 6

    # Final overall importance plot
    feature_names = [variable if variable == 'Dispensing' else variable.replace('SVI ', '') for variable in FEATURES if variable != 'Mortality']
    feature_importance_df = pd.DataFrame(yearly_importance_dict, index=feature_names)
    feature_importance_df['All Years'] = total_importance
    feature_importance_df = feature_importance_df.sort_values('All Years', ascending=True)
    cumulative_feature_importance_plot(feature_importance_df)

    # Plot average SHAP values
    plot_average_shap_values(shap_values_dict, features)

    # Grab the final predicted mortality encodings
    final_xgboost_predictions()

if __name__ == "__main__":
    main()

