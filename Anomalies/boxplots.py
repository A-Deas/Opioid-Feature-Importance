import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

DATA = ['Mortality', 
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
        # 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployment']
TAIL = 2

# Set up logging
log_file = 'Log Files/anomaly_means.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def construct_data_df():
    data_df = pd.DataFrame()
    for variable in DATA:
        if variable == 'Mortality':
            variable_path = f'Data/Mortality/Final Files/{variable}_final_rates.csv'
            variable_names = ['FIPS'] + [f'{year} {variable} Rates' for year in range(2010, 2023)]
        else:
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

def boxplots(data_df, year):
    mort_rates = data_df[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]
    norm_params = lognorm.fit(non_zero_mort_rates)
    log_shape, loc, scale = norm_params

    tail = TAIL / 100
    upper_threshold = lognorm.ppf(1-tail, log_shape, loc, scale)
    lower_threshold = lognorm.ppf(tail, log_shape, loc, scale)

    # Initialize county categories
    data_df['County Category'] = 'Other'

    data_df.loc[(data_df[f'{year} Mortality Rates'] > upper_threshold), 'County Category'] = 'Hot'
    data_df.loc[(data_df[f'{year} Mortality Rates'] < lower_threshold), 'County Category'] = 'Cold'

    # Calculate hot means for each feature
    hot_means = {}
    for feature in DATA:
        if feature != 'Mortality':
            hot_means[feature] = data_df.loc[data_df['County Category'] == 'Hot', f'{year} {feature} Rates'].mean()

    # Calculate cold means for each feature
    cold_means = {}
    for feature in DATA:
        if feature != 'Mortality':
            cold_means[feature] = data_df.loc[data_df['County Category'] == 'Cold', f'{year} {feature} Rates'].mean()
    
    # Sort features based on 'Hot' means
    sorted_features = sorted(hot_means, key=hot_means.get, reverse=True)

    # Define the order and colors of the boxplot categories
    category_order = ['Hot', 'Cold', 'Other']
    category_colors = {'Hot': 'red', 'Cold': 'blue', 'Other': 'green'}

    # Determine the number of rows and columns needed for subplots
    num_features = len(sorted_features)
    num_cols = 5
    num_rows = (num_features + num_cols - 1) // num_cols

    # Initialize subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))  # Adjust figsize accordingly

    # Flatten the axes array for easy indexing
    axs = axs.flatten()

    # Create a boxplot for each feature
    for idx, variable in enumerate(sorted_features):
        sns.boxplot(x='County Category', y=f'{year} {variable} Rates', data=data_df, 
                    hue='County Category', order=category_order, palette=category_colors, ax=axs[idx],
                    whis=[0, 100], dodge=False)
        axs[idx].set_title(f'{variable}')
        axs[idx].set_ylabel(f'{variable} rates')

    # Hide any extra subplots if the number of features is not a multiple of num_cols
    for idx in range(num_features, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'Anomalies/Boxplots/{year}_boxplots.png', bbox_inches='tight')
    plt.close()
    print(f'{year} boxplots printed.')

    return hot_means, cold_means

def hot_anomaly_summary(hot_means_by_year):
    # Convert the collected means to a DataFrame
    hot_means_df = pd.DataFrame(hot_means_by_year)

    # Calculate the average mean across all years for each variable
    hot_means_df['Average'] = hot_means_df.mean(axis=1)

    # Sort the DataFrame by the average mean
    hot_means_df = hot_means_df.sort_values(by='Average', ascending=True)

    # Print out the calculated means
    hot_means_df_flipped_for_printing = hot_means_df.sort_values(by='Average', ascending=False)
    logging.info(f"Average means in hot counties:")
    for feature, mean in hot_means_df_flipped_for_printing['Average'].items():
        logging.info(f"{feature}: {mean:.2f}")

    # Color the bars on the importance plot
    num_years = len(hot_means_by_year)
    colors = list(plt.cm.tab20.colors[:num_years]) + ['black']  # Add black for the 'Average' column

    # Plot the means over the years
    ax = hot_means_df.plot(kind='barh', figsize=(12, 8), legend=True, color=colors)

    plt.title('Mean Rates of SVI Variables in the Hot Counties', fontweight='bold')
    plt.xlabel('Mean Value', fontweight='bold')
    plt.legend(title='Year', bbox_to_anchor=(1, 0), loc='lower right')
    plt.tight_layout()
    plt.savefig('Feature Importance/hot_anomaly_summary.png', bbox_inches='tight')
    plt.close()

def cold_anomaly_summary(cold_means_by_year):
    # Convert the collected means to a DataFrame
    cold_means_df = pd.DataFrame(cold_means_by_year)

    # Calculate the average mean across all years for each variable
    cold_means_df['Average'] = cold_means_df.mean(axis=1)

    # Sort the DataFrame by the average mean
    cold_means_df = cold_means_df.sort_values(by='Average', ascending=False)

    # Print out the calculated means
    cold_means_df_flipped_for_printing = cold_means_df.sort_values(by='Average', ascending=True)
    logging.info(f"\nAverage means in cold counties:")
    for feature, mean in cold_means_df_flipped_for_printing['Average'].items():
        logging.info(f"{feature}: {mean:.2f}")

    # Color the bars on the importance plot
    num_years = len(cold_means_by_year)
    colors = list(plt.cm.tab20.colors[:num_years]) + ['black']  # Add black for the 'Average' column

    # Plot the means over the years
    ax = cold_means_df.plot(kind='barh', figsize=(12, 8), legend=True, color=colors)

    plt.title('Mean Rates of SVI Variables in the Cold Counties', fontweight='bold')
    plt.xlabel('Mean Value', fontweight='bold')
    plt.legend(title='Year', bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout()
    plt.savefig('Feature Importance/cold_anomaly_summary.png', bbox_inches='tight')
    plt.close()

def main():
    hot_means_by_year = {}
    cold_means_by_year = {}
    data_df = construct_data_df()

    for year in range(2010, 2023):
        hot_means, cold_means = boxplots(data_df, year)
        hot_means_by_year[year] = hot_means
        cold_means_by_year[year] = cold_means

    hot_anomaly_summary(hot_means_by_year)
    cold_anomaly_summary(cold_means_by_year)

if __name__ == "__main__":
    main()