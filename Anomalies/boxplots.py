import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import seaborn as sns

DATA = ['Mortality', 
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployed']
# THRESHOLD = 2

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

def boxplots(data_df, year):
    mort_rates = data_df[f'{year} Mortality Rates'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]
    params_lognorm = lognorm.fit(non_zero_mort_rates)
    shape, loc, scale = params_lognorm

    # Initialize county categories
    data_df['County Category'] = 'Other'

    # Constants for thresholds
    hot_threshold = lognorm.ppf(.99, shape, loc, scale)
    cold_threshold = lognorm.ppf(.01, shape, loc, scale)

    data_df.loc[(data_df[f'{year} Mortality Rates'] > hot_threshold), 'County Category'] = 'Hot'
    data_df.loc[(data_df[f'{year} Mortality Rates'] < cold_threshold), 'County Category'] = 'Cold'

    # Calculate mean for 'Hot' category for each feature, excluding 'Mortality'
    hot_means = {}
    for feature in DATA:
        if feature != 'Mortality':
            hot_means[feature] = data_df.loc[data_df['County Category'] == 'Hot', f'{year} {feature} Rates'].mean()

    # Sort features based on 'Hot' means
    sorted_features = sorted(hot_means, key=hot_means.get, reverse=True)

    # Define the order and colors of the boxplot categories
    category_order = ['Hot', 'Cold', 'Other']
    category_colors = {'Hot': 'red', 'Cold': 'blue', 'Other': 'green'}

    # Determine the number of rows and columns needed for subplots
    num_features = len(sorted_features)
    num_cols = 7
    num_rows = (num_features + num_cols - 1) // num_cols

    # Initialize subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 7))  # Increased figure size for better spacing

    # Flatten the axes array for easy indexing
    axs = axs.flatten()

    # Create a boxplot for each feature
    for idx, feature in enumerate(sorted_features):
        sns.boxplot(x='County Category', y=f'{year} {feature} Rates', data=data_df, 
                    hue='County Category', palette=category_colors, ax=axs[idx],
                    whis=[0, 100], dodge=False)
        axs[idx].set_title(f'Boxplot for {feature}')
        axs[idx].set_ylabel(f'Rates')

    # Hide any extra subplots if the number of features is not a multiple of num_cols
    for idx in range(num_features, len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'Anomalies/Boxplots/{year}_boxplots.png', bbox_inches='tight')
    plt.close()

def main():
    for year in range(2014, 2021):
        data_df = construct_data_df()
        boxplots(data_df, year)

if __name__ == "__main__":
    main()