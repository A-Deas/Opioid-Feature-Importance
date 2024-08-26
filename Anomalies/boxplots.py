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

    return hot_means

def main():
    means_by_year = {}
    data_df = construct_data_df()

    for year in range(2014, 2021):
        hot_means = boxplots(data_df, year)
        means_by_year[year] = hot_means

    # Convert the collected means to a DataFrame
    means_df = pd.DataFrame(means_by_year)

    # Calculate the average mean across all years for each variable
    means_df['Average'] = means_df.mean(axis=1)

    # Sort the DataFrame by the average mean
    means_df = means_df.sort_values(by='Average', ascending=True)

    # Use the 'Set1' colormap, which has distinct colors
    colors = plt.cm.tab10.colors[:len(means_df.columns)-1]  # Select colors for the years
    colors = list(colors) + ['black']  # Add black for the 'Average' column

    # Plot the means over the years
    ax = means_df.plot(kind='barh', figsize=(12, 8), legend=True, color=colors)

    plt.title('Mean Values of SVI Variable Rates in Hot Counties Over the Years', fontweight='bold')
    plt.xlabel('Mean Value', fontweight='bold')
    plt.ylabel('Variable', fontweight='bold')
    plt.legend(title='Year', bbox_to_anchor=(1, 0), loc='lower right')
    plt.tight_layout()
    plt.savefig('Feature Importance/anomaly_investigation_plot.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
