import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import seaborn as sns

DATA = ['Mortality', 
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
        # 'Disability', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployed']
TAIL = 3

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
        variable_df['FIPS'] = variable_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
        variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float)

        if data_df.empty:
            data_df = variable_df
        else:
            data_df = pd.merge(data_df, variable_df, on='FIPS', how='outer')

    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

def boxplots(data_df, year):
    # Calculate the year-over-year difference in mortality rates BEFORE THE INSETS
    current_mort_rates = data_df[f'{year} Mortality Rates'].values
    previous_mort_rates = data_df[f'{year-1} Mortality Rates'].values
    differences = current_mort_rates - previous_mort_rates
    data_df['Differences'] = differences

    # Fit a normal distribution to the differences
    mean, std_dev = stats.norm.fit(differences)

    # Initialize county categories
    data_df['County Category'] = 'Other'

    # Calculate the upper and lower thresholds for anomalies
    tail = TAIL / 100
    upper_threshold = stats.norm.ppf(1-tail, mean, std_dev)
    lower_threshold = stats.norm.ppf(tail, mean, std_dev)

    data_df.loc[(data_df[f'{year} Mortality Rates'] > upper_threshold), 'County Category'] = 'Hot'
    data_df.loc[(data_df[f'{year} Mortality Rates'] < lower_threshold), 'County Category'] = 'Cold'

    # Calculate mean for 'Hot' category for each feature, excluding 'Mortality'
    hot_means = {}
    for feature in DATA:
        if feature != 'Mortality':
            hot_means[feature] = data_df.loc[data_df['County Category'] == 'Hot', f'{year-1} {feature} Rates'].mean()

    return hot_means

def main():
    means_by_year = {}
    data_df = construct_data_df()

    for year in range(2011, 2023):
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

    plt.title('Mean Value in Hot Counties', fontweight='bold')
    plt.xlabel('Mean Value', fontweight='bold')
    plt.legend(title='Year', bbox_to_anchor=(1, 0), loc='lower right')
    plt.tight_layout()
    plt.savefig('Feature Importance/anomaly_investigation_plot.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
