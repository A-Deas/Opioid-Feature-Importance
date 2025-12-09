import pandas as pd
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

DATA = ['Mortality', 
        'Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
        'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
        'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
        'Single-Parent Household', 'Unemployment']
TAIL = 1

# Set up logging
log_file = 'Log Files/frozen_counties_importance_means.log'
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_file, mode='w'),
    logging.StreamHandler()
])

def construct_data_dfs():
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
        data_df = variable_df if data_df.empty else pd.merge(data_df, variable_df, on='FIPS', how='outer')
    data_df = data_df.sort_values(by='FIPS').reset_index(drop=True)
    return data_df

def compute_frozen_county_means(data_df, year):
    mort_rates = data_df[f'{year} Mortality Rates'].values

    # Initialize county categories
    data_df['County Category'] = 'Other'
    data_df.loc[data_df[f'{year} Mortality Rates'] == 0.0, 'County Category'] = 'Frozen'

    frozen_means = {}
    for feature in DATA:
        if feature != 'Mortality':
            frozen_means[feature] = data_df.loc[data_df['County Category'] == 'Frozen', f'{year} {feature} Rates'].mean()
    return frozen_means

def frozen_county_summary(frozen_means):
    df = pd.DataFrame(frozen_means)
    df['Average'] = df.mean(axis=1)
    df = df.sort_values(by='Average', ascending=False)
    logging.info("\nAverage means in frozen counties:")
    for feature, mean in df.sort_values(by='Average', ascending=True)['Average'].items():
        logging.info(f"{feature}: {mean:.2f}")
    return df

def importance_ranking(df, label):
    num_years = len(df.columns)
    colors = list(plt.cm.tab20.colors[:num_years-1]) + ['black']
    fig, ax = plt.subplots(figsize=(12, 8))
    variables = df.index
    years = df.columns
    bar_width = 0.6
    y_positions = np.arange(len(variables))
    for i, year in enumerate(years):
        ax.barh(y_positions - i * bar_width / num_years, df[year], 
                height=bar_width / num_years, label=year, color=colors[i])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables, fontsize=20)
    ax.set_xlabel('Mean Value', fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.set_title(f'Mean Rates of SVI Variables in the Zero Mortality Counties', fontsize=20, fontweight='bold')
    ax.legend(title='Year', fontsize=13, title_fontsize=13, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'Feature Importance/frozen_counties_summary.png', bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    frozen_means_by_year = {}
    data_df = construct_data_dfs()
    for year in range(2010, 2023):
        frozen_means = compute_frozen_county_means(data_df, year)
        frozen_means_by_year[year] = frozen_means
    df = frozen_county_summary(frozen_means_by_year)
    importance_ranking(df, 'Frozen')

if __name__ == "__main__":
    main()
