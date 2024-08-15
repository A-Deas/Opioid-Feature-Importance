import geopandas as gpd
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

# Constants
FEATURES = ['Mortality', 
            'DR', 
            'SVI Aged 17 or Younger', 'SVI Aged 65 or Older', 'SVI Below Poverty',
            'SVI Crowding', 'SVI Disability', 'SVI Group Quarters', 'SVI Limited English Ability',
            'SVI Minority Status', 'SVI Mobile Homes', 'SVI Multi-Unit Structures', 'SVI No High School Diploma',
            'SVI No Vehicle', 'SVI Single-Parent Household', 'SVI Unemployed']
DATA_NAMES = ['FIPS'] + [f'{yr} Data' for yr in range(2014, 2021)]

def get_mort_rates():
    mort_path = f'Clean Data/Mortality rates.csv'
    mort_names = ['FIPS'] + [f'{yr} Mortality' for yr in range(2014, 2021)]
    mort_df = pd.read_csv(mort_path, header=0, names=mort_names)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[mort_names[1:]] = mort_df[mort_names[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    return mort_df

def percentile_ranks(mort_df, year):
    yearly_mort = mort_df[f'{year} Mortality']
    mu, sigma = stats.norm.fit(yearly_mort)

    percentiles = []
    for i in range(5,101,5):
        perc = i / 100
        percentiles.append(perc)

    perct_ranks = []
    for point in yearly_mort:
        density_value = stats.norm.cdf(point, loc=mu, scale=sigma)

        """ Use the densities? 
                No, this completely dominates the model """
        #density_value = round(density_value*100,2)
        #perct_ranks.append(density_value)

        """ Use the percentile ranks? 
                No, also domintated the model but less so than
                the actual density values did """
        for i, perct in enumerate(percentiles):
                        if density_value <= perct:
                            perct_ranks.append(perct)
                            break

        """ Can we encode 'hot/cold' in some way? 
                WOW, this is too cool! This dominates as well but
                even less than the percentile ranks """
        #if density_value <= 0.25: # 25th percentile threshold for cold
        #    perct_ranks.append(-1)
        #elif 0.25 < density_value < 0.75:
        #    perct_ranks.append(0)
        #elif density_value >= 0.75: # 75th percentile threshold for hot
        #    perct_ranks.append(1)

        """ Let's try different less informative thresholds for hot/cold? 
                This is just incredible to see firsthand, even these
                very uninformative markers of being above or below
                the mean dominate the model significantly """
        #if density_value <= 0.50: # 25th percentile threshold for cold
        #    perct_ranks.append(-1)
        #elif density_value > 0.50:
        #    perct_ranks.append(1)

    mort_df[f'{year} MPR'] = perct_ranks
    return mort_df

def neighbor_ranks(mort_df, neighs_df):
    # Convert the string representation of lists in '1st Neighbors' to actual lists
    neighs_df['1st Neighbors'] = neighs_df['1st Neighbors'].apply(lambda x: eval(x) if isinstance(x, str) else [])

    # For each year, calculate the neighbor ranks
    for year in range(2014, 2021):
        column_name = f'{year} MPR'
        mort_df[f'{year} Neighbor Ranks'] = mort_df.apply(lambda row: calculate_neighbor_avg(row, neighs_df, mort_df, column_name), axis=1)

def calculate_neighbor_avg(row, neighs_df, mort_df, column_name):
    # Find the row for the current county in the neighbors DataFrame
    neighbors = neighs_df.loc[neighs_df['FIPS'] == row['FIPS'], '1st Neighbors'].values
    if len(neighbors) > 0 and len(neighbors[0]) > 0:
        neighbors = neighbors[0]
        # Select the rows of these neighbors in the mortality DataFrame
        neighbor_data = mort_df[mort_df['FIPS'].isin(neighbors)]
        # Include the current county's rank
        all_ranks = [row[column_name]] + neighbor_data[column_name].tolist()
        # Calculate the average rank
        return sum(all_ranks) / len(all_ranks)
    return row[column_name]  # Return the county's own rank if no neighbors found

def strip_ranks(mort_df):
    columns_to_keep = ['FIPS'] + [f'{year} MPR' for year in range(2014, 2021)]
    perct_ranks_df = mort_df[columns_to_keep]
    perct_ranks_df.to_csv('Clean Data/Mortality Percentile Ranks rates.csv', index=False)

def main():
    mort_df = get_mort_rates()
    neighbor_ranks(mort_df)
    for year in range(2014, 2021):
        mort_df = percentile_ranks(mort_df, year)
    strip_ranks(mort_df)

if __name__ == "__main__":
    main()