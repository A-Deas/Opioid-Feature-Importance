import geopandas as gpd
import pandas as pd

# Constants
EXCLUDE_TERITORIES = ['03', '07', '14', '43', '52', '60', '69', '72', '78']
SHAPE_PATH = '/Users/deas/Documents/Research/2020 USA County Shapefile/FIPS_usa.shp'
NEIGHS_PATH = 'Dirty Data/2023 County Adjacencies Master File.csv'


""" Add some comments here maybe or figure out what some of those lines and
lambda calls that are confusing you are doing """

def preliminary_df(neighs_path, exclude_territories):
    prelim_df = pd.read_csv(neighs_path, header=0)
    prelim_df['County FIPS'] = prelim_df['County FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    prelim_df['Neighbor FIPS'] = prelim_df['Neighbor FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    prelim_df = prelim_df[['County FIPS', 'Neighbor FIPS']]
    prelim_df = prelim_df[~prelim_df['County FIPS'].str.startswith(tuple(exclude_territories))]
    prelim_df = prelim_df[~prelim_df['Neighbor FIPS'].str.startswith(tuple(exclude_territories))]
    return prelim_df

def first_neighbors(prelim_df):
    neighs_df = prelim_df[prelim_df['County FIPS'] != prelim_df['Neighbor FIPS']]
    neighs_df = neighs_df.groupby('County FIPS')['Neighbor FIPS'].apply(lambda x: ', '.join(x)).reset_index()
    neighs_df.columns = ['FIPS', '1st Neighbors']
    return neighs_df

def second_neighbors(neighs_df):
    neighs_df['1st Neighbors'] = neighs_df['1st Neighbors'].apply(lambda x: x.split(', '))
    neighs_df['2nd Neighbors'] = neighs_df['1st Neighbors'].apply(lambda neighbors: find_second_order_neighbors(neighs_df, neighbors))
    neighs_df['2nd Neighbors'] = neighs_df.apply(lambda row: list(set(row['2nd Neighbors']) - set(row['1st Neighbors']) - {row['FIPS']}), axis=1)
    return neighs_df

def third_neighbors(neighs_df):
    neighs_df['2nd Neighbors'] = neighs_df['2nd Neighbors'] # no need for .apply() here because haven't been saved yet and are therefore still a list
    neighs_df['3rd Neighbors'] = neighs_df['2nd Neighbors'].apply(lambda neighbors: find_third_order_neighbors(neighs_df, neighbors))
    neighs_df['3rd Neighbors'] = neighs_df.apply(lambda row: list(set(row['3rd Neighbors']) - set(row['2nd Neighbors']) - set(row['1st Neighbors']) - {row['FIPS']}), axis=1)
    return neighs_df

def find_second_order_neighbors(neighs_df, neighbors):
    second_order = []
    for neighbor in neighbors:
        second_order += neighs_df.loc[neighs_df['FIPS'] == neighbor, '1st Neighbors'].values[0]
    return second_order

def find_third_order_neighbors(neighs_df, neighbors):
    third_order = []
    for neighbor in neighbors:
        third_order += neighs_df.loc[neighs_df['FIPS'] == neighbor, '2nd Neighbors'].values[0]
    return third_order

def match_shapefile(shape_path, neighs_df):
    shape = gpd.read_file(shape_path)
    shape_fips = shape['FIPS'].astype(str)
    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str)

    missing_in_neighs = shape_fips[~shape_fips.isin(neighs_df['FIPS'])]
    if missing_in_neighs.empty:
        print("No missing FIPS codes in neighs_df from the shapefile.\n")
    else:
        print(f'FIPS codes missing in neighs_df (found in shapefile but not in neighs_df): {missing_in_neighs.tolist()}\n')

    # Add these missing FIPS to neighs_df with empty lists for 1st and 2nd neighbors
    new_rows = pd.DataFrame({
        'FIPS': missing_in_neighs,
        '1st Neighbors': ['' for _ in range(len(missing_in_neighs))],
        '2nd Neighbors': ['' for _ in range(len(missing_in_neighs))]
    })
    neighs_df = pd.concat([neighs_df, new_rows], ignore_index=True)

    # Find FIPS codes in neighs_df that are NOT in the shapefile
    extra_in_neighs = neighs_df['FIPS'][~neighs_df['FIPS'].isin(shape_fips)]
    if extra_in_neighs.empty:
        print("No extra FIPS codes in neighs_df not found in the shapefile.\n")
    else:
        print(f'Extra FIPS codes in neighs_df (found in neighs_df but not in shapefile): {extra_in_neighs.tolist()}\n')

    neighs_df = neighs_df[neighs_df['FIPS'].isin(shape_fips)]
    neighs_df = neighs_df.sort_values('FIPS')
    return neighs_df


    
def main():
    prelim_df = preliminary_df(NEIGHS_PATH, EXCLUDE_TERITORIES)
    neighs_df = first_neighbors(prelim_df)
    neighs_df = second_neighbors(neighs_df)
    #neighs_df = third_neighbors(neighs_df)
    #neighs_df = match_shapefile(SHAPE_PATH, neighs_df)
    neighs_df.to_csv('Clean Data/National County Neighbors.csv', index=False)

if __name__ == "__main__":
    main()





"""# Assuming 'neighbors_df' is the DataFrame and you want to access neighbors of a specific FIPS code
fips_code = '01001'  # Example FIPS code
neighbors_list = neighbors_df.loc[neighbors_df['FIPS'] == fips_code, 'Neighbors'].iloc[0].split(', ')

# Now 'neighbors_list' contains the neighbors as a list
print(neighbors_list)
"""