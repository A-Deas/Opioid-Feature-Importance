import pandas as pd

FEATURE_LIST = ['Aged 17 or Younger', 'Aged 65 or Older', 'Below Poverty', 'Crowding', 
                # 'Disability', 
                'Group Quarters', 'Limited English Ability', 'Minority Status', 'Mobile Homes', 
                'Multi-Unit Structures', 'No High School Diploma', 'No Vehicle', 
                'Single-Parent Household', 'Unemployed']

def load_neighbors():
    neighs_path = 'Data/Neighbors/2022_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )
    return neighs_df

def load_variable_rates(var):    
    variable_path = f'Data/SVI/Interim Files/{var}_interim.csv'
    variable_names = ['FIPS'] + [f'{year}' for year in range(2010, 2023)]
    variable_df = pd.read_csv(variable_path, header=0, names=variable_names)
    variable_df['FIPS'] = variable_df['FIPS'].astype(str).str.zfill(5)
    variable_df[variable_names[1:]] = variable_df[variable_names[1:]].astype(float)
    variable_df = variable_df.sort_values(by='FIPS').reset_index(drop=True)
    return variable_df

def fill_missing_neighbors(variable_df, neighs_df, year, num_missing):
    step_count = 0
    for fips, row in variable_df.iterrows():
        if row[f'{year}'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
            neighbors = neighbors.values[0]
            available_neighbors = [neighbor for neighbor in neighbors if neighbor in variable_df.index and variable_df.loc[neighbor, f'{year}'] != -9]
            missing_neighbors = [neighbor for neighbor in neighbors if neighbor in variable_df.index and variable_df.loc[neighbor, f'{year}'] == -9]

            if len(missing_neighbors) == num_missing and len(available_neighbors) > 0:
                new_value = sum([variable_df.loc[neighbor, f'{year}'] for neighbor in available_neighbors]) / len(available_neighbors)
                variable_df.loc[fips, f'{year}'] = new_value
                step_count += 1
    return variable_df, step_count

def fill_continental_holes(variable_df, neighs_df, year):
    # Fill in continental counties with no missing neighbors
    for fips, row in variable_df.iterrows():
        if row[f'{year}'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']  # list of neighbors for this county
            neighbors = neighbors.values[0]
            neighbor_rates = [variable_df.loc[neighbor, f'{year}'] for neighbor in neighbors if neighbor in variable_df.index and variable_df.loc[neighbor, f'{year}'] != -9]

            if len(neighbor_rates) == len(neighbors) and len(neighbor_rates) > 0:  # full neighbor set is available and not empty
                new_value = sum(neighbor_rates) / len(neighbor_rates)
                variable_df.at[fips, f'{year}'] = new_value
    return variable_df

def handle_island_counties(variable_df, year):
    island_fips = ['25019', '15003', '15007', '15001', '53055']
    # Deal with the island counties
    for fips in island_fips:
        variable_df.at[fips, f'{year}'] = -9.0  # Example: Keep them as missing, or you could apply another method
    return variable_df

def impute_rates(variable_df, neighs_df, var):
    output_path = f'Data/SVI/Final Files/{var}_final_rates.csv'
    variable_df = variable_df.set_index('FIPS')

    for year in range(2010, 2023):
        while True:
            count = 0

            # Fill in counties with exactly one, two, then three missing neighbors
            for num_missing in [1, 2, 3]:
                variable_df, step_count = fill_missing_neighbors(variable_df, neighs_df, year, num_missing)
                count += step_count

            # Once all categories have been properly dealt with, break the loop
            if count == 0:
                break

        # Final steps: fill in the continental holes (counties with no missing neighbors)
        variable_df = fill_continental_holes(variable_df, neighs_df, year)

        # Final steps: handle the islands
        # variable_df = handle_island_counties(variable_df, year)

        # Round the mortality rates to two decimal places
        variable_df[f'{year}'] = variable_df[f'{year}'].round(2)

    variable_df = variable_df.reset_index()
    variable_df.to_csv(output_path, index=False)
    print(f'{var} final rates saved.')

def main():
    neighs_df = load_neighbors()
    for var in FEATURE_LIST:
        variable_df = load_variable_rates(var)
        impute_rates(variable_df, neighs_df, var)

if __name__ == "__main__":
    main()