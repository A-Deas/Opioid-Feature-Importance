import pandas as pd

def load_neighbors():
    neighs_path = 'CDC Wonder/Neighbor Files/2022_neighbors_list.csv'
    neighs_names = ['FIPS', 'Neighbors']
    neighs_df = pd.read_csv(neighs_path, header=None, names=neighs_names)

    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    neighs_df['Neighbors'] = neighs_df['Neighbors'].apply(
        lambda x: x.split(',') if isinstance(x, str) and ',' in x else ([] if pd.isna(x) or x == '' else [x])
    )
    return neighs_df

def load_yearly_mortality(year):
    input_path = f'CDC Wonder/Filtered Files/{year}_mortality_filtered.csv'
    mort_names = ['FIPS', 'Deaths', 'Population', f'{year} MR']
    cols_to_keep = ['FIPS', f'{year} MR']

    mort_df = pd.read_csv(input_path, header=0, names=mort_names)
    mort_df = mort_df[cols_to_keep]
    
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].astype(float)
    return mort_df

def fill_missing_neighbors(mort_df, neighs_df, year, num_missing):
    step_count = 0
    for fips, row in mort_df.iterrows():
        if row[f'{year} MR'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']
            neighbors = neighbors.values[0]
            available_neighbors = [neighbor for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]
            missing_neighbors = [neighbor for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] == -9]

            if len(missing_neighbors) == num_missing and len(available_neighbors) > 0:
                new_value = sum([mort_df.loc[neighbor, f'{year} MR'] for neighbor in available_neighbors]) / len(available_neighbors)
                mort_df.loc[fips, f'{year} MR'] = new_value
                step_count += 1
    return mort_df, step_count

def fill_continental_holes(mort_df, neighs_df, year):
    # Fill in continental counties with no missing neighbors
    for fips, row in mort_df.iterrows():
        if row[f'{year} MR'] == -9.0:
            neighbors = neighs_df.loc[neighs_df['FIPS'] == fips, 'Neighbors']  # list of neighbors for this county
            neighbors = neighbors.values[0]
            neighbor_rates = [mort_df.loc[neighbor, f'{year} MR'] for neighbor in neighbors if neighbor in mort_df.index and mort_df.loc[neighbor, f'{year} MR'] != -9]

            if len(neighbor_rates) == len(neighbors) and len(neighbor_rates) > 0:  # full neighbor set is available and not empty
                new_value = sum(neighbor_rates) / len(neighbor_rates)
                mort_df.at[fips, f'{year} MR'] = new_value
    return mort_df

def handle_island_counties(mort_df, year):
    island_fips = ['25019', '15003', '15007', '15001', '53055']
    # Deal with the island counties
    for fips in island_fips:
        mort_df.at[fips, f'{year} MR'] = -9.0  # Example: Keep them as missing, or you could apply another method
    return mort_df

def clean_rates(mort_df, neighs_df, year):
    output_path = f'CDC Wonder/Final Files/{year}_mortality_final.csv'
    mort_df = mort_df.set_index('FIPS')

    while True:
        count = 0

        # Fill in counties with exactly one, two, then three missing neighbors
        for num_missing in [1, 2, 3]:
            mort_df, step_count = fill_missing_neighbors(mort_df, neighs_df, year, num_missing)
            count += step_count

        # Once all categories have been properly dealt with, break the loop
        if count == 0:
            break

    # Final steps: fill in the continental holes (counties with no missing neighbors)
    mort_df = fill_continental_holes(mort_df, neighs_df, year)

    # Final steps: handle the islands
    # mort_df = handle_island_counties(mort_df, year)

    # Round the mortality rates to two decimal places
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].round(2)

    mort_df = mort_df.reset_index()
    mort_df.to_csv(output_path, index=False)
    print(f'{year} final rates saved.')

def main():
    neighs_df = load_neighbors()
    for year in range(2010, 2023):
        mort_df = load_yearly_mortality(year)
        clean_rates(mort_df, neighs_df, year)

if __name__ == "__main__":
    main()