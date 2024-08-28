import pandas as pd
import geopandas as gpd

RAW_MORTALITY_NAMES = ['FIPS', 'Deaths', 'Population', 'Crude Rate']
COLUMNS_TO_KEEP = ['FIPS', 'Deaths', 'Population']
MISSING_VALUE = -9

def construct_paths(year):
    input_path = f'CDC Wonder/Raw Files/{year}_cdc_wonder_raw_mortality.csv'
    output_path = f'CDC Wonder/Filtered Files/{year}_mortality_filtered.csv'
    return input_path, output_path

def clean_rates(year, input_path):
    mort_df = pd.read_csv(input_path, header=0, names=RAW_MORTALITY_NAMES)
    
    # Format the columns
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df['Deaths'] = mort_df['Deaths'].astype(str) # need to be strings for now b/c they contain "suppressed/missing" values
    mort_df['Population'] = mort_df['Population'].astype(str) # these also contain "suppressed/missing" values

    # Keep only the required columns
    mort_df = mort_df[COLUMNS_TO_KEEP]
    
    # Replace "Supressed/Missing/Not Available" values with -9
    mort_df['Deaths'] = mort_df['Deaths'].replace(['Suppressed', 'Missing', 'Not Available'], MISSING_VALUE)
    mort_df['Population'] = mort_df['Population'].replace(['Suppressed', 'Missing', 'Not Available'], MISSING_VALUE)
    
    # Convert 'Deaths' and 'Population' columns to float and int respectively
    mort_df['Deaths'] = mort_df['Deaths'].astype(float)
    mort_df['Population'] = mort_df['Population'].astype(int)
    
    # Create a new column for the mortality rates 
    mort_df[f'{year} MR'] = mort_df.apply(
        lambda row: MISSING_VALUE if row['Deaths'] == MISSING_VALUE or row['Population'] <= 0 else (row['Deaths'] / row['Population']) * 100000,
        axis=1
    )

    # Round the mortality rate column to 2 decimal places
    mort_df[f'{year} MR'] = mort_df[f'{year} MR'].round(2)
    return mort_df

def load_shapefile():
    shapefile_path = f'2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    shape = gpd.read_file(shapefile_path)
    return shape

def filter_fips_codes(year, mort_df, shape, output_path):
    # Extract FIPS codes from both DataFrames
    data_fips = mort_df['FIPS']
    shape_fips = shape['FIPS'] # there are 3144 counties in the shapefile

    # Keep only the counties in the data that are also in the shape 
    # (counties in data but not in shape won't show up on a map)
    filtered_df = mort_df[mort_df['FIPS'].isin(shape_fips)].reset_index(drop=True)

    # Counties in shape but not in data
    # (counties that will show up on the map but for which we don't have data)
    missing_data = shape_fips[~shape_fips.isin(data_fips)]

    # Add missing counties to the data with Mortality Rates set to -9
    missing_df = pd.DataFrame({'FIPS': missing_data, f'{year} MR': -9})
    filtered_df = pd.concat([filtered_df, missing_df], ignore_index=True)
    filtered_df = filtered_df.sort_values(by='FIPS').reset_index(drop=True)

    # Save the final filtered result
    filtered_df.to_csv(output_path, index=False)
    print(f'{year} filtered rates saved.')
    return filtered_df

def main():
    for year in range(2010,2023):
        input_path, output_path = construct_paths(year)
        mort_df = clean_rates(year, input_path)
        shape = load_shapefile()
        filtered_df = filter_fips_codes(year, mort_df, shape, output_path)

if __name__ == "__main__":
    main()