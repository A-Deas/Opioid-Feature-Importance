import logging
import pandas as pd
import geopandas as gpd

# Constants
SVI_VAR_LIST = {
    'EPL_POV': 'Below Poverty',
    'EPL_UNEMP': 'Unemployed',
    'EPL_NOHSDP': 'No High School Diploma',
    'EPL_AGE65': 'Aged 65 or Older',
    'EPL_AGE17': 'Aged 17 or Younger',
    # 'EPL_DISABL': 'Disability',
    'EPL_SNGPNT': 'Single-Parent Household',
    'EPL_MINRTY': 'Minority Status',
    'EPL_LIMENG': 'Limited English Ability',
    'EPL_MUNIT': 'Multi-Unit Structures',
    'EPL_MOBILE': 'Mobile Homes',
    'EPL_CROWD': 'Crowding',
    'EPL_NOVEH': 'No Vehicle',
    'EPL_GROUPQ': 'Group Quarters'
}
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
PATH_2010 = 'Data/SVI/Raw Files/SVI_2010_US_county.csv'
PATH_2014 = 'Data/SVI/Raw Files/SVI_2014_US_county.csv'
PATH_2016 = 'Data/SVI/Raw Files/SVI_2016_US_county.csv'
PATH_2018 = 'Data/SVI/Raw Files/SVI_2018_US_county.csv'
PATH_2020 = 'Data/SVI/Raw Files/SVI_2020_US_county.csv'
PATH_2022 = 'Data/SVI/Raw Files/SVI_2022_US_county.csv'
MISSING_DATA_VALUE = -9

log_file = 'Log Files/svi_cleaning.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def load_dataframes():
    # Load 2010
    # 2010 had a different labeling scheme that we need to account for
    df_2010 = pd.read_csv(PATH_2010)
    df_2010['FIPS'] = df_2010['FIPS'].astype(str).str.zfill(5)  # Convert FIPS to strings and zero-pad from the left
    for var, _ in SVI_VAR_LIST.items():
        incorrect_name = var.replace('EPL', 'E_PL') # create the 2010 name from the correct name

        # Some variables we need to change more specifically
        if var == 'EPL_NOHSDP':
            incorrect_name = 'E_PL_NOHSDIP'
        if var == 'EPL_AGE65':
            incorrect_name = 'PL_AGE65' # No 'estimates' for this in 2010, just exact percentiles
        if var == 'EPL_AGE17':
            incorrect_name = 'PL_AGE17' # No 'estimates' for this in 2010, just exact percentiles
        if var == 'EPL_SNGPNT':
            incorrect_name = 'PL_SNGPRNT' # No 'estimates' for this in 2010, just exact percentiles, also acronym change
        if var == 'EPL_MINRTY':
            incorrect_name = 'PL_MINORITY' # No 'estimates' for this in 2010, just exact percentiles, also acronym change
        if var == 'EPL_GROUPQ':
            incorrect_name = 'PL_GROUPQ' # No 'estimates' for this in 2010, just exact percentiles

        df_2010.rename(columns={incorrect_name: var}, inplace=True)

    # Load 2014
    df_2014 = pd.read_csv(PATH_2014)
    df_2014['FIPS'] = df_2014['FIPS'].astype(str).str.zfill(5)

    # Load 2016
    df_2016 = pd.read_csv(PATH_2016)
    df_2016['FIPS'] = df_2016['FIPS'].astype(str).str.zfill(5)

    # Load 2018
    df_2018 = pd.read_csv(PATH_2018)
    df_2018['FIPS'] = df_2018['FIPS'].astype(str).str.zfill(5)

    # Load 2020
    df_2020 = pd.read_csv(PATH_2020)
    df_2020['FIPS'] = df_2020['FIPS'].astype(str).str.zfill(5)
    df_2020.rename(columns={'EPL_POV150': 'EPL_POV'}, inplace=True) # variable name changed to POV150 in 2020

    # Load 2022
    df_2022 = pd.read_csv(PATH_2022)
    df_2022['FIPS'] = df_2022['FIPS'].astype(str).str.zfill(5)
    df_2022.rename(columns={'EPL_POV150': 'EPL_POV'}, inplace=True)
    return df_2010, df_2014, df_2016, df_2018, df_2020, df_2022

def construct_var_dataframe(df_2010, df_2014, df_2016, df_2018, df_2020, df_2022, var):
    # Start by grabbing 2010 data
    variable_df = df_2010[['FIPS', var]].copy() # we just copy the FIPS and variable columns
    variable_df.rename(columns={var: f'2010 {var}'}, inplace=True) # need to account for the year

    # Merge the data for the other years one year at a time
    # The FIPS column will progressively accumulate all FIPS codes up to the yearly merge
    # If a FIPS code was present in a previous year but not in the current year, the corresponding variable value for that year will be NaN
    # If a FIPS code is present in the current year but not in a previous year, it simply gets added to the list with its data
    dummy_df = df_2014[['FIPS', var]].copy()
    dummy_df.rename(columns={var: f'2014 {var}'}, inplace=True)
    variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='left')

    dummy_df = df_2016[['FIPS', var]].copy()
    dummy_df.rename(columns={var: f'2016 {var}'}, inplace=True)
    variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='left')

    dummy_df = df_2018[['FIPS', var]].copy()
    dummy_df.rename(columns={var: f'2018 {var}'}, inplace=True)
    variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='left')

    dummy_df = df_2020[['FIPS', var]].copy()
    dummy_df.rename(columns={var: f'2020 {var}'}, inplace=True)
    variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='left')

    dummy_df = df_2022[['FIPS', var]].copy()
    dummy_df.rename(columns={var: f'2022 {var}'}, inplace=True)
    variable_df = pd.merge(variable_df, dummy_df, on='FIPS', how='left')

    # If we accumulated some NaN values, switch them to the missing data value
    variable_df.fillna(-9, inplace=True)

    # Set the FIPS codes to be string values
    variable_df['FIPS'] = variable_df['FIPS'].astype(str).str.zfill(5)
    return variable_df

def impute_data(variable_df, var):
    # Impute data for the missing years
    variable_df[f'2011 {var}'] = variable_df[f'2010 {var}'] + 0.25 * (variable_df[f'2014 {var}'] - variable_df[f'2010 {var}'])
    variable_df[f'2012 {var}'] = variable_df[f'2010 {var}'] + 0.50 * (variable_df[f'2014 {var}'] - variable_df[f'2010 {var}'])
    variable_df[f'2013 {var}'] = variable_df[f'2010 {var}'] + 0.75 * (variable_df[f'2014 {var}'] - variable_df[f'2010 {var}'])

    variable_df[f'2015 {var}'] = variable_df[f'2014 {var}'] + 0.50 * (variable_df[f'2016 {var}'] - variable_df[f'2014 {var}'])
    variable_df[f'2017 {var}'] = variable_df[f'2016 {var}'] + 0.50 * (variable_df[f'2018 {var}'] - variable_df[f'2016 {var}'])
    variable_df[f'2019 {var}'] = variable_df[f'2018 {var}'] + 0.50 * (variable_df[f'2020 {var}'] - variable_df[f'2018 {var}'])
    variable_df[f'2021 {var}'] = variable_df[f'2020 {var}'] + 0.50 * (variable_df[f'2022 {var}'] - variable_df[f'2020 {var}'])
    return variable_df

def fix_rio_arriba(variable_df, var):
    # Fix the data error for Rio Arriba NM in 2018
    variable_df.loc[variable_df['FIPS'] == '35039', f'2017 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.25 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    variable_df.loc[variable_df['FIPS'] == '35039', f'2018 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.50 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    variable_df.loc[variable_df['FIPS'] == '35039', f'2019 {var}'] = variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] + 0.75 * ( variable_df.loc[variable_df['FIPS'] == '35039', f'2020 {var}'] - variable_df.loc[variable_df['FIPS'] == '35039', f'2016 {var}'] )
    return variable_df

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
    return shape

def fix_fips(shape, variable_df, var):
    # Remove FIPS codes from data that do not exist in the shape
    # Remember that FIPS codes were accumulated over all years, so we may have a few of these
    # These are counties that have been removed from the national county structure by the year 2022
    # These counties are "excess" data which will not be plotted on any of the maps
    variable_df = variable_df[variable_df['FIPS'].isin(shape['FIPS'])] # Keep only counties that exist in the shape

    # Now we need to add any missing FIPS codes
    # These are counties which are in the shapefile, but not in the data
    # These counties will be plotted on every map, but do not have data
    missing_fips = shape[~shape['FIPS'].isin(variable_df['FIPS'])]['FIPS']
    missing_fips_list = missing_fips.to_list()
    logging.info(f'Missing data for variable {var}: {missing_fips_list}.')
    
    # Create a DataFrame with the missing FIPS codes 
    # Then fill in every year with the missing data value b/c these codes were not present during ANY year of data
    missing_df = pd.DataFrame({
        'FIPS': missing_fips
    })
    
    for yr in range(2010, 2023):
        missing_df[f'{yr} {var}'] = MISSING_DATA_VALUE

    # Add the missing data to the variable_df
    variable_df = pd.concat([variable_df, missing_df], ignore_index=True, sort=False)
    variable_df = variable_df.sort_values(by='FIPS').reset_index(drop=True)
    return variable_df

def save_final_rates(variable_df, var):
    plain_english_var = SVI_VAR_LIST.get(var, '')
    output_path = f'Data/SVI/Interim Files/{plain_english_var}_interim.csv'

    # Multiply values by 100 so that our rates are now between 0 and 100
    variable_df = variable_df.apply(
        lambda x: x.apply(lambda y: round(y * 100, 2) if y != MISSING_DATA_VALUE else y) 
        if x.name != 'FIPS' else x
    )

    # Make some technical adjustments for saving
    column_order = ['FIPS'] + [f'{year} {var}' for year in range(2010, 2023)] # Reorder the columns 
    variable_df = variable_df[column_order]

    # Add leading zeros to the FIPS codes
    variable_df['FIPS'] = variable_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)

    # Save the datafram to a csv
    variable_df.to_csv(output_path, index=False)
    logging.info(f'Rates saved for {var}.\n')

def main():
    df_2010, df_2014, df_2016, df_2018, df_2020, df_2022 = load_dataframes()
    for var, _ in SVI_VAR_LIST.items():
        variable_df = construct_var_dataframe(df_2010, df_2014, df_2016, df_2018, df_2020, df_2022, var)
        variable_df = impute_data(variable_df, var)
        variable_df = fix_rio_arriba(variable_df, var)
        shape = load_shapefile(SHAPE_PATH)
        variable_df = fix_fips(shape, variable_df, var)
        save_final_rates(variable_df, var)

if __name__ == "__main__":
    main()