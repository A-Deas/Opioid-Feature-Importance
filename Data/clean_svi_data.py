import logging
import pandas as pd
import geopandas as gpd

# Constants
VARIABLES = {
    'EPL_POV': 'Below Poverty',
    'EPL_UNEMP': 'Unemployed',
    'EPL_NOHSDP': 'No High School Diploma',
    'EPL_AGE65': 'Aged 65 or Older',
    'EPL_AGE17': 'Aged 17 or Younger',
    'EPL_DISABL': 'Disability',
    'EPL_SNGPNT': 'Single-Parent Household',
    'EPL_MINRTY': 'Minority Status',
    'EPL_LIMENG': 'Limited English Ability',
    'EPL_MUNIT': 'Multi-Unit Structures',
    'EPL_MOBILE': 'Mobile Homes',
    'EPL_CROWD': 'Crowding',
    'EPL_NOVEH': 'No Vehicle',
    'EPL_GROUPQ': 'Group Quarters'
}
SHAPE_PATH = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
DATA_2014_PATH = 'Data/Dirty/SVI_2014_US_county.csv'
DATA_2016_PATH = 'Data/Dirty/SVI_2016_US_county.csv'
DATA_2018_PATH = 'Data/Dirty/SVI_2018_US_county.csv'
DATA_2020_PATH = 'Data/Dirty/SVI_2020_US_county.csv'
MISSING_DATA_VALUE = -9

log_file = 'Log Files/svi_cleaning.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S', handlers=[
    logging.FileHandler(log_file, mode='w'),  # Overwrite the log file
    logging.StreamHandler()
])

def get_output_path(category):
    plain_english_category = VARIABLES.get(category, 'Unspecified Category')
    output_path = f'Data/Clean/{plain_english_category}_rates.csv'
    return output_path

def construct_new_dataframe(data_2014_path, data_2016_path, data_2018_path, data_2020_path, category):
    data_2014 = pd.read_csv(data_2014_path)
    data_2016 = pd.read_csv(data_2016_path)
    data_2018 = pd.read_csv(data_2018_path)
    data_2020 = pd.read_csv(data_2020_path)
    data_2020.rename(columns={'EPL_POV150': 'EPL_POV'}, inplace=True) # Category name changed in 2020

    # Construct the new dataframe to edit
    new_dataframe = data_2014[['FIPS', category]].copy()
    new_dataframe.rename(columns={category: f'2014 {category}'}, inplace=True)

    # Merge the data for the other years
    temp_df = data_2016[['FIPS', category]].copy()
    temp_df.rename(columns={category: f'2016 {category}'}, inplace=True)
    new_dataframe = pd.merge(new_dataframe, temp_df, on='FIPS', how='left')

    temp_df = data_2018[['FIPS', category]].copy()
    temp_df.rename(columns={category: f'2018 {category}'}, inplace=True)
    new_dataframe = pd.merge(new_dataframe, temp_df, on='FIPS', how='left')

    temp_df = data_2020[['FIPS', category]].copy()
    temp_df.rename(columns={category: f'2020 {category}'}, inplace=True)
    new_dataframe = pd.merge(new_dataframe, temp_df, on='FIPS', how='left')

    # Fill in the rest of the dataframe with the synthetic data
    new_dataframe[f'2015 {category}'] = new_dataframe[f'2014 {category}'] + 0.5 * (new_dataframe[f'2016 {category}'] - new_dataframe[f'2014 {category}'])
    new_dataframe[f'2017 {category}'] = new_dataframe[f'2016 {category}'] + 0.5 * (new_dataframe[f'2018 {category}'] - new_dataframe[f'2016 {category}'])
    new_dataframe[f'2019 {category}'] = new_dataframe[f'2018 {category}'] + 0.5 * (new_dataframe[f'2020 {category}'] - new_dataframe[f'2018 {category}'])

    # Multiply values by 100 so that our rates are now between 0 and 100 (not between 0 and 1)
    new_dataframe = new_dataframe.apply(lambda x: (x*100).round(2) if x.name != 'FIPS' else x)

    # Reorder the columns 
    column_order = ['FIPS'] + [f'{year} {category}' for year in range(2014, 2021)]
    new_dataframe = new_dataframe[column_order]

    # Add leading zeros to the FIPS codes
    new_dataframe['FIPS'] = new_dataframe['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)

    # Finally, fix the data error for Rio Arriba NM in 2018
    new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2017 {category}'] = new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2016 {category}'] + 0.25 * ( new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2020 {category}'] - new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2016 {category}'] )
    new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2018 {category}'] = new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2017 {category}'] + 0.25 * ( new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2020 {category}'] - new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2016 {category}'] )
    new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2019 {category}'] = new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2018 {category}'] + 0.25 * ( new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2020 {category}'] - new_dataframe.loc[new_dataframe['FIPS'] == '35039', f'2016 {category}'] )
    return new_dataframe

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def fix_fips(shape, new_dataframe, category):
    new_dataframe = new_dataframe[new_dataframe['FIPS'].isin(shape['FIPS'])] # Remove codes that do not exist in the shape
    dropped_fips_codes = new_dataframe[~new_dataframe['FIPS'].isin(shape['FIPS'])]
    dropped_fips_codes_list = dropped_fips_codes['FIPS'].tolist()
    logging.info(f'Counties in {category} but not in shape (excess data): {dropped_fips_codes_list}.')

    missing_fips = shape[~shape['FIPS'].isin(new_dataframe['FIPS'])]['FIPS'] # Codes present in shape but not in new_dataframe
    missing_fips_list = missing_fips.to_list()
    logging.info(f'Counties in shape but not in {category} (missing data): {missing_fips_list}.\n')
    
    # Create a DataFrame with missing FIPS codes and fill all year-category columns with the missing_data_value
    missing_df = pd.DataFrame({
        'FIPS': missing_fips
    })
    
    for yr in range(2014, 2021):
        missing_df[f'{yr} {category}'] = MISSING_DATA_VALUE
    logging.info(f'Missing dataframe for {category}:')
    logging.info(missing_df)

    new_dataframe = pd.concat([new_dataframe, missing_df], ignore_index=True, sort=False)
    new_dataframe = new_dataframe.sort_values(by='FIPS').reset_index(drop=True)
    return new_dataframe

def save_new_dataframe(new_dataframe, output_path):
    new_dataframe.to_csv(output_path, index=False)

def main():
    for category, _ in VARIABLES.items():
        output_path = get_output_path(category)
        new_dataframe = construct_new_dataframe(DATA_2014_PATH, DATA_2016_PATH, DATA_2018_PATH, DATA_2020_PATH, category)
        shape = load_shapefile(SHAPE_PATH)
        new_dataframe_fixed = fix_fips(shape, new_dataframe, category)
        save_new_dataframe(new_dataframe_fixed, output_path)

if __name__ == "__main__":
    main()