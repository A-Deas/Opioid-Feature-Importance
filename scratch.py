import geopandas as gpd
import pandas as pd

shapefile_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
shape = gpd.read_file(shapefile_path)

neighs_df = pd.read_csv('CDC Wonder/Neighbor Files/2022_neighbors_list.csv', header=0, names=['FIPS', 'Neighbors'])

# Ensure FIPS codes are strings and padded to 5 characters
shape['FIPS'] = shape['FIPS'].astype(str).apply(lambda x: x.zfill(5))
neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).apply(lambda x: x.zfill(5))

# Extract FIPS codes
shape_fips = set(shape['FIPS'])
neighs_fips = set(neighs_df['FIPS'])

# Find the difference between the two sets
missing_in_neighbors = shape_fips - neighs_fips

# Print the differences
print("Missing FIPS:")
print(missing_in_neighbors)
