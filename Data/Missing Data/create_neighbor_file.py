import geopandas as gpd
import pandas as pd

# Load the shapefile
shapefile = gpd.read_file('2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp')

# Initialize an empty dictionary to store neighbors
neighbors = {}

# Iterate through each county
for index, row in shapefile.iterrows():
    county_geom = row['geometry']
    county_name = row['FIPS']
    neighbors[county_name] = []
    
    # Check for neighbors
    for idx, test_county in shapefile.iterrows():
        if county_geom.touches(test_county['geometry']):
            neighbors[county_name].append(test_county['FIPS'])

# Convert to DataFrame
data = []
for county, neighbor_list in neighbors.items():
    for neighbor in neighbor_list:
        data.append({'County': county, 'Neighbor': neighbor})

neighbors_df = pd.DataFrame(data)

# Group by 'County' and aggregate neighbors into a list
neighbors_df = neighbors_df.groupby('County')['Neighbor'].apply(lambda x: ','.join(x.astype(str))).reset_index()

# Save to CSV
neighbors_df.to_csv('county_neighbors.csv', index=False)
