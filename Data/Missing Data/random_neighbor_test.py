import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import random

# Constants
SHAPE_PATH = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
NEIGHBORS_PATH = 'county_neighbors.csv'
NEIGHBORS_NAMES = ['FIPS', 'Neighbor List']

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_neighbor_file():
    neighs_df = pd.read_csv(NEIGHBORS_PATH, header=0, names=NEIGHBORS_NAMES)
    neighs_df['FIPS'] = neighs_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    neighs_df = neighs_df.sort_values(by='FIPS').reset_index(drop=True)
    return neighs_df

def merge_data_shape(shape, neighs_df):
    return shape.merge(neighs_df, on='FIPS')

def construct_map(shape):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'Random Neighbor Testing'
    plt.title(title, size=16, weight='bold')

    # Alaska and Hawaii insets
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4]) 
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])  
    
    # Plot state boundaries
    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)

    alaska_state = state_boundaries[state_boundaries['STATEFP'] == '02']
    alaska_state.boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)

    hawaii_state = state_boundaries[state_boundaries['STATEFP'] == '15']
    hawaii_state.boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]

    # Select a random row
    random_row = shape.sample(1).iloc[0]
    random_county = random_row['FIPS']
    neighbors = random_row['Neighbor List'].split(',')

    # Color the maps
    for inset, ax, _ in shapes:
        if random_county in inset['FIPS'].values:
            inset[inset['FIPS'] == random_county].plot(ax=ax, color='red')

        # just need to make sure we plot neighbors on the same inset since I have three on each map
        inset_neighbors = [neigh for neigh in neighbors if neigh in inset['FIPS'].values]
        if inset_neighbors:
            inset[inset['FIPS'].isin(inset_neighbors)].plot(ax=ax, color='blue')

    # Adjust the viewing
    set_view_window(main_ax, alaska_ax, hawaii_ax)

    plt.show()
    plt.close(fig)

def set_view_window(main_ax, alaska_ax, hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def max_neighbors(neighs_df):
    neighs_df['Neighbor Count'] = neighs_df['Neighbors List'].apply(lambda x: len(x.split(',')))
    max_count = neighs_df['Neighbor Count'].max()
    county_with_max_neighbors = neighs_df.loc[neighs_df['Neighbor Count'] == max_count, 'FIPS'].values[0]
    return county_with_max_neighbors, max_count

def main():
    shape = load_shapefile(SHAPE_PATH)
    neighs_df = load_neighbor_file()
    shape = merge_data_shape(shape, neighs_df)
    construct_map(shape)

    # Get the county with the maximum number of neighbors
    county, max_count = max_neighbors(neighs_df)
    print(f"The county with the maximum number of neighbors is {county} with {max_count} neighbors.")

if __name__ == "__main__":
    main()
