import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

################# Could add a legend to these maps !!!!!!!!!!!!

# Constants
SHAPE_PATH = '2020 USA County Shapefile/FIPS_usa.shp'
ANOMALY_PATH = 'Hybrid Model/Anomalies/anomalies.csv'
ANOMALY_NAMES = [f'{yr} Anomalies' for yr in range(2015, 2021)]
MORTALITY_PATH = 'Clean Data/Mortality rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{yr} Mortality rates' for yr in range(2014, 2021)]

def construct_output_map_path(year):
    output_map_path = f'Hybrid Model/Anomalies/Anomaly Maps/{year}_anomaly_map'
    return output_map_path

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_mort_and_fips():
    mort_df = pd.DataFrame()
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    fips_df = mort_df[['FIPS']]
    return mort_df, fips_df

def load_anomalies(fips_df):
    anom_df = pd.read_csv(ANOMALY_PATH, header=0, names=ANOMALY_NAMES)
    anom_df[ANOMALY_NAMES] = anom_df[ANOMALY_NAMES].astype(int)
    anom_df = pd.concat([fips_df, anom_df], axis=1)
    return anom_df

def merge_data_shape(shape, anom_df, mort_df):
    shape = shape.merge(anom_df, on='FIPS')
    shape = shape.merge(mort_df, on='FIPS')
    return shape

def plot_anomaly_map(shape, year, output_map_path):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'Anomaly Map for the Mortality Rates in {year}'
    plt.title(title, size=16, weight='bold')

    # Construct the map
    construct_map(shape, fig, main_ax, year)

    plt.savefig(output_map_path, bbox_inches=None, pad_inches=0, dpi=300)
    plt.show()
    plt.close(fig)

def construct_map(shape, fig, main_ax, year):
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

    # Color the maps
    mort_mu = shape[f'{year} Mortality rates'].mean()
    
    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            anom = row[f'{year} Anomalies']
            mort = row[f'{year} Mortality rates']
            if anom == 1:
                if mort > mort_mu:
                    inset[inset['FIPS'] == county].plot(ax=ax, color='red')
                elif mort < mort_mu:
                    inset[inset['FIPS'] == county].plot(ax=ax, color='blue')
            else: 
                inset[inset['FIPS'] == county].plot(ax=ax, color='lightgrey')

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_legend(main_ax)

def set_view_window(main_ax,alaska_ax,hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def add_legend(main_ax):
    red_patch = mpatches.Patch(color='red', label='Hot Anomaly')
    blue_patch = mpatches.Patch(color='blue', label='Cold Anomaly')
    main_ax.legend(handles=[red_patch, blue_patch], loc='lower right', bbox_to_anchor=(1.05, 0))
    
def main():
    for year in range(2015, 2021):
        output_map_path = construct_output_map_path(year)
        shape = load_shapefile(SHAPE_PATH)
        mort_df, fips_df = load_mort_and_fips()
        anom_df = load_anomalies(fips_df)
        shape = merge_data_shape(shape, anom_df, mort_df)
        plot_anomaly_map(shape, year, output_map_path)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()