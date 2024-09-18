import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Constants
OLD_CT_FIPS = ['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015']
NEW_CT_FIPS = ['09110', '09120', '09130', '09140', '09150', '09160', '09170', '09180', '09190']

def load_2020_shapefile():
    shape_2020_path = '2020 USA County Shapefile/Filtered Files/2020_filtered_shapefile.shp'
    shape_2020 = gpd.read_file(shape_2020_path)
    shape_2020['FIPS'] = shape_2020['FIPS'].astype(str)

    shape_2020 = shape_2020[shape_2020['FIPS'].isin(OLD_CT_FIPS)]
    return shape_2020

def load_2022_shapefile():
    shape_2022_path = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
    shape_2022 = gpd.read_file(shape_2022_path)
    shape_2022['FIPS'] = shape_2022['FIPS'].astype(str)

    shape_2022 = shape_2022[shape_2022['FIPS'].isin(NEW_CT_FIPS)]
    return shape_2022

def load_raw_data(year):
    raw_path = f'Data/Mortality/Raw Files/{year}_cdc_wonder_raw_mortality.csv'
    raw_names = ['FIPS', f'{year} Deaths', f'{year} Pop', f'{year} MR']
    raw_df = pd.read_csv(raw_path, header=0, names=raw_names)

    raw_df['FIPS'] = raw_df['FIPS'].astype(str).str.zfill(5) 
    raw_df = raw_df[raw_df['FIPS'].isin(OLD_CT_FIPS)][['FIPS', f'{year} MR']]

    raw_df[f'{year} MR'] = pd.to_numeric(raw_df[f'{year} MR'], errors='coerce').fillna(0)

    raw_df = raw_df.sort_values(by='FIPS').reset_index(drop=True)
    return raw_df

def load_final_data(year):
    final_path = f'Data/Mortality/Final Files/Mortality_final_rates.csv'
    final_names = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
    final_df = pd.read_csv(final_path, header=0, names=final_names)
    final_df['FIPS'] = final_df['FIPS'].astype(str).str.zfill(5)
    final_df[final_names[1:]] = final_df[final_names[1:]].astype(float)

    final_df = final_df[final_df['FIPS'].isin(NEW_CT_FIPS)][['FIPS', f'{year} MR']]
    final_df = final_df.sort_values(by='FIPS').reset_index(drop=True)
    return final_df

def merge_shapes_with_data(shape_2020, shape_2022, final_df, raw_df):
    shape_2020 = shape_2020.merge(raw_df, on='FIPS')
    shape_2022 = shape_2022.merge(final_df, on='FIPS')
    return shape_2020, shape_2022

def plot_comparison_heat_map(shape_2020, shape_2022, year):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    title = f'Results of Connecticut Data Interpolation in {year}'
    plt.suptitle(title, size=15, weight='bold')

    old_data = shape_2020[f'{year} MR'].values
    new_data = shape_2022[f'{year} MR'].values

    cmap = plt.get_cmap('RdYlBu_r')

    # Create a common color normalization for both maps
    vmin = min(np.min(old_data), np.min(new_data))
    vmax = max(np.max(old_data), np.max(new_data))
    norm = BoundaryNorm(np.linspace(vmin, vmax, 21), cmap.N)

    # Plot the old data map
    shape_2020.plot(column=f'{year} MR', cmap=cmap, linewidth=0.8, ax=axes[0], edgecolor='0.8', norm=norm)
    axes[0].set_title(f'Old CT County Structure', fontsize=12)
    axes[0].axis('off')

    # Plot the new data map
    shape_2022.plot(column=f'{year} MR', cmap=cmap, linewidth=0.8, ax=axes[1], edgecolor='0.8', norm=norm)
    axes[1].set_title(f'New CT County Structure', fontsize=12)
    axes[1].axis('off')

    # Add colorbar with the maximum value explicitly displayed
    sm = ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    
    # Customize tick labels to include the max value
    tick_values = np.linspace(vmin, vmax, 5)  # Adjust number of ticks if needed
    tick_labels = [f'{val:.2f}' for val in tick_values]
    tick_labels[-1] = f'{vmax:.2f}'  # Add "Max: " to the final tick label
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)

    # Save or show the plot
    output_map_path = f'Heat Maps/Connecticut/{year}_ct_comparison.png'
    plt.savefig(output_map_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()
    plt.close(fig)

def main():
    for year in range(2010,2022):
        shape_2020 = load_2020_shapefile()
        shape_2022 = load_2022_shapefile()

        raw_df = load_raw_data(year)
        final_df = load_final_data(year)

        shape_2020, shape_2022 = merge_shapes_with_data(shape_2020, shape_2022, final_df, raw_df)

        plot_comparison_heat_map(shape_2020, shape_2022, year)
        print(f'Plot printed for {year}.')

if __name__ == "__main__":
    main()