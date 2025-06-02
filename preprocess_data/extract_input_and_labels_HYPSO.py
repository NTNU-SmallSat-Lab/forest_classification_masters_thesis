# This script loads the hyperspectral data and labels, filters them based on the specified masks, and generates plots for visualization.

# ----------------------------------- Imports --------------------------------- #

import os
import numpy as np
from hypso import Hypso1
import rasterio
from hypso.resample import resample_dataarray_kd_tree_nearest
from hypso.geometry_definition import generate_swath_def
from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import LAND
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.ticker as ticker

# ------------------------- Global variables ------------------------- #

# Plotting
BLUE_IDX = 120 - 89 # Blue wavelength index
RED_IDX = 120 - 59
GREEN_IDX = 120 - 70
AOI_LONGITUDES = [9.910652026, 10.31926067, 10.31729069, 9.912950596, 9.910652026]
AOI_LATITUDES = [63.3014335, 63.29877609, 63.11933985, 63.11904485, 63.3014335]
PROJECTION = ccrs.PlateCarree()

# Create a custom colormap for the labels
viridis_colors = cm.viridis([0.2, 0.4, 0.6, 0.8])  # Choose evenly spaced colors from viridis
cmap_colors = list(viridis_colors)
CMAP4COLORS = ListedColormap(cmap_colors)
CMAP3COLORS = ListedColormap(cmap_colors[1:])  # Exclude the first color (0) for the relevant labels

# ------------------------- Data functions ------------------------- #


def load_coordinates(hypso_capture):
    """
    Load the coordinates from the hypso_capture directory.
    """
    try:        
        # Check for the presence of indirect georef files
        longitudes_file = f'{hypso_directory}/longitudes_indirectgeoref.dat'
        latitudes_file = f'{hypso_directory}/latitudes_indirectgeoref.dat'
        
        if not os.path.exists(longitudes_file):
            longitudes_file = f'{hypso_directory}/longitudes.dat'
            latitudes_file = f'{hypso_directory}/latitudes.dat'

        # Load lat/lon data
        lon_flat = np.fromfile(longitudes_file, dtype=np.float32)
        lat_flat = np.fromfile(latitudes_file, dtype=np.float32)

        if lon_flat.size == 598 * 1092 and lat_flat.size == 598 * 1092:
            longitudes = lon_flat.reshape((598, 1092))
            latitudes = lat_flat.reshape((598, 1092))
        elif lon_flat.size == 956 * 684 and lat_flat.size == 956 * 684:
            longitudes = lon_flat.reshape((956, 684))
            latitudes = lat_flat.reshape((956, 684))
        else:
            raise ValueError("Unexpected size of longitude or latitude data.")

        print(f"Loaded longitudes {longitudes.shape} and latitudes {latitudes.shape}")


        return longitudes, latitudes
        
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        return None, None

def load_hypso_reflectance(hypso_capture, longitudes, latitudes):
    """
    Load hyperspectral data from the hypso_capture directory.
    """
    try:
        hyperspectral_file = f'{hypso_directory}/{hypso_capture}-l1a.nc'

        # Load HYPSO-1 Capture
        satobj_h1 = Hypso1(path=hyperspectral_file, verbose=True) # L1a: Raw data
        satobj_h1.generate_l1b_cube() # L1b: TOA radiance
        satobj_h1._run_custom_georeferencing(latitudes, longitudes) # L1c: Georeferenced TOA radiance
        satobj_h1.generate_l1d_cube() # L1d: TOA reflectance with georeferencing
        # The message "[WARNING] Computing TOA reflectance using DIRECT georeferencing geometry." not followed by "[INFO] Running direct georeferencing..." indicates that the custom georeferencing with specified lat/lon is used as intended in hypso v2.1.2. 
        l1d_cube = satobj_h1.l1d_cube
        hypso_data = l1d_cube.transpose('band', 'y', 'x') # Rearrange to (bands, height, width)

        print(f"Hyperspectral data loaded successfully with shape: {hypso_data.shape}")

        return hypso_data
        
    except Exception as e:
        print(f"Error loading hyperspectral data: {e}")
        return None
    
def load_labels(label_file):
        
    try:    
        # Open the label raster file
        with rasterio.open(label_file) as src_labels:
            # Read label data
            original_labels = src_labels.read(1)
            label_transform = src_labels.transform

        # Generate coordinate grid for label raster
        label_longitudes, label_latitudes = np.meshgrid(
            np.arange(original_labels.shape[1]) * label_transform.a + label_transform.c,
            np.arange(original_labels.shape[0]) * label_transform.e + label_transform.f
        )

        original_labels = original_labels.astype(np.int8)

        print(f"Original labels loaded successfully with shape: {original_labels.shape}")

        return original_labels, label_longitudes, label_latitudes
    
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None, None
    
def apply_homogeneous_filter(labels, window_size=10):
    """
    Apply a filter to keep only pixels that are part of a homogeneous window (same label in a 10x10 area).
    Pixels that do not belong to such areas are set to -9999.
    """
    print(f"Applying homogeneous filter with window size {window_size}...")
    # Define a uniform filter that finds the majority class in the window
    uniform_filtered_labels = uniform_filter(labels, size=window_size, mode='constant', cval=-15)
    
    # Check if the entire window has the same label
    mask = (uniform_filtered_labels == labels)
    
    # Apply mask: keep original labels if homogeneous, otherwise set to -15
    labels_filtered = np.where(mask, labels, -15)
    
    return labels_filtered

def load_matching_labels(original_labels, label_longitudes, label_latitudes, longitudes, latitudes):
    """
    Load matching labels from the label file.
    """
    try:
        print(f"Generating swath definition with latitudes {latitudes.shape} and longitudes {longitudes.shape}")
        swath_def = generate_swath_def(latitudes, longitudes)
        print(f"Swath definition generated with shape: {swath_def.shape}")

        print(f"Resampling labels...")
        resampled_labels = resample_dataarray_kd_tree_nearest(swath_def, original_labels, label_latitudes, label_longitudes)

        print(f"Labels loaded successfully with shape: {resampled_labels.shape}")

        return resampled_labels
    except Exception as e:
        print(f"Error loading matching labels: {e}")
        return None
    
def generate_flat_valid_labels_mask(resampled_labels, valid_labels):
    """
    Generate flat mask which only includes valid labels.
    """
    try:
        resampled_labels_flat = resampled_labels.flatten()
        flat_valid_labels_mask = np.isin(resampled_labels_flat, valid_labels)

        
        print(f"Flat valid labels-mask generated with {np.sum(flat_valid_labels_mask)} valid pixels")

        return flat_valid_labels_mask
    
    except Exception as e:
        print(f"Error generating flat valid labels-mask: {e}")
        return None
    
def generate_flat_relevant_labels_mask(resampled_labels, relevant_labels):
    """
    Generate flat mask which only includes relevant labels.
    """
    try:
        resampled_labels_flat = resampled_labels.flatten()
        flat_relevant_labels_mask = np.isin(resampled_labels_flat, relevant_labels)

        
        print(f"Flat relevant labels-mask generated with {np.sum(flat_relevant_labels_mask)} relevant pixels")

        return flat_relevant_labels_mask
    
    except Exception as e:
        print(f"Error generating flat relevant labels-mask: {e}")
        return None
    
def generate_flat_land_mask(hypso_capture):
    """
    Generate flat mask which only includes land.
    """
    try:        
        sea_land_cloud_file = f'{hypso_directory}/{hypso_capture}.dat'

        sea_land_cloud_array = np.fromfile(sea_land_cloud_file, dtype=np.uint8)
        flat_land_mask = sea_land_cloud_array == 1  # Assuming land is represented by 1

        print(f"Flat land mask generated with {np.sum(flat_land_mask)} land pixels")

        return flat_land_mask
    
    except Exception as e:
        print(f"Error generating flat land mask: {e}")
        return None
    
def combine_masks(mask_list):
    """
    Combine the valid labels mask and the land mask.
    """
    try:
        # Combine the masks
        for i, mask in enumerate(mask_list):
            if i == 0:
                combined_mask = mask.copy()
            else:
                combined_mask &= mask.copy()

        print(f"Combined mask generated with {np.sum(combined_mask)} pixels")

        return combined_mask
    
    except Exception as e:
        print(f"Error combining masks: {e}")
        return None
    
def flatten_data(hypso_data, resampled_labels, longitudes, latitudes):
    try:
        # Convert hypso_data to a NumPy array if it's an xarray DataArray
        hypso_data_np = hypso_data.values if hasattr(hypso_data, "values") else hypso_data
        
        # Print original shapes
        print(f"Original data shapes: {hypso_data_np.shape}, {resampled_labels.shape}, {longitudes.shape}, {latitudes.shape}")

        # Flatten data
        bands, height, width = hypso_data_np.shape
        hypso_data_flat = hypso_data_np.reshape(bands, height * width)  # (bands, height * width)
        labels_flat = resampled_labels.ravel()
        lon_flat = longitudes.ravel()
        lat_flat = latitudes.ravel()

        # Print flattened shapes
        print(f"Flattened data shapes: {hypso_data_flat.shape}, {labels_flat.shape}, {lon_flat.shape}, {lat_flat.shape}")

        return hypso_data_flat, labels_flat, lon_flat, lat_flat

    except Exception as e:
        print(f"Error flattening data: {e}")
        return None, None, None, None

def filter_data_with_mask(hypso_data_flat, labels_flat, lon_flat, lat_flat, mask):
    try:
        # Apply the mask
        filtered_hypso_data_flat = hypso_data_flat[:, mask]  # (bands, selected pixels)
        filtered_labels_flat = labels_flat[mask]  # (selected pixels,)
        filtered_lon_flat = lon_flat[mask]  # (selected pixels,)
        filtered_lat_flat = lat_flat[mask]  # (selected pixels,)

        print(f"Filtered data shapes: {filtered_hypso_data_flat.shape}, {filtered_labels_flat.shape}, {filtered_lon_flat.shape}, {filtered_lat_flat.shape}")

        return filtered_hypso_data_flat, filtered_labels_flat, filtered_lon_flat, filtered_lat_flat

    except Exception as e:
        print(f"Error filtering data with mask: {e}")
        return None, None, None, None

# ------------------------------- Plotting functions --------------------------------- #

def plot_RGBs(hypso_data_flat, lon_flat, lat_flat, hypso_capture):

    if not os.path.exists(f"{hypso_directory}/Figures"):
        os.makedirs(f"{hypso_directory}/Figures")

    R_flat = hypso_data_flat[RED_IDX, :]
    G_flat = hypso_data_flat[GREEN_IDX, :]
    B_flat = hypso_data_flat[BLUE_IDX, :]

    RGB_bands_min = np.min([np.min(R_flat), np.min(G_flat), np.min(B_flat)])
    RGB_bands_max = np.max([np.max(R_flat), np.max(G_flat), np.max(B_flat)])

    # Normalize the bands to the range [0,1]
    R_flat_normalized = (R_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)
    G_flat_normalized = (G_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)
    B_flat_normalized = (B_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)

    colors = np.vstack((R_flat_normalized, G_flat_normalized, B_flat_normalized)).T

    colors = np.array(colors)

    brightness_factor = 2
    colors = np.clip(colors * brightness_factor, 0, 1)

    projection = ccrs.PlateCarree()

    fig, ax4 = plt.subplots(1, figsize=(8, 8), subplot_kw={'projection': projection})

    # Plot the B band with coastline
    ax4.set_title("RGB")
    ax4.set_xlabel("Longitude")
    ax4.set_extent([lon_flat.min(), lon_flat.max(), lat_flat.min(), lat_flat.max()])  # Set extent
    ax4.scatter(lon_flat, lat_flat, s=10, facecolors=colors)
    ax4.coastlines(resolution='10m', edgecolor='none')  # Add coastline
    ax4.add_feature(LAND, facecolor='lightgrey')  # Add land feature

    # Add ticks for latitude and longitude
    gl = ax4.gridlines(draw_labels=True, crs=projection, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{hypso_directory}/Figures/{hypso_capture}_RGB.png")
    plt.close(fig)  # Close the figure to prevent it from popping up

    # Set up the figure and map projection
    fig, ax4 = plt.subplots(1, figsize=(8, 8), subplot_kw={'projection': projection})

    # Plot RGB scatter points
    ax4.set_title("RGB with AOI")
    ax4.set_xlabel("Longitude")
    ax4.set_extent([lon_flat.min(), lon_flat.max(), lat_flat.min(), lat_flat.max()])  # Set map extent
    ax4.scatter(lon_flat, lat_flat, s=10, facecolors=colors)

    # Add coastline and land
    ax4.coastlines(resolution='10m', edgecolor='none')
    ax4.add_feature(LAND, facecolor='lightgrey')

    # Plot the polygon outline
    ax4.plot(AOI_LONGITUDES, AOI_LATITUDES, marker='o', linestyle='-', color='red', transform=ccrs.PlateCarree(), label="AOI")

    # Fill the polygon area
    ax4.fill(AOI_LONGITUDES, AOI_LATITUDES, color='red', alpha=0.3, transform=ccrs.PlateCarree())

    # Add gridlines with labels
    gl = ax4.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}

    plt.tight_layout()
    plt.savefig(f"{hypso_directory}/Figures/{hypso_capture}_RGB_AOI.png")
    plt.close(fig)  # Close the figure to prevent it from popping up

    return RGB_bands_max, RGB_bands_min

def plot_masked_RGB(masked_hypso_data, masked_lons, masked_lats, mask_name, RGB_bands_max, RGB_bands_min):

    # Extract RGB bands
    R_flat = masked_hypso_data[RED_IDX, :]
    G_flat = masked_hypso_data[GREEN_IDX, :]
    B_flat = masked_hypso_data[BLUE_IDX, :]

    R_flat_normalized = (R_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)
    G_flat_normalized = (G_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)
    B_flat_normalized = (B_flat - RGB_bands_min) / (RGB_bands_max - RGB_bands_min)

    colors = np.vstack((R_flat_normalized, G_flat_normalized, B_flat_normalized)).T

    colors = np.array(colors)

    brightness_factor = 2
    colors = np.clip(colors * brightness_factor, 0, 1)

    projection = ccrs.PlateCarree()

    fig, ax4 = plt.subplots(1, figsize=(8, 8), subplot_kw={'projection': projection})

    # Plot RGB scatter points
    ax4.set_title(f"{mask_name}-masked RGB in AOI")
    ax4.set_xlabel("Longitude")
    ax4.set_extent([min(AOI_LONGITUDES), max(AOI_LONGITUDES), min(AOI_LATITUDES), max(AOI_LATITUDES)])  # Set map extent
    ax4.scatter(masked_lons, masked_lats, s=30, facecolors=colors)

    # Add gridlines with labels
    gl = ax4.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}

    plt.tight_layout()
    plt.savefig(f"{hypso_directory}/Figures/{hypso_capture}_{mask_name}-masked_RGB_in_AOI.png")
    plt.close(fig)  # Close the figure to prevent it from popping up

def plot_masked_labels(masked_labels, masked_lons, masked_lats, mask_name):
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6), subplot_kw={'projection': PROJECTION})
    ax1.set_title(f"{mask_name}-masked resampled labels")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_extent([min(AOI_LONGITUDES), max(AOI_LONGITUDES), min(AOI_LATITUDES), max(AOI_LATITUDES)])  # Set map extent

    unique_labels = np.unique(masked_labels)
    if unique_labels.size == 3:
        cmap = CMAP3COLORS
        ticks = [1, 2, 3]
        tick_labels = ["1", "2", "3"]
    else:
        cmap = CMAP4COLORS
        ticks = [0, 1, 2, 3]
        tick_labels = ["0", "1", "2", "3"]

    # Define scatter plot
    sc = ax1.scatter(masked_lons, masked_lats, c=masked_labels, 
                    s=30, cmap=cmap, edgecolor='none', 
                    transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax1, orientation='vertical', pad=0.02)  # Adjust padding to avoid overlap
    cbar.set_label("Labels")

    # Set colorbar ticks to discrete values
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)  # Explicitly label them

    # Adjust y-axis tick formatting
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))  # Keep lat/lon readable

    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.right_labels = False  # Hide right-side labels if needed
    gl.top_labels = False    # Hide top labels

    # Save & show
    plt.savefig(f"{hypso_directory}/Figures/{hypso_capture}_{mask_name}-masked_resampled_labels.png", bbox_inches="tight")  # Prevent cropping issues
    plt.close(fig)  # Close the figure to prevent it from popping up

# ------------------------------- Main --------------------------------- #

def extract_hypso_data_and_labels(hypso_capture, hypso_directory, original_labels, label_longitudes, label_latitudes, relevant_labels, valid_labels):
    """
    Extract hyperspectral data and labels.
    """

    longitudes, latitudes = load_coordinates(hypso_capture)
    hypso_data = load_hypso_reflectance(hypso_capture, longitudes, latitudes)
    import xarray as xr

    print("\n===== DEBUG: Inputs before resampling =====")
    print(f"type(original_labels): {type(original_labels)}")
    print(f"original_labels shape: {original_labels.shape}")

    print(f"label_latitudes type: {type(label_latitudes)}, shape: {label_latitudes.shape}")
    print(f"label_longitudes type: {type(label_longitudes)}, shape: {label_longitudes.shape}")

    print(f"longitudes type: {type(longitudes)}, shape: {longitudes.shape}")
    print(f"latitudes type: {type(latitudes)}, shape: {latitudes.shape}")

    resampled_labels = load_matching_labels(original_labels, label_longitudes, label_latitudes, longitudes, latitudes)
    flat_valid_labels_mask = generate_flat_valid_labels_mask(resampled_labels, valid_labels)
    flat_land_mask = generate_flat_land_mask(hypso_capture)
    flat_relevant_labels_mask = generate_flat_relevant_labels_mask(resampled_labels, relevant_labels)
    masks = [flat_land_mask, flat_relevant_labels_mask, flat_valid_labels_mask]
    combined_mask = combine_masks(masks)
    hypso_data_flat, labels_flat, lon_flat, lat_flat = flatten_data(hypso_data, resampled_labels, longitudes, latitudes)
    valid_filtered_hypso_data_flat, valid_filtered_labels_flat, valid_filtered_lon_flat, valid_filtered_lat_flat = filter_data_with_mask(hypso_data_flat, labels_flat, lon_flat, lat_flat, flat_valid_labels_mask)
    land_filtered_hypso_data_flat, land_filtered_labels_flat, land_filtered_lon_flat, land_filtered_lat_flat = filter_data_with_mask(hypso_data_flat, labels_flat, lon_flat, lat_flat, flat_land_mask)
    relevant_filtered_hypso_data_flat, relevant_filtered_labels_flat, relevant_filtered_lon_flat, relevant_filtered_lat_flat = filter_data_with_mask(hypso_data_flat, labels_flat, lon_flat, lat_flat, flat_relevant_labels_mask)
    filtered_hypso_data_flat, filtered_labels_flat, filtered_lon_flat, filtered_lat_flat = filter_data_with_mask(hypso_data_flat, labels_flat, lon_flat, lat_flat, combined_mask)
    
    RGB_bands_max, RGB_bands_min = plot_RGBs(hypso_data_flat, lon_flat, lat_flat, hypso_capture)
    plot_masked_RGB(filtered_hypso_data_flat, filtered_lon_flat, filtered_lat_flat, "Combined", RGB_bands_max, RGB_bands_min)
    plot_masked_RGB(valid_filtered_hypso_data_flat, valid_filtered_lon_flat, valid_filtered_lat_flat, "Valid", RGB_bands_max, RGB_bands_min)    
    plot_masked_RGB(land_filtered_hypso_data_flat, land_filtered_lon_flat, land_filtered_lat_flat, "Land", RGB_bands_max, RGB_bands_min)
    plot_masked_RGB(relevant_filtered_hypso_data_flat, relevant_filtered_lon_flat, relevant_filtered_lat_flat, "Relevant", RGB_bands_max, RGB_bands_min)
    plot_masked_labels(filtered_labels_flat, filtered_lon_flat, filtered_lat_flat, "Combined")
    plot_masked_labels(valid_filtered_labels_flat, valid_filtered_lon_flat, valid_filtered_lat_flat, "Valid")
    print(f"Plots saved successfully in {hypso_directory}/Figures/")

    np.save(f"{hypso_directory}/relevant_hypso_data.npy", relevant_filtered_hypso_data_flat) # Filtered with only relevant mask
    np.save(f"{hypso_directory}/relevant_labels.npy", relevant_filtered_labels_flat)
    np.save(f"{hypso_directory}/filtered_hypso_data.npy", filtered_hypso_data_flat) # Filtered with combined mask for relevant, valid and land
    np.save(f"{hypso_directory}/filtered_labels.npy", filtered_labels_flat)
    print(f"Data saved successfully in {hypso_directory}")

if __name__ == "__main__":
    """
    Main entry point for the script.
    """

    label_file = "C:/Users/elise/Master/labels_reprojected_lat_lon_nores.tif"
    # label_file = "C:/Users/elise/Master/SR16_troendelag/sr16_50_SRRTRESLAG_EPSG4326.tif"
    hypso_captures = ['trondheim_2022-08-23T10-26-45Z']
    relevant_labels = [1, 2, 3] # 1: Spruce, 2: Pine, 3: Deciduous
    valid_labels = [0, 1, 2, 3] # 0: Other, 1: Spruce, 2: Pine, 3: Deciduous
    
    original_labels, label_longitudes, label_latitudes = load_labels(label_file)
    print(f"Original labels shape: {original_labels.shape}")
    labels, count = np.unique(original_labels, return_counts=True)
    print(f"Unique labels: {labels}")
    print(f"Label counts: {count}")
    print(f"Label latitudes shape: {label_latitudes.shape}")
    print(f"Label longitudes shape: {label_longitudes.shape}")
    filtered_labels= apply_homogeneous_filter(original_labels, window_size=10)
    print(f"Filtered labels shape: {filtered_labels.shape}")
    labels, count = np.unique(filtered_labels, return_counts=True)
    print(f"Unique labels: {labels}")
    print(f"Label counts: {count}")
    # plot_masked_labels(filtered_labels, label_longitudes, label_latitudes, "Filtered")

    # Loop through each hypso_capture
    for hypso_capture in hypso_captures:
        try:
            print("------------------------------------------")
            print(f"Processing {hypso_capture}...")
            hypso_directory = f'C:/Users/elise/OneDrive - NTNU/Documents/A.NTNU10/TEST/{hypso_capture}-l1a'
            extract_hypso_data_and_labels(hypso_capture, hypso_directory, filtered_labels, label_longitudes, label_latitudes, relevant_labels, valid_labels)
            print(f"Finished processing {hypso_capture}.\n")
        except Exception as e:
            print(f"Error processing {hypso_capture}: {e}")
        finally:
            pass
    print("------------------------------------------")
    print("All captures processed")

    