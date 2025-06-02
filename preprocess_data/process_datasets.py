import numpy as np
import os
from sklearn.model_selection import train_test_split

hypso_captures = ['trondheim_2024-09-14T09-42-48Z', 'trondheim_2024-04-29T09-44-07Z']

def load_and_combine_datasets(hypso_captures):
    """
    Load and combine datasets from the specified paths.
    
    """
    combination_name = ''

    for hypso_capture in hypso_captures:
        
        # hypso_directory = f'C:/Users/elise/OneDrive - NTNU/Documents/A.NTNU10/HYPSO data/{hypso_capture}-l1a/BiggerAOI'
        # hypso_directory = f'C:/Users/elise/OneDrive - NTNU/Documents/A.NTNU10/HYPSO data/{hypso_capture}-l1a/Test_filtered'
        hypso_directory = f'C:/Users/elise/OneDrive - NTNU/Documents/A.NTNU10/HYPSO data/{hypso_capture}-l1a/'

        data_path = f"{hypso_directory}/filtered_hypso_data.npy"
        labels_path= f"{hypso_directory}/filtered_labels.npy"

        # Check if the files exist
        try:
            data = np.load(data_path)
            labels = np.load(labels_path)
        except FileNotFoundError:
            print(f"File not found: {data_path} or {labels_path}")
            continue

        # Combine datasets
        if 'combined_data' in locals():
            combined_data = np.concatenate((combined_data, data), axis=1)
            combined_labels = np.concatenate((combined_labels, labels), axis=0)
        else:
            combined_data = data
            combined_labels = labels

        # Update the combination name
        combination_name += f"{hypso_capture}_"

        print(f"Loaded data with shape {data.shape} and labels with shape {labels.shape} for {hypso_capture}")
        
    print(f"Combined data shape {combined_data.shape} and labels shape {combined_labels.shape} for {combination_name}")

    return combined_data, combined_labels, combination_name


def save_dataset(data, labels, output_directory, save_name, bool_save_labels=True):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    data_path = f"{output_directory}/{save_name}_hypso_data.npy"
    np.save(data_path, data)

    if bool_save_labels:
        # Save labels only if bool_save_labels is True
        labels_path = f"{output_directory}/{save_name}_labels.npy"
        np.save(labels_path, labels)
        print(f"Labels saved to {labels_path}")

    print(f"Data saved to {data_path}")

def remove_bands(data, bands_to_remove):
    """
    Remove bands from the combined dataset.
    
    """
    # Remove the specified bands
    data_with_removed_bands = np.delete(data, bands_to_remove, axis=0)
    # Update the shape of the data
    print(f"Original data shape: {data.shape}")
    print(f"New data shape: {data_with_removed_bands.shape}")

    print(f"Removed bands {bands_to_remove} from data")

    return data_with_removed_bands

def select_bands(data, bands_to_select):
    """
    Select specific bands from the combined dataset.
    
    """
    # Select the specified bands
    data_with_selected_bands = data[bands_to_select, :]
    # Update the shape of the data
    print(f"Original data shape: {data.shape}")
    print(f"New data shape: {data_with_selected_bands.shape}")

    print(f"Selected bands {bands_to_select} from data")

    return data_with_selected_bands

def select_wavelengths(data, wavelengths_to_select):
    wavelength_file = "C:/Users/elise/Master/HYPSO/spectral_bands_HYPSO-1_v1.npz"
    # Satellite bands (wavelengths in nanometers)
    all_wavelengths = np.load(wavelength_file)["arr_0"]
    print(f"Number of bands: {len(all_wavelengths)}")

    # Find the indices of the nearest wavelengths
    wavelength_indices = np.argmin(np.abs(all_wavelengths[:, None] - np.array(wavelengths_to_select)[None, :]), axis=0)

    # Create the new array with selected wavelengths
    wavelengths = all_wavelengths[wavelength_indices]

    data_with_selected_bands = data[wavelength_indices, :]
    print("Indices:", wavelength_indices)
    print("Wavelengths:", wavelengths)

    return data_with_selected_bands

def train_test_split_dataset(data, labels):
    """
    Split the data into training and testing sets.
    
    """
    X_train, X_test, y_train, y_test = train_test_split(data.T, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    
    combined_data, combined_labels, combination_name = load_and_combine_datasets(hypso_captures)

    output_directory = f'C:/Users/elise/OneDrive - NTNU/Documents/A.NTNU10/HYPSO data/Combined/{combination_name}'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(f"{output_directory}/combined_data.npy", combined_data)
    np.save(f"{output_directory}/combined_labels.npy", combined_labels)

    operation = "remove"  # "select" or "remove"
    bands_to_remove = [0,1,2,3,4,5, 106, 107, 108, 109, 119, 118, 117]
    wavelengths_to_select = [450, 500, 550, 650, 700]
    
    if operation == "select":
        combined_data = select_wavelengths(combined_data, wavelengths_to_select)
        np.save(f"{output_directory}/combined_data_with_selected_bands.npy", combined_data)
    elif operation == "remove":
        combined_data = remove_bands(combined_data, bands_to_remove)
        np.save(f"{output_directory}/combined_data_with_removed_bands.npy", combined_data)

    X_train, y_train, X_test, y_test = train_test_split_dataset(combined_data, combined_labels)

    np.save(f"{output_directory}/X_train_{operation}.npy", X_train)
    np.save(f"{output_directory}/X_test_{operation}.npy", X_test)
    np.save(f"{output_directory}/y_train.npy", y_train)
    np.save(f"{output_directory}/y_test.npy", y_test)

    