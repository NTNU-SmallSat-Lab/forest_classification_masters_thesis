# Feature Engineering and Deep Learning for Forest Classification

This repository contains code and data associated with the forthcoming Master's thesis:  
**"Feature Engineering and Deep Learning for Forest Classification Based on Hyperspectral HYPSO-1 Data"**  
*(Mjøen, 2025)*

---

## Dataset Preparation

### HYPSO
Use `extract_input_and_labels_HYPSO.ipynb` to prepare the input and labels.  
Expected files in each `HYPSO` capture folder:
- `{hypso_capture}-l1a.nc` — Hyperspectral data
- `{hypso_capture}.dat` — Sea-land-cloud segmentation
- `latitudes_indirectgeoref.dat` or `latitudes.dat` — Latitude coordinates
- `longitudes_indirectgeoref.dat` or `longitudes.dat` — Longitude coordinates
- `.tif` file — Label raster (e.g., forest type)

### Landsat
Use `extract_input_and_labels_Landsat.ipynb` for preprocessing.  
Expected input:
- Directory with multispectral `.tif` files
- `.tif` file — QA (quality assessment) band
- `.tif` file — Label raster

### Sentinel-2
Use `extract_input_and_labels_Sentinel.ipynb` for preprocessing.  
Expected input:
- `.tif` file — Multispectral image
- `.tif` file — Label raster

---

## Dataset Processing

- `process_datasets.py` — For combining inputs and performing band selection  
- `match_datasets.ipynb` — For matching bands across different satellites

---

## Feature Analysis

- `feature_analysis.ipynb` — PCA, band selection, and spectral response analysis  
- `spectral_signatures.ipynb` — Comparison of spectral signatures between sensors

---

## Model Development

- `CNN_training.ipynb` — Training MobileNet-based 1D CNN models  
- `transfer_learning_mapping.ipynb` — Transfer learning with a wavelength mapping layer  
- `transfer_learning_matching.ipynb` — Transfer learning using matched satellite datasets

---

## Available Datasets

All datasets are located in the `data/` folder.  
HYPSO has separate label files for the intermediate dataset and common label files for the remaining sets. Sentinel and Landsat have one set of labels each for training and testing.

### HYPSO Final Dataset
- 107 bands
- 15 principal components (PCs)
- 10 PCs
- 7 PCs
- 3 PCs (95% explained variance)

### HYPSO Intermediate Dataset
- 111 bands

### HYPSO PCA-Based Band Selection
- 66 bands
- 10 PCs
- 2 PCs (95% explained variance)

### HYPSO Literature-Based Band Selection
- 7 bands
- 7 PCs
- 2 PCs (95% explained variance)

### Landsat-Matched HYPSO
- 5 averaged bands

### Landsat Original
- 7 bands

### HYPSO-Matched Landsat
- 5 bands

### Sentinel Original
- 20 bands

---

## License

This repository uses and adapts several open datasets and open-source libraries. See the [LICENSE](LICENSE) file for full terms and conditions.

Key licenses:
- Sentinel-2: [Sentinel Data Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice)
- Landsat: [USGS Landsat Data Policy](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data)
- HYPSO: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en)
- hypso package code: [MIT License](https://opensource.org/license/mit/)
- 