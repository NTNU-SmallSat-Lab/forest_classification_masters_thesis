# Dimensionality Reduction and Deep Learning for Forest Classification

This repository contains code associated with the forthcoming Master's thesis:  
**"Dimensionality Reduction and Deep Learning for Forest Classification Based on Hyperspectral HYPSO-1 Data"**  
*(Mjøen, 2025)*
It will be published to [NTNU Open](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/227492)

As of June 2025, the datasets used in the Trondheim dataset can be found [here](https://studntnu-my.sharepoint.com/personal/samuelbo_ntnu_no/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsamuelbo%5Fntnu%5Fno%2FDocuments%2FS%26T%2DForest%2DData%2DAutumn%2D2024&T-Forest-Data-Autumn-2024=). It includes captures from an area outside of Trondheim from an aerial vehicle and the satellites Sentinel-2, Landsat-8 and Pleiades-1A. Additionally, it contains the forest type map SR16 from the same area. The constructed datasets described in the Master's thesis can be found [here](https://studntnu-my.sharepoint.com/:f:/g/personal/elismj_ntnu_no/EmT5lfMVyI5Dg74N_LUTVzUB6BUmS4VQM8y3B3CfxedtqA?e=klIN20). 

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

- `process_datasets.py` — For combining datasets and performing band selection  
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

## License

This repository uses and adapts several open-source codes. See the [LICENSE](LICENSE) file for full terms and conditions.