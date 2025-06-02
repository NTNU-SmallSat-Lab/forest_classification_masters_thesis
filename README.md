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

## License

This repository uses and adapts several open datasets and open-source libraries. See the [LICENSE](LICENSE) file for full terms and conditions.