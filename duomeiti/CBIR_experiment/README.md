# Color Histogram CBIR Experiment

A simple Content-Based Image Retrieval (CBIR) experiment based on global RGB color histogram features.

## 1. Project Goal

This project implements a complete CBIR pipeline:

1. Extract global color histogram features from images.
2. Compute image similarity using histogram intersection.
3. Retrieve Top-K most similar images for a query image.
4. Evaluate retrieval quality with Precision, Recall, and AP/MAP.
5. Visualize retrieval results and Precision-Recall curve.

## 2. Project Structure

```text
CBIR_experiment/
|-- dataset/            # image dataset
|-- query.jpg           # query image (query.jps/query.jpeg/query.png also supported)
|-- cbir_experiment.py  # main experiment script
|-- README.md
```

## 3. Environment

- Python 3.8+
- numpy
- opencv-python
- matplotlib

Install dependencies:

```bash
pip install numpy opencv-python matplotlib
```

## 4. How It Works

### 4.1 Feature Extraction

- Use 3D RGB histogram as global image feature.
- Default histogram bins: `(8, 8, 8)`.
- Feature dimension is `8 * 8 * 8 = 512`.

### 4.2 Similarity

- Similarity metric: histogram intersection.
- Larger score means more similar color distribution.

### 4.3 Label Rule

- Class label is extracted from the leading alphabetic prefix of file name.
- Example: `cat10.png -> cat`, `dog2.jpg -> dog`, `beach1.png -> beach`.

### 4.4 Evaluation

- Precision and Recall curves are computed from full ranking.
- AP is computed for one query.
- In this script, `MAP` equals `AP` for single-query evaluation.

## 5. Run

From project root:

```bash
python cbir_experiment.py
```

The script will:

1. Index all images in `dataset/`.
2. Resolve query image automatically (`query.jps`, `query.jpg`, `query.jpeg`, `query.png`, `query.bmp`).
3. Retrieve Top-3 by default.
4. Print metrics in terminal.
5. Show two figures:
   - Figure 1: Precision-Recall curve + query image.
   - Figure 2: Top-K retrieval result images.

## 6. Adjustable Parameters

Edit the call at the bottom of `cbir_experiment.py`:

```python
run_experiment(dataset_dir="dataset", query_path=None, top_k=3, bins=(8, 8, 8))
```

- `top_k`: number of returned images.
- `bins`: histogram granularity, e.g. `(4,4,4)`, `(8,8,8)`, `(16,16,16)`.
- `query_path`: custom absolute/relative path for query image.

## 7. Notes

- This method compares global color distribution, not pixel-by-pixel alignment.
- It is robust to translation/scale changes to some extent.
- It may fail when different objects share similar color statistics.

## 8. GitHub Upload Suggestion

- Safe to upload code directly.
- For dataset images, ensure you have redistribution rights.
- If image copyright is unclear, upload only code and keep dataset local.
