# DA5401 A4 : Gaussian Mixture Model (GMM) Based Synthetic Sampling

*Name*: Krithi Shailya
*Roll Number*: DA25S009

This repository contains the solution for the **DA5401 Assignment 4 (A4)** on **Gaussian Mixture Models (GMMs) and Imbalanced Data Handling**.

## Overview

The assignment focuses on applying Gaussian Mixture Models (GMMs) to generate synthetic samples for handling imbalanced datasets (fraud detection in credit card transactions). The work compares baseline models with GMM-based oversampling and clustering-based undersampling (CBU).

### Key Components
- **Baseline Model**: Logistic Regression trained on the imbalanced dataset.
- **GMM-based Oversampling**: Fit a GMM on the minority class, select number of components via BIC/AIC, and generate synthetic samples.
- **GMM+ Clustering-Based Undersampling (CBU)**: Reduce majority class size via KMeans clustering while retaining diversity and oversample minority class. 
- **Evaluation**: Precision, Recall, F1, Accuracy, Confusion Matrix on the untouched imbalanced test set.
- **Visualization**: Bar plots comparing baseline vs. rebalanced models.

---

## Files

- `A4_GMM.ipynb` — Main Jupyter Notebook with full assignment solution.
- `README.md` — Project documentation (this file).
- `requirements.txt` — Dependencies required to run the notebook.

---

## Setup Instructions

1. Clone this repository or download the files.
2. Download the dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
3. Place the dataset file `creditcard.csv` in the same directory as the notebook.
4. Install required dependencies:

```bash
pip install -r requirements.txt
```

5. Open and run the notebook:

```bash
jupyter notebook A4_GMM.ipynb
```

---

## Workflow

1. **Data Preprocessing**:
   - Log-transform `Amount` feature.
   - Drop or scale `Time` feature as appropriate.

2. **Baseline**:
   - Logistic Regression on imbalanced data.
   - Evaluation on untouched imbalanced test set.

3. **GMM-Based Oversampling**:
   - Fit GMM to minority class.
   - Choose `k` using BIC/AIC.
   - Generate synthetic samples to balance dataset.

4. **Clustering-Based Undersampling (CBU)**:
   - Cluster majority class with KMeans.
   - Sample proportionally from clusters to reduce size.

5. **Evaluation**:
   - Compare Baseline, GMM Oversampled, and CBU+ GMM Hybrid, Rebalanced models. 
   - Metrics: Precision, Recall, F1, Accuracy.
   - Visualization with bar plots.

6. **Conclusion**:
   - Trade-offs between recall and precision are discussed.
   - Recommendation provided for fraud detection deployment.

---

## Requirements

See `requirements.txt` for full details.

---

## Notes

- Ensure `creditcard.csv` is placed in the correct directory before running the notebook.
- For reproducibility, random seeds (`random_state=42`) are fixed for sklearn models.
- The notebook avoids copying code from class material and instead uses original implementations inspired by the taught algorithms.

---

