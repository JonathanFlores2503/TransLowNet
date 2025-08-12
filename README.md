# TransLowNet


# Online Modular Framework for Anomaly Detection and Multiclass Classification in Video Surveillance

<img width="1298" height="801" alt="image" src="https://github.com/user-attachments/assets/b288cf71-bb82-4659-aec4-5f33ed06e720" />

This repository contains the inference implementation for the article **"An Online Modular Framework for Anomaly Detection and Multiclass Classification in Video Surveillance"**, currently under review.  
The following steps describe how to set up the environment, download the necessary precomputed features and model weights, and run the inference locally.

> **Note:** The current state of this code is for the inference stage only. The full training code is under improvement and will be updated in a future release.

---

## Step 1 – Environment Setup

This repository was developed and tested using **Python 3.8.10**.  
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Step 2 – Download Precomputed Features

To replicate the experiments, download the precomputed features used in inference:

- **Features – 10 Crops**: UniFormer-S *(link to be provided)*
- **Features – Size Level**: UniFormer-S *(link to be provided)*

After downloading, place the files in the correct directories as specified in the scripts.


## Step 3 – Download Model Weights

Download the pretrained model weights for both the anomaly detection and classification modules:

- **C2FPL_UniFormer-S** – Detector Weights *(link to be provided)*
- **TransLowNet_UniFormer-S** – Classifier Weights *(link to be provided)*

Ensure the script paths match the location where these files are stored.


## Step 4 – Run Inference in Jupyter

Open the notebook **`inferenceV1.ipynb`** and execute it step by step to perform the inference process. The notebook contains:

- **Anomaly Detection** — using C2FPL with UniFormer-S features.
- **Frame-level AUC Calculation** — ROC curve generation and midpoint threshold selection.
- **Multiclass Classification** — using TransLowNet for anomalous clip classification.
- **Final Metrics Computation** — Precision, Recall, F1-score, and Accuracy at clip and video level.

