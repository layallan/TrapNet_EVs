# **TrapNet_EVs**  
### *High-Accuracy Label-Free Classification of Cancerous Extracellular Vesicles with Nanoaperture Optical Tweezers and Deep Learning*


---

## 🔍 **Overview**

This code supports:

- Training and evaluation of the **TrapNet** architecture  
- Baseline comparisons with LSTM, BiLSTM, CNN, CLSTM, and Transformer models  
- Visualization of latent features using **t-SNE** and analysis via **confusion matrices**

---

## 📁 **Directory Structure**

```
TrapNet_EVs/
│
├── TrapNet.py                 # Main training/testing script for TrapNet
├── KAN.py                     # Kolmogorov-Arnold Networks (optional model)
│
├── model/
│   └── best_model_by_accuracy.pth  # Pretrained TrapNet model checkpoint
│
├── data/
│   ├── data_new.mat           # Preprocessed dataset (MAT format)
│   ├── Data_formation.m       # MATLAB script for data segmentation & preprocessing
│   └── Data_formation.py      # Python equivalent of data preprocessing
│
├── baselines/
│   ├── 0_compare_LSTM.py      # LSTM baseline
│   ├── 0_compare_BiLSTM.py    # Bidirectional LSTM baseline
│   ├── 0_compare_CNN_pure.py  # CNN-only baseline
│   ├── 0_compare_CLSTM.py     # Convolutional LSTM baseline
│   └── 0_compare_Transformer.py # Transformer baseline
│
├── visualization/
│   └── plot_tnse_test.py      # t-SNE plots + confusion matrices for model evaluation
│
└── FDTD/
    └── DNH_Curve.py           # FDTD-based simulation of double-nanohole field distributions
```

---

## 🚀 **Quick Start**

### **1. Training TrapNet**

```bash
python TrapNet.py
```

### **2. Running Baseline Models**

```bash
python baselines/0_compare_LSTM.py
python baselines/0_compare_CNN_pure.py
# etc.
```

### **3. Visualization (t-SNE + Confusion Matrix)**

```bash
python plot_tnse_test.py
```

---

## 📊 **Data Formation**

The dataset used is preprocessed using consistent segmentation length (in MATLAB or Python). Raw optical trapping signals are converted into structured sequences stored in `data/data_new.mat`.

- MATLAB: `Data_formation.m`
- Python: `Data_formation.py`

---

## 📡 **FDTD Simulation**

To visualize the near-field electric field enhancement in the double-nanohole setup, refer to:

```bash
python FDTD/DNH_Curve.py
```

This uses Total-Field Scattered-Field (TFSF) methods for simulating the NOT setup.

---

## 📦 **Pretrained Model**

A pretrained checkpoint of TrapNet (`best_model_by_accuracy.pth`) is available in the `model/` folder. It can be directly loaded in **plot_tnse_test.py** for testing or fine-tuning.

---

## ✏️ **Citation**

If you find this repository useful for your research, please cite our paper (citation to be added upon publication):


