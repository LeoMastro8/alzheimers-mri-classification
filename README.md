# 🧠 3D Deep Learning Pipeline for Alzheimer's Disease Classification

This repository contains the codebase for an end-to-end deep learning pipeline designed to classify patients into **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer's Disease (AD)** using 3D multi-modal Magnetic Resonance Imaging (MRI).

## 🚀 Project Overview
Diagnosing Alzheimer's from structural imaging is highly challenging due to subtle anatomical differences and high inter-subject variability. This project leverages an **Early Fusion** strategy to combine T1-weighted, T2-weighted, and FLAIR MRI sequences into a unified 4D tensor, processing the full spatial volume of the brain to identify disease biomarkers.

### ✨ Key Features
* **Robust Preprocessing:** Automated skull-stripping using HD-BET, rigid co-registration via ANTsPy, and 3D volumetric standardization/resizing using MONAI.
* **Custom 3D CNN Baseline:** A lightweight 3D Convolutional Neural Network trained from scratch.
* **Transfer Learning:** Integration of the pre-trained **MedicalNet** (3D ResNet-18) backbone to leverage prior anatomical knowledge, preventing overfitting on small medical datasets.
* **Imbalance Handling:** Implementation of Focal Loss and Inverse Frequency Weighting to handle a severe minority AD class distribution.
* **Explainable AI (XAI):** Visual validation using Captum's Layer Grad-CAM, confirming the models successfully learned to target clinical biomarkers.

## 📁 Repository Structure
* `preprocess.py`: Complete MRI preprocessing pipeline (HD-BET, ANTsPy, MONAI).
* `dataloader.py` & `adni_dataset.py`: Custom PyTorch datasets with 3D data augmentation (affine transformations, intensity scaling, Gaussian noise).
* `model.py`: Architecture for the custom baseline 3D CNN.
* `medical_net.py`: Adaptation of the MedicalNet Transfer Learning model.
* `train.py`: Training loops, loss functions (Cross-Entropy, Focal Loss), and validation logic.
* `explainability.py`: Using Captum's Layer Grad-CAM to generate 3D visual heatmaps, confirming the models' focus on key anatomical biomarkers.

Data used in this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. Special thanks to the developers of MedicalNet and HD-BET.
