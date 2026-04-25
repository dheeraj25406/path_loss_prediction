# 📡 ML-Based Path Loss Prediction for 6G Wireless Channels with RIS

This project focuses on predicting **path loss in next-generation (6G) wireless communication systems** using Machine Learning, with and without the effect of **Reconfigurable Intelligent Surfaces (RIS)**.

---

##  Project Overview

Traditional path-loss models rely on fixed mathematical formulas and fail to capture real-world complexity.  
This project uses **Machine Learning models** to learn patterns directly from data and improve prediction accuracy.

We also incorporate the impact of **RIS (Reconfigurable Intelligent Surfaces)** to demonstrate how signal propagation can be improved.

---

##  Objectives

- Predict **Path Loss without RIS**
- Predict **Path Loss with RIS**
- Compare model performance using standard metrics
- Visualize results using meaningful graphs
- Demonstrate real-time prediction using custom inputs

---

##  Machine Learning Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor ✅ (Best performing)

---

## 📊 Evaluation Metrics

- R² Score (Accuracy)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## 📁 Dataset

The dataset contains wireless channel parameters such as:

- T-R Separation Distance (m)
- Frequency
- Time Delay (ns)
- RMS Delay Spread (ns)
- Azimuth & Elevation (AoD, AoA)
- Received Power (dBm)
- Phase (rad)
- RIS Gain (dB)
- Path Loss (dB)
- RIS Path Loss (dB)

---

## 📈 Generated Graphs

The following graphs are automatically generated and saved:

1. **Path Loss vs Distance (With & Without RIS)**
2. **Actual vs Predicted (Without RIS)**
3. **Actual vs Predicted (With RIS)**
4. **Model Comparison (R² Score)**
5. **Improvement due to RIS**
6. **Error Distribution (Without RIS)**
7. **Error Distribution (With RIS)**

---

## 🖥️ How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```
## 2. Run the Script
```bash
python path_loss_model.py
```
