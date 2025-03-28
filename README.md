# Machine Learning Project: Baseball Game Outcome Prediction

### Overview

This project focuses on predicting the outcomes of baseball games using machine learning techniques. Our main objectives included thorough data preprocessing, feature engineering, and designing sophisticated multi-model ensemble architectures to achieve accurate and reliable predictions.

---

### Tasks

The project is divided into three major subtasks:

1. **Data Observation and Preprocessing**
2. **Feature Engineering**
3. **Training Architecture Design and Model Selection**

---

### Data Preprocessing



  - Applied advanced imputation methods such as MissForest and KNN.
  - Filtered out data points with high missing-value ratios (>10%).

---

### Feature Engineering

We experimented with both dimensionality reduction and feature augmentation:

- **Dimensionality Reduction:**
  - Removed redundant features.
  - Applied PCA, t-SNE.
  - Selected top 20 influential features using Random Forest importance rankings.

- **Feature Augmentation:**
  - Generated new domain-specific features:
    - Batting average differences
    - Offensive efficiency metrics
    - Offense-defense efficiency ratio
    - Run conversion rates

---

### Model Architectures

- Employed a layered architecture combining Any Blending and Linear Blending.
- Base models included Random Forest and XGBoost.
- Validation-set blending involved CatBoost, Random Forest, and XGBoost.
- Logistic Regression was used for final predictions on the blended test-set features.
- Implemented Stratified K-Fold validation.

![image](https://hackmd.io/_uploads/S1MTTRmTyx.jpg)


---

### Key Findings

- Temporal characteristics significantly influence prediction accuracy.
- Effective feature engineering with domain knowledge drastically improves performance.
- Multi-layered model blending reduces overfitting and enhances generalization.
- Optimal model performance achieved by combining nonlinear models (XGBoost, CatBoost) with linear models (Logistic Regression).

---

### Team Members

- 古振宏 (Data Imputation, Feature Engineering, Model Training, Experimentation, Report Writing)
- 賴堯順 (Data Imputation, Feature Engineering, Model Training, Experimentation, Report Writing)
- 李怡潔 (Report Writing, Data Visualization, Training Framework Design, Experiment Integration)
- 張嘉亦 (Report Writing, Data Observation, Model Implementation)

---

### Reference

- Chen, P.-L., et al. "A linear ensemble of individual and blended models for music rating prediction." *Proceedings of KDD Cup 2011*, 2012.

---

### Usage

Follow these steps to use the project:

#### Data Download (Required)

Before running the project, you must download the dataset files:

- `train_data.csv`
- `test_data.csv`

These files are available via Google Drive. Please download them from the provided link: Download from Google Drive: [Link](https://drive.google.com/drive/folders/19UN5aJ48Ha6iBVEQDuMEHOhF-270CnKF?usp=sharing)

After downloading, place both `train_data.csv` and `test_data.csv` in the appropriate directory (e.g., a `data/` folder at the root of this project). Ensuring the files are in the correct location is essential for the code to locate the data.

#### Running the Project

```bash
git clone https://github.com/HungLinnnnn/Baseball-prediction.git
cd Baseball-prediction
pip install -r requirements.txt
python predict.py
```

