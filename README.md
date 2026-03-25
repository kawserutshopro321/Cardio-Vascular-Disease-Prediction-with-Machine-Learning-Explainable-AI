# 🫀 Heart Disease Prediction with Machine Learning & Explainable AI

> **Built a complete ML pipeline that predicts cardiovascular disease with 95.6% accuracy, then used Explainable AI (LIME + SHAP) to make the model's decisions transparent and interpretable for medical professionals.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-006400?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8B0000?style=flat)
![LIME](https://img.shields.io/badge/LIME-Explainability-FF6F00?style=flat)

---

## 📌 Problem Statement

Cardiovascular diseases are the **#1 cause of death globally**. Early and accurate prediction can save lives — but black-box ML models aren't enough in healthcare. Doctors need to understand *why* a model flags a patient as high-risk. This project tackles both challenges: **high-accuracy prediction** and **model transparency**.

---

## 🔑 Key Highlights

| What I Did | Details |
|---|---|
| **Trained & compared 7 ML models** | Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, XGBoost, KNN |
| **Best accuracy: 95.6%** | K-Nearest Neighbors (Precision: 0.97, Recall: 0.94, F1: 0.95) |
| **Explainable AI with LIME** | Local, per-patient explanations showing which features drive each prediction |
| **Explainable AI with SHAP** | Global feature importance + individual force plots using TreeExplainer on XGBoost |
| **Ensemble model (Voting Classifier)** | Combined DT, LR, SVM, KNN, and RF via hard voting |
| **Comprehensive EDA** | 20+ visualizations covering distributions, correlations, and class balance |

---

## 🏗️ Project Architecture

```
📂 heart-disease-prediction-using-ML-and-XAI
│
├── cardio disease (2).ipynb    # Full pipeline: EDA → Preprocessing → Training → XAI
├── heart.csv                   # Dataset (1,025 records, 14 features)
├── CSE_499-journal.pdf         # Research paper / journal article
├── CSE_499B-PROGRESSREPORT.pdf # Project progress report
└── README.md
```

---

## 📊 Dataset

The dataset contains **1,025 patient records** with **13 clinical features** and a binary target variable.

| Feature | Description | Type |
|---|---|---|
| `age` | Patient age in years | Numeric |
| `sex` | Gender (1 = Male, 0 = Female) | Categorical |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0–4) | Categorical |
| `thal` | Thallium stress test result (0–3) | Categorical |
| **`target`** | **Heart disease (1 = Yes, 0 = No)** | **Binary** |

---

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis for every feature (pie charts, histograms, box plots)
- Cross-tabulations of disease prevalence by gender, vessel count, and chest pain type
- Correlation heatmap to identify multicollinearity and feature relationships
- Outlier detection using box plots for continuous variables

### 2. Data Preprocessing
- **One-hot encoding** for multi-class categorical features (`cp`, `thal`, `slope`)
- **Min-Max normalization** to scale all features to [0, 1]
- **80/20 train-test split** with fixed random state for reproducibility

### 3. Model Training & Evaluation
Seven classifiers trained with hyperparameter tuning:

| Model | Accuracy | Key Hyperparameters |
|---|---|---|
| Logistic Regression | 85.25% | `solver='liblinear'`, `penalty='l1'`, `max_iter=1000` |
| Naive Bayes | 85.25% | `var_smoothing=0.1` |
| Decision Tree | 80.33% | `max_depth=3`, `criterion='entropy'`, `min_samples_leaf=5` |
| Random Forest | 86.89% | `n_estimators=1000`, `max_leaf_nodes=20` |
| SVM | 86.89% | `kernel='linear'`, `C=10` |
| XGBoost | 95.08% | `objective='binary:logistic'` |
| **KNN (Best)** | **95.60%** | **`n_neighbors=3`** |

Additionally, a **Voting Classifier** (hard voting) was built by combining DT, LR, SVM, KNN, and RF.

### 4. Explainable AI (XAI)

**Why XAI matters:** In healthcare, a prediction alone isn't enough — clinicians need to know *which factors* are driving a diagnosis.

#### LIME (Local Interpretable Model-agnostic Explanations)
- Generated per-patient explanations for the KNN model
- Visualized which features pushed predictions toward "Disease" vs. "No Disease"
- Example: For a high-risk patient, LIME might show that `cp=0` (typical angina) and high `oldpeak` were the dominant factors

#### SHAP (SHapley Additive exPlanations)
- Used `TreeExplainer` on the XGBoost model for exact Shapley values
- **Summary plot** — global feature importance ranking across all test patients
- **Force plots** — individual prediction breakdowns showing how each feature shifts the output from the baseline
- **Beeswarm plot** — shows both importance and directionality of each feature

#### Permutation Importance (eli5)
- Measured feature importance by observing accuracy drop when each feature's values are shuffled
- Confirmed that `cp`, `thal`, and `ca` are the top-3 most predictive features

---

## 🧰 Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Yellowbrick |
| **Machine Learning** | scikit-learn, XGBoost |
| **Explainability** | LIME (Skater), SHAP, eli5 |
| **Preprocessing** | MinMaxScaler, One-Hot Encoding |
| **Environment** | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or later
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/UtshoData/heart-disease-prediction-using-ML-and-XAI.git
cd heart-disease-prediction-using-ML-and-XAI

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost yellowbrick lime shap eli5 skater

# Launch the notebook
jupyter notebook "cardio disease (2).ipynb"
```

---

## 📈 Results at a Glance

**Best Model — KNN (k=3)**

| Metric | Disease (1) | No Disease (0) |
|---|---|---|
| Precision | 0.97 | 0.94 |
| Recall | 0.94 | 0.97 |
| F1-Score | 0.95 | 0.95 |
| **Overall Accuracy** | | **95.60%** |

**Top Features (via SHAP & LIME)**
- `cp` — Chest pain type
- `thal` — Thallium stress test result
- `ca` — Number of major vessels colored by fluoroscopy
- `oldpeak` — ST depression (exercise vs. rest)

---

## 🔮 Future Improvements

- Deploy as a web app using Flask/Streamlit for real-time predictions
- Experiment with deep learning models (neural networks)
- Test on larger, more diverse clinical datasets
- Add cross-validation and hyperparameter optimization (GridSearchCV / Optuna)
- Build an interactive SHAP dashboard for clinicians

---

## 📄 License

This project is open-source and available for academic and research purposes.

---

## 🙋 Author

**Utsho** — [GitHub Profile](https://github.com/https://github.com/kawserutshopro321)

If you found this useful, consider giving the repo a ⭐!
