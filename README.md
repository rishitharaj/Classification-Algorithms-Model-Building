# Classification-Algorithms-Model-Building - Breast Cancer Classification – Supervised Learning

## 📌 Objective

The objective of this assessment is to evaluate the understanding and ability to apply supervised learning techniques to a real-world dataset using the **Breast Cancer** dataset from the `sklearn` library.

---

## 📂 Dataset

- **Source**: `sklearn.datasets.load_breast_cancer`
- **Type**: Binary classification (Malignant vs Benign)
- **Features**: 30 numerical features computed from digitized images of breast mass
- **Target**: 0 – Malignant, 1 – Benign

---

## 🧹 Data Loading & Preprocessing

Steps performed in the notebook:

1. **Load Dataset**
   - Loaded using `load_breast_cancer()` from `sklearn`.
   - Converted to a pandas DataFrame for easier manipulation.

2. **Missing Values Handling**
   - Used `SimpleImputer(strategy="mean")` as a robust step, although the dataset typically has no missing values.
   - This makes the pipeline more generalizable to real-world situations.

3. **Feature Scaling**
   - Applied `StandardScaler` to standardize features.
   - Necessary because models like **Logistic Regression**, **SVM**, and **k-NN** are sensitive to the scale of features.

4. **Train–Test Split**
   - Used `train_test_split` with:
     - `test_size=0.33`
     - `random_state=42`
     - `stratify=y` to preserve class distribution.

---

## 🤖 Models Implemented

The following **five classification algorithms** were implemented:

1. **Logistic Regression**
   - Linear model using the logistic (sigmoid) function to estimate class probabilities.
   - Works well for linearly separable data and high-dimensional feature spaces.

2. **Decision Tree Classifier**
   - Tree-based model that splits data based on feature thresholds to maximize purity.
   - Interpretable but prone to overfitting if not regularized.

3. **Random Forest Classifier**
   - Ensemble of multiple decision trees trained on bootstrapped samples with feature subsampling.
   - Reduces overfitting and usually improves performance over a single decision tree.

4. **Support Vector Machine (SVM)**
   - Finds an optimal separating hyperplane that maximizes margin between classes.
   - With the RBF kernel, can model non-linear decision boundaries.
   - Works very well on standardized, high-dimensional data.

5. **k-Nearest Neighbors (k-NN)**
   - Non-parametric algorithm that classifies a sample based on the majority class among its k nearest neighbors.
   - Relies on distance metrics, so feature scaling is crucial.

---

## 📊 Evaluation & Metrics

For each model, the notebook computes:

- **Accuracy Score**

A comparison table of accuracies for all models is generated to identify:

- ✅ **Best performing algorithm**
     **Logistic Regression** tends to perform the best.
- ❌ **Worst performing algorithm**
     **Decision Tree** often performs the worst due to overfitting.

- **Confusion Matrix**
  - Visualized using `matplotlib` for each algorithm.

---

## 🧪 Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- Colab Notebook

---

## ▶ How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
