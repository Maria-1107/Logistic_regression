<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=160&section=header&text=Insurance%20Claimant%20%E2%80%94%20ML%20Classification%20Suite&fontSize=28&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Logistic%20Regression%20%7C%20Decision%20Tree%20%7C%20Ensemble%20Methods%20%7C%20Model%20Evaluation&descAlignY=62&descSize=14" width="100%"/>

<div align="center">

  ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
  ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
  ![Status](https://img.shields.io/badge/Status-Complete-38BDAE?style=flat-square)

</div>

---

## 📌 Project Overview

A complete machine learning classification project built on **1,300+ insurance bodily injury claim records**, progressively applying multiple algorithms and evaluation strategies to predict whether a claimant will hire an attorney.

The repo is structured as a **learning-to-production pipeline** — starting from logistic regression, progressing to decision trees and ensemble methods, and wrapping up with advanced model validation techniques and bias-variance analysis.

---

## 🧠 Business Understanding

| | |
|:---|:---|
| **Domain** | Insurance / Bodily Injury Claims |
| **Dataset** | ~1,300+ historical claim records |
| **Target Variable** | `ATTORNEY` → 1 = Hired an attorney, 0 = Did not |
| **ML Problem** | Binary Classification |

**Why it matters:** When claimants hire attorneys, settlement amounts increase — directly raising costs for the insurer. Predicting this early enables proactive case management.

**Business Questions:**
- Who is likely to hire an attorney?
- Which regions have higher attorney-hiring rates?
- What features most influence this decision?

---

## 🗂️ Repository Structure

```
📦 Logistic_regression
 ┣ 📓 Claimant.ipynb                    ← Logistic Regression — base model
 ┣ 📓 Evaluation_metrics.ipynb          ← Full evaluation (Confusion Matrix, ROC-AUC)
 ┣ 📓 Train_Test_Split.ipynb            ← Train/test split walkthrough
 ┣ 📓 Decision_tree.ipynb               ← Decision Tree + Bias-Variance + Grid Search CV
 ┣ 📓 Ensemble_techniques.ipynb         ← Random Forest + AdaBoost + Voting Classifier
 ┣ 📓 Model_evaluation_tech.ipynb       ← KFold CV + LOOCV + Stacking
 ┣ 💾 Claimant.pkl                      ← Saved Logistic Regression model
 ┣ 💾 Claimant_decision_metrics.pkl     ← Saved Decision Tree model
 ┣ 💾 Claimant_train_test_split.pkl     ← Saved train/test split
 ┗ 📄 README.md
```

---

## ⚙️ ML Pipeline

```
Raw Data (Claimant.csv — 1,300+ records)
        │
        ▼
  Data Understanding
  ├── .shape, .dtypes, .isna().sum(), .duplicated().sum()
  └── .describe(include='all')
        │
        ▼
  Data Cleaning
  ├── .dropna() → remove rows with missing values
  └── Drop: CASENUM (ID column, not a feature)
        │
        ▼
  Feature / Target Split
  ├── X = all columns except CASENUM, ATTORNEY
  └── y = ATTORNEY (target)
        │
        ▼
  Model Validation Strategy          ← Model_evaluation_tech.ipynb
  ├── Train/Test Split (80/20, shuffle=True, random_state)
  ├── K-Fold Cross Validation (k=5)
  └── Leave-One-Out CV (LOOCV)
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
  Logistic Regression               Decision Tree Classifier
  (Claimant.ipynb)                  (Decision_tree.ipynb)
        │                                                 │
        │                              ├── Default tree (overfitting check)
        │                              ├── max_depth=5 (bias-variance fix)
        │                              ├── Grid Search CV (best hyperparams)
        │                              └── plot_tree() visualization
        │                                                 │
        └──────────────────┬──────────────────────────────┘
                           ▼
              Ensemble Methods (Ensemble_techniques.ipynb)
              ├── Random Forest (n_estimators=100, max_depth=5)
              ├── AdaBoost Classifier
              └── Voting Classifier (LR + DT + KNN — Stacking)
                           │
                           ▼
              Model Evaluation (all notebooks)
              ├── Confusion Matrix (train + test)
              ├── Classification Report (Precision, Recall, F1)
              ├── ROC Curve + AUC Score
              └── Bias-Variance Analysis
                           │
                           ▼
              Model Export (.pkl via pickle)
```

---

## 🔑 Key Concepts Covered

### Bias-Variance Tradeoff
```
Training Accuracy  → measures BIAS
Test Accuracy      → measures VARIANCE

Overfitting  = Low Bias  + High Variance  ← Default Decision Tree
Underfitting = High Bias + Low Variance

✅ Target: Generalized Model = Low Bias + Low Variance
   → Fixed by setting max_depth=5 on the Decision Tree
```

### Model Validation Techniques
```
1. Train/Test Split    → Quick, single split (80/20)
2. K-Fold CV (k=5)     → 5 rotations, avg accuracy + std deviation
3. LOOCV               → Each row used once as test set (most thorough)
```

### Ensemble Methods
```
Random Forest    → Bagging — multiple trees, majority vote
AdaBoost         → Boosting — sequential correction of errors
Voting Classifier → Stacking — LR + DT + KNN combined prediction
Grid Search CV   → Hyperparameter tuning (max_depth, criterion, n_estimators)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|:---|:---|
| Language | Python 3 |
| Data Handling | Pandas |
| Models | `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `VotingClassifier`, `KNeighborsClassifier` |
| Validation | `train_test_split`, `KFold`, `LeaveOneOut`, `cross_val_score`, `GridSearchCV` |
| Evaluation | `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_curve`, `roc_auc_score` |
| Visualization | Matplotlib, `plot_tree` |
| Model Persistence | Pickle |
| Environment | Jupyter Notebook |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Maria-1107/Logistic_regression.git
cd Logistic_regression

# 2. Install dependencies
pip install pandas scikit-learn matplotlib notebook

# 3. Add the dataset
# Place Claimant.csv in the project root

# 4. Run notebooks in this order
jupyter notebook Claimant.ipynb
jupyter notebook Train_Test_Split.ipynb
jupyter notebook Evaluation_metrics.ipynb
jupyter notebook Decision_tree.ipynb
jupyter notebook Ensemble_techniques.ipynb
jupyter notebook Model_evaluation_tech.ipynb
```

---

## 💡 Key Learnings

- Applied **three model validation strategies** (Train/Test, K-Fold, LOOCV) and understood when each is appropriate
- Diagnosed and **fixed overfitting** in the Decision Tree using `max_depth` constraint and bias-variance analysis
- Used **Grid Search CV** to systematically find optimal hyperparameters (max_depth, criterion, n_estimators)
- Combined weak learners into stronger predictions using **Bagging (Random Forest), Boosting (AdaBoost), and Stacking (Voting Classifier)**
- Evaluated every model on **both training and test sets** to detect generalization issues
- Interpreted **ROC-AUC curves** to compare model discrimination ability beyond simple accuracy
- Exported trained models with **pickle** for reuse without retraining

---

## 👩‍💻 Author

**Maria Rexcy B**
📍 Trichy, India
🔗 [GitHub](https://github.com/Maria-1107) · [LinkedIn](https://www.linkedin.com/in/maria-rexcy-670268281) · [Email](mailto:mariarexcy11@gmail.com)

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
