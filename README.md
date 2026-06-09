## 📌 Project Overview

This project builds a **binary classification model** to predict whether an insurance claimant will hire an attorney — a key cost driver for insurance companies handling bodily injury claims.

When a claimant hires an attorney, settlement amounts tend to increase significantly, directly impacting the **insurer's cost and profitability**. By predicting this outcome early, insurers can take proactive action.

---

## 🧠 Business Understanding

| | |
|:---|:---|
| **Domain** | Insurance / Bodily Injury Claims |
| **Dataset Size** | ~1,300+ historical claim cases |
| **Target Variable** | `ATTORNEY` → 1 = Hired, 0 = Not Hired |
| **ML Problem Type** | Binary Classification |

**Business Questions Answered:**
- Who is likely to hire an attorney?
- Which regions have a high attorney-hiring rate?
- What factors most influence the decision?

**Business Actions Enabled:**
- Improve claim handling strategy
- Better reserve planning
- Customer segmentation
- Cost reduction

---

## 🗂️ Project Structure

```
📦 Claimant-Attorney-Prediction
 ┣ 📓 Claimant.ipynb                      ← End-to-end pipeline (EDA + Model)
 ┣ 📓 Train_Test_Split.ipynb              ← Train/test split walkthrough
 ┣ 📓 Evaluation_metrics.ipynb            ← Confusion matrix, ROC-AUC, reports
 ┣ 📓 Evaluation_metrics-Copy1.ipynb      ← Iterative version with pickle export
 ┣ 💾 Claimant.pkl                        ← Saved trained model
 ┣ 💾 Claimant_decision_metrics.pkl       ← Decision threshold metrics
 ┣ 💾 Claimant_train_test_split.pkl       ← Saved train/test split
 ┗ 📄 README.md
```

---

## ⚙️ Workflow

```
Raw Data (Claimant.csv)
        │
        ▼
  Data Understanding       ← shape, dtypes, null check, duplicates
        │
        ▼
  Data Cleaning            ← dropna (handled missing values)
        │
        ▼
  Feature Engineering      ← Separate inputs (X) and target (y = ATTORNEY)
        │
        ▼
  Train / Test Split       ← 80/20 split, shuffle=True, random_state fixed
        │
        ▼
  Model Training           ← LogisticRegression (sklearn)
        │
        ▼
  Model Evaluation         ← Accuracy, Classification Report,
                              Confusion Matrix, ROC Curve, AUC Score
        │
        ▼
  Model Export             ← Saved via pickle (.pkl)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|:---|:---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (LogisticRegression, train_test_split) |
| Evaluation | accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score |
| Visualization | Matplotlib |
| Model Persistence | Pickle |
| Environment | Jupyter Notebook |

---

## 📊 Evaluation Metrics Used

- ✅ **Confusion Matrix** — True/False Positives and Negatives
- ✅ **Classification Report** — Precision, Recall, F1-Score
- ✅ **ROC Curve** — Receiver Operating Characteristics
- ✅ **AUC Score** — Area Under the Curve (model discrimination power)

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Maria-1107/claimant-attorney-prediction.git
cd claimant-attorney-prediction

# 2. Install dependencies
pip install pandas scikit-learn matplotlib notebook

# 3. Add the dataset
# Place Claimant.csv in the project root

# 4. Run notebooks in order
jupyter notebook Claimant.ipynb
jupyter notebook Train_Test_Split.ipynb
jupyter notebook Evaluation_metrics.ipynb
```

---

## 💡 Key Learnings

- Applied the full ML lifecycle: **data understanding → cleaning → modeling → evaluation → deployment**
- Used `random_state` in train/test split to ensure **reproducible results**
- Evaluated model on both **training and test data** to detect overfitting
- Exported the trained model with **pickle** for reuse without retraining
- Interpreted ROC-AUC to assess the model's ability to **distinguish between classes**

---

## 👩‍💻 Author

**Maria Rexcy B**
📍 Trichy, India
🔗 [GitHub](https://github.com/Maria-1107) · [LinkedIn](https://www.linkedin.com/in/maria-rexcy-670268281) · [Email](mailto:mariarexcy11@gmail.com)
