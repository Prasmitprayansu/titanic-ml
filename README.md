# 🚢 Titanic — Machine Learning from Disaster

A comprehensive machine learning classification project predicting passenger survival on the Titanic using scikit-learn. This repository demonstrates end-to-end ML workflow including EDA, feature engineering, preprocessing, model selection, and hyperparameter tuning.

**Kaggle Competition:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/)

---

## 📊 Project Overview

### Objective
Binary classification: predict which passengers survived the Titanic shipwreck based on passenger data (demographics, socioeconomic status, embarkation port, etc.).

### Dataset
- **Training set:** 891 passengers with survival labels  
- **Test set:** 418 passengers (submit predictions to Kaggle)
- **Features:** Passenger ID, Class, Name, Sex, Age, Family relationships, Fare, Cabin, Embarkation port

### Key Insights
- 🚺 **Gender matters:** Women had ~74% survival rate vs. men at ~19% ("women and children first" protocol)
- 💎 **Class hierarchy:** 1st-class passengers survived at 63% vs. 3rd-class at 24%
- 👶 **Age advantage:** Children had better survival odds
- 💰 **Fare correlation:** Higher-paying passengers (usually 1st class) survived more often
- 📍 **Embarkation port:** Cherbourg (C) passengers had higher survival (more 1st-class tickets)

---

## 🛠️ Technical Stack

| Category | Tools |
|----------|-------|
| **Data Processing** | pandas, numpy |
| **ML Models** | scikit-learn (LogisticRegression, RandomForest, GradientBoosting, SVM, KNN, DecisionTree) |
| **Visualization** | matplotlib, seaborn |
| **Evaluation** | cross-validation, confusion matrix, classification report |
| **Hyperparameter Tuning** | GridSearchCV |

---

## 📁 Repository Structure

```
titanic-ml/
├── README.md                          # This file
├── titanic_ml.ipynb                   # Complete Jupyter notebook (main work)
├── requirements.txt                   # Python dependencies
├── data/
│   ├── train.csv                      # Training data (download from Kaggle)
│   ├── test.csv                       # Test data (download from Kaggle)
├── submissions/
    └── gender_submission.csv          # Generated predictions (auto-created)

```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/titanic-ml.git
cd titanic-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data
- Go to [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
- Download `train.csv` and `test.csv`
- Place both files in the `data/` directory

### 4. Run the Notebook
```bash
jupyter notebook titanic_ml.ipynb
```

### 5. Generate Submission
The notebook automatically creates `submission.csv` when you run all cells.

### 6. Submit to Kaggle
**Option A: CLI (requires Kaggle API)**
```bash
kaggle competitions submit -c titanic -f submission.csv -m "Your message here"
```

**Option B: Manual**
Visit [Kaggle Titanic Submit Page](https://www.kaggle.com/competitions/titanic/submit) and upload `submission.csv`

---

## 📈 Model Performance

### Baseline Comparison (5-Fold Cross-Validation)
| Model | CV Accuracy | Notes |
|-------|------------|-------|
| Logistic Regression | ~81.5% | Fast, interpretable baseline |
| Decision Tree | ~78.2% | Prone to overfitting |
| **Random Forest** ⭐ | **~83.2%** | Best with tuning |
| **Gradient Boosting** ⭐ | **~83.1%** | Competitive alternative |
| SVM | ~80.8% | Slower, requires scaling |
| K-Nearest Neighbors | ~79.5% | Simple, baseline |

### Final Model
- **Algorithm:** Random Forest (tuned with GridSearchCV)
- **Expected Kaggle Score:** 0.78–0.82 (top 10% leaderboard)
- **Validation Accuracy:** ~83%

---

## 🔧 Key Features Engineered

| Feature | Description | Impact |
|---------|-------------|--------|
| **Title** | Extracted from Name (Mr., Mrs., Miss, Master, Rare) | High correlate with survival |
| **FamilySize** | SibSp + Parch + 1 | Family dynamics affect odds |
| **IsAlone** | Binary: traveling alone? | Traveling alone = lower survival |
| **FamilyCategory** | Alone / Small (2-4) / Large (5+) | Better than raw family size |
| **HasCabin** | Whether Cabin info is known | ~77% missing; presence is a signal |
| **Deck** | Cabin letter (A-F, T, Unknown) | Deck location may matter |
| **AgeBin** | Child / Teen / Adult / MiddleAge / Senior | Categorical age groups |
| **FareBin** | Low / Medium / High / VeryHigh | Quartile-based fare categories |

**Feature Engineering Decision:** Age imputed using median per Title (more accurate than global median). Cabin dropped due to 77% missing; instead, we use presence/absence as `HasCabin`.

---

## 📊 Preprocessing Pipeline

1. **Data Combination:** Merge train + test for consistent transformation
2. **Age Imputation:** Fill missing Age with median grouped by Title
3. **Categorical Filling:** Embarked → mode, Fare → median
4. **Label Encoding:** Convert categorical features (Sex, Embarked, Title, etc.)
5. **Feature Scaling:** StandardScaler for distance-based models
6. **Train/Val Split:** 80/20 stratified split to maintain class balance

---

## 🎓 Learning Outcomes

This project teaches:
- ✅ End-to-end ML workflow (EDA → modeling → submission)
- ✅ Handling imbalanced data and missing values
- ✅ Feature engineering and selection strategies
- ✅ Model comparison with cross-validation
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Confusion matrix and classification metrics interpretation
- ✅ Kaggle competition workflow

---

## 🚀 Next Steps to Improve Score

### 1. Advanced Modeling
```python
# Try XGBoost (often beats sklearn's GBM)
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)

# Or LightGBM (faster, memory-efficient)
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31)
```

### 2. Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('lr', LogisticRegression())
], voting='soft')
```

### 3. More Feature Engineering
- Extract cabin number from Cabin column
- Ticket prefix analysis
- Name length as a feature
- Interaction features: `Pclass × Sex`, `Age × Title`
- Fare per family member

### 4. Bayesian Optimization
```python
from optuna import create_study
study = create_study()
study.optimize(objective, n_trials=100)  # Smarter hyperparameter search
```

### 5. SHAP Values for Interpretability
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

### 6. Stacking
```python
from sklearn.ensemble import StackingClassifier
meta_learner = LogisticRegression()
stacking = StackingClassifier(estimators=[...], final_estimator=meta_learner)
```

---

## 📚 Dataset Details

### Columns
- **PassengerId:** Unique identifier
- **Survived:** 0 = No, 1 = Yes (target variable)
- **Pclass:** Passenger class (1, 2, or 3)
- **Name:** Passenger name
- **Sex:** gender (male or female)
- **Age:** Age in years (float, ~20% missing)
- **SibSp:** Number of siblings/spouses aboard
- **Parch:** Number of parents/children aboard
- **Ticket:** Ticket number
- **Fare:** Passenger fare in British pounds (~0.3% missing in test)
- **Cabin:** Cabin number (77% missing)
- **Embarked:** Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton; 2 missing)

---

## 📝 License

This project is open source and available under the **MIT License** (see LICENSE file).

---

## 🙏 Acknowledgments

- Kaggle for the dataset and competition platform
- scikit-learn for excellent ML tools
- The Titanic community for inspiring notebooks and kernels

---

## 📞 Contact & Support

- **Author:** Prasmit Prayansu
- **GitHub:** [Prasmitprayansu](https://github.com/Prasmitprayansu)
- **LinkedIn:** www.linkedin.com/in/prasmit-prayansu
- **Kaggle:** [prasmitprayansu2201](https://www.kaggle.com/prasmitprayansu2201)

---

**⭐ If you found this helpful, please star the repository! It helps others discover the project.**
