# Classifying Heart Disease using Random Forest Classifier

## 📋 Overview

This project explores the end-to-end workflow of a **Random Forest Classifier** applied to predicting heart disease from clinical patient data. It covers exploratory data analysis (EDA), training a default Random Forest, hyperparameter tuning with `GridSearchCV`, performance evaluation, feature importance analysis, and comparison against a single Decision Tree.

## 📊 Dataset

- **File:** `heart.csv`
- **Size:** 303 patient records, 13 clinical features + 1 target label
- **Target:** `1` = heart disease present, `0` = absent
- **Features:** age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`), max heart rate (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), ST slope (`slope`), number of major vessels (`ca`), thalassemia (`thal`)

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn (`RandomForestClassifier`, `DecisionTreeClassifier`, `GridSearchCV`, `cross_val_score`)
- Matplotlib, Seaborn
- pickle (for saving results)

## 🔍 Workflow

1. **Load data** — reads `heart.csv`, checks shape, missing values, and target class balance
2. **Exploratory Data Analysis** — target distribution, correlation heatmap, age distribution, max heart rate boxplot, chest pain type breakdown (all saved as PNGs)
3. **Train/test split** — 80/20 stratified split (`random_state=42`)
4. **Default Random Forest** — baseline model with scikit-learn defaults
5. **Hyperparameter tuning** — `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` with 5-fold CV
6. **Feature importance** — ranks which clinical features matter most to the tuned model
7. **Decision Tree comparison** — trains a single Decision Tree and compares accuracy, F1, and 5-fold CV stability against the tuned Random Forest, including an ROC curve

## 📁 Project Structure

```
├── heart.csv                          # Dataset (place in same folder as the script)
├── heart_disease_rf.py                # Main analysis script (this file)
├── heart_project/
│   ├── results.pkl                    # Pickled dictionary of all computed metrics
│   └── figs/                          # All generated chart PNGs
│       ├── target_dist.png
│       ├── corr_heatmap.png
│       ├── age_dist.png
│       ├── thalach_box.png
│       ├── cp_count.png
│       ├── cm_default.png
│       ├── cm_tuned.png
│       ├── feat_importance.png
│       ├── model_comparison.png
│       └── roc_curve.png
└── README.md
```

> `heart_project/` and `heart_project/figs/` are created automatically by the script (`os.makedirs`) — you don't need to make them yourself.

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

python heart_disease_rf.py
```

Make sure `heart.csv` is in the same folder as the script, or update the `CSV_PATH` variable at the top of the file to point to its actual location.

## 📈 Key Results

| Metric | Decision Tree | RF (Default) | RF (Tuned) |
|---|---|---|---|
| Accuracy | ~0.705 | ~0.836 | ~0.820 |
| Recall | — | ~0.970 | ~0.970 |
| F1-score | ~0.743 | ~0.865 | ~0.853 |
| 5-fold CV mean accuracy | ~0.755 | — | ~0.835 |

**Top predictive features:** chest pain type (`cp`), thalassemia (`thal`), and maximum heart rate (`thalach`).

## 🎯 Conclusion

The tuned Random Forest substantially outperforms a single Decision Tree on this dataset — both in raw accuracy and in stability across cross-validation folds — at the cost of some interpretability. Chest pain type and exercise-related cardiac stress indicators are the strongest predictors of heart disease in this sample.

## 📄 License

Created for educational purposes as part of an internship training program.
