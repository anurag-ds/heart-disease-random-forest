{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "5c50a2f4-bc13-4e6b-affd-409a7467c046",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib\n",
    "matplotlib.use(\"Agg\")\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,\n",
    "                              roc_curve, auc, precision_score, recall_score, f1_score)\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9cd15475-2cb2-4707-a502-bcdd86f3c7fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set_style(\"whitegrid\")\n",
    "plt.rcParams['figure.dpi'] = 140"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "cc1f355c-c335-46ba-ae33-c1c68dc59770",
   "metadata": {},
   "outputs": [],
   "source": [
    "BASE_DIR = \"./heart_project\"    \n",
    "FIGDIR = os.path.join(BASE_DIR, \"figs\")\n",
    "CSV_PATH = \"heart.csv\"\n",
    "RESULTS_PATH = os.path.join(BASE_DIR, \"results.pkl\")\n",
    "\n",
    "os.makedirs(FIGDIR, exist_ok=True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "3d823783-3810-492f-ba72-e131ecaf7356",
   "metadata": {},
   "outputs": [],
   "source": [
    "RESULTS = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "24c3e9aa-8c96-405b-8613-fa9c02fef7ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(CSV_PATH)\n",
    "RESULTS['shape'] = df.shape\n",
    "RESULTS['columns'] = list(df.columns)\n",
    "RESULTS['missing'] = int(df.isnull().sum().sum())\n",
    "RESULTS['target_counts'] = df['target'].value_counts().to_dict()\n",
    "RESULTS['describe'] = df.describe().round(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3a031cbd-4786-4f10-b3c5-8ea0acd398b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "col_meaning = {\n",
    "    \"age\": \"Age in years\",\n",
    "    \"sex\": \"Sex (1 = male, 0 = female)\",\n",
    "    \"cp\": \"Chest pain type (0-3)\",\n",
    "    \"trestbps\": \"Resting blood pressure (mm Hg)\",\n",
    "    \"chol\": \"Serum cholesterol (mg/dl)\",\n",
    "    \"fbs\": \"Fasting blood sugar > 120 mg/dl (1 = true)\",\n",
    "    \"restecg\": \"Resting ECG results (0-2)\",\n",
    "    \"thalach\": \"Maximum heart rate achieved\",\n",
    "    \"exang\": \"Exercise induced angina (1 = yes)\",\n",
    "    \"oldpeak\": \"ST depression induced by exercise\",\n",
    "    \"slope\": \"Slope of peak exercise ST segment\",\n",
    "    \"ca\": \"Number of major vessels colored by fluoroscopy (0-4)\",\n",
    "    \"thal\": \"Thalassemia (0-3 code)\",\n",
    "    \"target\": \"1 = heart disease present, 0 = absent\"\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "64af1183-aa5a-4ec9-b539-7332bebef8ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(5, 4))\n",
    "ax = sns.countplot(x='target', hue='target', data=df, palette=[\"#4C72B0\", \"#C44E52\"], legend=False)\n",
    "ax.set_xticks([0, 1])\n",
    "ax.set_xticklabels([\"No Disease (0)\", \"Disease (1)\"])\n",
    "plt.title(\"Distribution of Target Classes\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.xlabel(\"\")\n",
    "for p in ax.patches:\n",
    "    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),\n",
    "                ha='center', va='bottom')\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/target_dist.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "97118d58-e622-41e9-b07c-25209359cdb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(9, 7))\n",
    "corr = df.corr()\n",
    "sns.heatmap(corr, annot=True, fmt=\".2f\", cmap=\"coolwarm\", cbar=True, annot_kws={\"size\": 7})\n",
    "plt.title(\"Feature Correlation Heatmap\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/corr_heatmap.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "24c4a637-4ba6-40d6-ada8-43fb29da469f",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.histplot(data=df, x=\"age\", hue=\"target\", multiple=\"stack\", palette=[\"#4C72B0\", \"#C44E52\"], bins=20)\n",
    "plt.title(\"Age Distribution by Heart Disease Status\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/age_dist.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "45ace1a9-e85f-4eb0-9023-f60a63a45c9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.boxplot(x=\"target\", y=\"thalach\", hue=\"target\", data=df, palette=[\"#4C72B0\", \"#C44E52\"], legend=False)\n",
    "plt.xticks([0, 1], [\"No Disease\", \"Disease\"])\n",
    "plt.title(\"Max Heart Rate Achieved vs Heart Disease\")\n",
    "plt.xlabel(\"\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/thalach_box.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "f25dcf61-530a-45bc-b41b-59384c2497d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x=\"cp\", hue=\"target\", data=df, palette=[\"#4C72B0\", \"#C44E52\"])\n",
    "plt.title(\"Chest Pain Type vs Heart Disease\")\n",
    "plt.xlabel(\"Chest Pain Type (cp)\")\n",
    "plt.legend([\"No Disease\", \"Disease\"])\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/cp_count.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "f3a1b8f6-d143-4b57-a144-b19b47b8a53b",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"target\", axis=1)\n",
    "y = df[\"target\"]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "RESULTS['train_shape'] = X_train.shape\n",
    "RESULTS['test_shape'] = X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "2418627b-9189-4cb9-85a4-17c262624f6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_default = RandomForestClassifier(random_state=42)\n",
    "rf_default.fit(X_train, y_train)\n",
    "y_pred_default = rf_default.predict(X_test)\n",
    "\n",
    "RESULTS['default_acc'] = accuracy_score(y_test, y_pred_default)\n",
    "RESULTS['default_prec'] = precision_score(y_test, y_pred_default)\n",
    "RESULTS['default_rec'] = recall_score(y_test, y_pred_default)\n",
    "RESULTS['default_f1'] = f1_score(y_test, y_pred_default)\n",
    "RESULTS['default_report'] = classification_report(y_test, y_pred_default)\n",
    "cm_default = confusion_matrix(y_test, y_pred_default)\n",
    "\n",
    "plt.figure(figsize=(4.5, 4))\n",
    "sns.heatmap(cm_default, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False,\n",
    "            xticklabels=[\"No Disease\", \"Disease\"], yticklabels=[\"No Disease\", \"Disease\"])\n",
    "plt.title(\"Confusion Matrix - Default Random Forest\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/cm_default.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "db093d71-dd53-40bf-bd85-c4a19c1b83dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = {\n",
    "    \"n_estimators\": [100, 200, 300],\n",
    "    \"max_depth\": [None, 5, 10, 15],\n",
    "    \"min_samples_split\": [2, 5, 10],\n",
    "    \"min_samples_leaf\": [1, 2, 4],\n",
    "    \"max_features\": [\"sqrt\", \"log2\"]\n",
    "}\n",
    "\n",
    "grid_search = GridSearchCV(\n",
    "    RandomForestClassifier(random_state=42),\n",
    "    param_grid,\n",
    "    cv=5,\n",
    "    scoring=\"accuracy\",\n",
    "    n_jobs=-1\n",
    ")\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "RESULTS['best_params'] = grid_search.best_params_\n",
    "RESULTS['best_cv_score'] = grid_search.best_score_\n",
    "\n",
    "rf_tuned = grid_search.best_estimator_\n",
    "y_pred_tuned = rf_tuned.predict(X_test)\n",
    "\n",
    "RESULTS['tuned_acc'] = accuracy_score(y_test, y_pred_tuned)\n",
    "RESULTS['tuned_prec'] = precision_score(y_test, y_pred_tuned)\n",
    "RESULTS['tuned_rec'] = recall_score(y_test, y_pred_tuned)\n",
    "RESULTS['tuned_f1'] = f1_score(y_test, y_pred_tuned)\n",
    "RESULTS['tuned_report'] = classification_report(y_test, y_pred_tuned)\n",
    "cm_tuned = confusion_matrix(y_test, y_pred_tuned)\n",
    "\n",
    "plt.figure(figsize=(4.5, 4))\n",
    "sns.heatmap(cm_tuned, annot=True, fmt=\"d\", cmap=\"Greens\", cbar=False,\n",
    "            xticklabels=[\"No Disease\", \"Disease\"], yticklabels=[\"No Disease\", \"Disease\"])\n",
    "plt.title(\"Confusion Matrix - Tuned Random Forest\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/cm_tuned.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "8e504bae-ed8a-4eb5-bf6c-7bc5b59e5331",
   "metadata": {},
   "outputs": [],
   "source": [
    "importances = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)\n",
    "RESULTS['importances'] = importances\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "sns.barplot(x=importances.values, y=importances.index, hue=importances.index,\n",
    "            palette=\"viridis\", legend=False)\n",
    "plt.title(\"Feature Importance - Tuned Random Forest\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/feat_importance.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "010b8332-7c6b-4d76-b334-e6489fe611c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "dt = DecisionTreeClassifier(random_state=42)\n",
    "dt.fit(X_train, y_train)\n",
    "y_pred_dt = dt.predict(X_test)\n",
    "RESULTS['dt_acc'] = accuracy_score(y_test, y_pred_dt)\n",
    "RESULTS['dt_f1'] = f1_score(y_test, y_pred_dt)\n",
    "\n",
    "# CV comparison for stability\n",
    "rf_cv_scores = cross_val_score(rf_tuned, X, y, cv=5)\n",
    "dt_cv_scores = cross_val_score(dt, X, y, cv=5)\n",
    "RESULTS['rf_cv_mean'] = rf_cv_scores.mean()\n",
    "RESULTS['rf_cv_std'] = rf_cv_scores.std()\n",
    "RESULTS['dt_cv_mean'] = dt_cv_scores.mean()\n",
    "RESULTS['dt_cv_std'] = dt_cv_scores.std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "ae93e07d-eab5-423c-9cb9-9b3399c0dbab",
   "metadata": {},
   "outputs": [],
   "source": [
    "comp_df = pd.DataFrame({\n",
    "    \"Model\": [\"Decision Tree\", \"Random Forest (default)\", \"Random Forest (tuned)\"],\n",
    "    \"Accuracy\": [RESULTS['dt_acc'], RESULTS['default_acc'], RESULTS['tuned_acc']]\n",
    "})\n",
    "plt.figure(figsize=(6, 4))\n",
    "ax = sns.barplot(x=\"Model\", y=\"Accuracy\", hue=\"Model\", data=comp_df,\n",
    "                  palette=[\"#DD8452\", \"#4C72B0\", \"#55A868\"], legend=False)\n",
    "plt.ylim(0, 1)\n",
    "for p in ax.patches:\n",
    "    ax.annotate(f\"{p.get_height():.3f}\", (p.get_x() + p.get_width()/2, p.get_height()),\n",
    "                ha='center', va='bottom')\n",
    "plt.title(\"Model Accuracy Comparison\")\n",
    "plt.xticks(rotation=10)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/model_comparison.png\")\n",
    "plt.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d21c0360-7e82-45da-b29f-a6d6c45e0717",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(5.5, 5))\n",
    "for name, model in [(\"Random Forest (tuned)\", rf_tuned), (\"Decision Tree\", dt)]:\n",
    "    probs = model.predict_proba(X_test)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, probs)\n",
    "    roc_auc = auc(fpr, tpr)\n",
    "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {roc_auc:.3f})\")\n",
    "plt.plot([0, 1], [0, 1], linestyle=\"--\", color=\"gray\")\n",
    "plt.xlabel(\"False Positive Rate\")\n",
    "plt.ylabel(\"True Positive Rate\")\n",
    "plt.title(\"ROC Curve Comparison\")\n",
    "plt.legend(loc=\"lower right\", fontsize=8)\n",
    "plt.tight_layout()\n",
    "plt.savefig(f\"{FIGDIR}/roc_curve.png\")\n",
    "plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "caf83770-f8b6-4ce6-a5e9-1d39ade3ae4f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DONE\n",
      "{'shape': (303, 14), 'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'], 'missing': 0, 'target_counts': {1: 165, 0: 138}, 'train_shape': (242, 13), 'test_shape': (61, 13), 'default_acc': 0.8360655737704918, 'default_prec': 0.7804878048780488, 'default_rec': 0.9696969696969697, 'default_f1': 0.8648648648648649, 'default_report': '              precision    recall  f1-score   support\\n\\n           0       0.95      0.68      0.79        28\\n           1       0.78      0.97      0.86        33\\n\\n    accuracy                           0.84        61\\n   macro avg       0.87      0.82      0.83        61\\nweighted avg       0.86      0.84      0.83        61\\n', 'best_params': {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}, 'best_cv_score': np.float64(0.8473639455782311), 'tuned_acc': 0.819672131147541, 'tuned_prec': 0.7619047619047619, 'tuned_rec': 0.9696969696969697, 'tuned_f1': 0.8533333333333334, 'tuned_report': '              precision    recall  f1-score   support\\n\\n           0       0.95      0.64      0.77        28\\n           1       0.76      0.97      0.85        33\\n\\n    accuracy                           0.82        61\\n   macro avg       0.85      0.81      0.81        61\\nweighted avg       0.85      0.82      0.81        61\\n', 'dt_acc': 0.7049180327868853, 'dt_f1': 0.7428571428571429, 'rf_cv_mean': np.float64(0.834808743169399), 'rf_cv_std': np.float64(0.03852579155589034), 'dt_cv_mean': np.float64(0.755464480874317), 'dt_cv_std': np.float64(0.052902271246937915)}\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "with open(RESULTS_PATH, \"wb\") as f:\n",
    "    pickle.dump(RESULTS, f)\n",
    "\n",
    "print(\"DONE\")\n",
    "print({k: v for k, v in RESULTS.items() if k not in ['describe', 'importances']})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa35057f-5a99-4433-939a-7860bd2cee23",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
