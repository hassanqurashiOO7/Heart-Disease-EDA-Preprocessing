# Heart Disease EDA & Preprocessing

Exploratory Data Analysis and Machine Learning preprocessing pipeline on the Heart Failure Prediction dataset, identifying key clinical indicators associated with heart disease through statistical analysis and visualization.

## Overview

This project performs an end-to-end EDA on a clinical heart disease dataset, covering data cleaning, univariate and bivariate analysis, correlation analysis, and feature preprocessing (encoding and scaling) in preparation for machine learning. The goal is to uncover which medical attributes are most strongly associated with heart disease risk.

## Dataset

- **Source:** [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** 918 rows × 12 columns
- **Target variable:** `HeartDisease` (0 = No disease, 1 = Disease present)

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | M / F |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | Fasting blood sugar (1 if > 120 mg/dl) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y / N |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |
| HeartDisease | Target (0 / 1) |

## Project Workflow

### 1. Data Cleaning
- Identified invalid zero values in `RestingBP` and `Cholesterol` (physiologically impossible)
- Replaced zero entries with the mean of valid (non-zero) values for each column
- Verified no remaining nulls or invalid entries post-cleaning

### 2. Univariate Analysis
- Distribution plots (`histplot` with KDE) for all numeric features to assess skewness and spread
- Bar plots for categorical features (Sex, ChestPainType, ST_Slope) against `HeartDisease` to compare disease prevalence across groups

![Age Distribution](images/age_distribution.png)

### 3. Bivariate & Outlier Analysis
- Box plots comparing numeric features (Cholesterol, MaxHR, Oldpeak) across `HeartDisease` classes to detect outliers and distribution shifts
- Age binned into groups (`pd.cut`) to analyze disease rate trends across age ranges

![Cholesterol by Heart Disease](images/cholesterol_boxplot.png)

### 4. Correlation Analysis
- Computed Pearson correlation matrix (`df.corr()`) across all numeric features
- Visualized with a Seaborn heatmap for quick pattern identification

![Correlation Heatmap](images/correlation_heatmap.png)

**Key findings:**

| Feature | Correlation with HeartDisease | Strength |
|---|---|---|
| Oldpeak | +0.40 | Strong positive |
| MaxHR | -0.40 | Strong negative |
| Age | +0.28 | Moderate |
| FastingBS | +0.27 | Moderate |
| RestingBP | +0.11 | Weak |
| Cholesterol | +0.09 | Weak |

`Oldpeak` and `MaxHR` emerged as the strongest linear predictors of heart disease, while `Cholesterol` showed surprisingly weak correlation — confirmed visually through box plot overlap between disease and no-disease groups.

### 5. Additional Visualizations
- Grouped bar plots (Age vs Sex, hued by HeartDisease) and violin plots (Age distribution by HeartDisease) to explore multi-variable relationships
- Subplot grids comparing distributions of cleaned features side by side

### 6. Preprocessing for ML
- **Encoding:** Categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) one-hot encoded using `pd.get_dummies(drop_first=True)` and cast to integer type
- **Scaling:** Applied `StandardScaler` from scikit-learn to normalize numeric features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) to zero mean and unit variance

```python
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)

from sklearn.preprocessing import StandardScaler

numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
```

The dataset is now fully numeric and scaled, ready as input for a classification model.

> **Note:** This notebook covers EDA and preprocessing only. No model has been trained yet — `df_encoded` is the final, ML-ready output of this pipeline.

## Tools & Libraries

- **Pandas** — data manipulation and cleaning
- **NumPy** — numerical operations
- **Matplotlib** — base plotting
- **Seaborn** — statistical visualizations (histplot, boxplot, barplot, heatmap)
- **Scikit-learn** — preprocessing (StandardScaler, encoding)

## Key Insights

- `MaxHR` and `Oldpeak` are the most clinically and statistically significant predictors of heart disease in this dataset
- Heart disease prevalence is notably higher in males (~63%) compared to females (~26%)
- `Cholesterol` alone is a weak standalone predictor despite being a commonly assumed risk factor — its distribution overlaps heavily between disease and non-disease groups
- Data quality issues (zero values in RestingBP/Cholesterol) required careful handling before any reliable analysis could proceed

## Project Structure

```
Heart-Disease-EDA-ML/
│
├── heart.csv          # Raw dataset
├── heart.ipynb         # Main analysis notebook
├── images/             # Exported plots for README
└── README.md
```

## How to Run

```bash
git clone https://github.com/hassanqurashiOO7/Heart-Disease-EDA-ML.git
cd Heart-Disease-EDA-ML
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook heart.ipynb
```

## Next Steps

- Train and evaluate classification models (Logistic Regression, Random Forest, etc.) on `df_encoded`
- Compare model performance using accuracy, precision, recall, and ROC-AUC
- Perform feature importance analysis to validate findings from the correlation study

## Author

**Noor ul Hassan**
Final-year BS Computer Science/IT student | Aspiring Data Scientist
GitHub: [@hassanqurashiOO7](https://github.com/hassanqurashiOO7)
