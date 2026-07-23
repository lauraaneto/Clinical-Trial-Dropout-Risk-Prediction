# Clinical Trial Dropout Risk Prediction

Predicting participant dropout risk in a clinical trial using exploratory data analysis, statistical testing, and machine learning.

## Objective

This project explores a synthetic clinical trial dataset to identify factors associated with participant dropout and to build machine learning models capable of predicting which patients are at risk of abandoning the study. Dropout prevention is treated as the guiding clinical priority throughout the analysis — from exploratory data analysis to model evaluation and decision threshold tuning.

## Dataset

The dataset used is a **synthetic** clinical trial dataset simulating a Phase III hypertension treatment study, containing 1,000 patient records with the following variables:

| Column | Description |
|---|---|
| `Subject_ID` | Unique patient identifier |
| `Site_ID` | Clinical site identifier |
| `Age` | Patient age |
| `Gender` | Patient gender |
| `Enrollment_Date` | Date the patient enrolled in the trial |
| `Treatment_Group` | Placebo, Drug A, or Drug B |
| `Adverse_Events` | Number of adverse events reported |
| `Dropout` | Target variable (1 = dropped out, 0 = completed) |
| `Systolic_BP` | Systolic blood pressure |
| `Diastolic_BP` | Diastolic blood pressure |
| `Cholesterol_Level` | Cholesterol level |

> ⚠️ **Note:** The data is synthetic and used for portfolio/demonstration purposes only. It should not be interpreted as clinically validated or used to draw real-world clinical conclusions.

## Technologies

- **Python 3**
- **Pandas** / **NumPy** — data manipulation and numerical operations
- **Matplotlib** / **Seaborn** — data visualization
- **SciPy** (`scipy.stats`) — statistical hypothesis testing (t-test, chi-square test)
- **Scikit-learn** — machine learning:
  - `train_test_split`, `StandardScaler`
  - `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
  - `classification_report`, `ConfusionMatrixDisplay`, `accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `precision_recall_curve`

## Notebook Structure

1. **Data Preparation**
   - Dataset structure and descriptive statistics
2. **Exploratory Data Analysis (EDA)**
   - Target variable analysis (class distribution)
   - Categorical variables (Gender, Treatment Group)
   - Numerical variables (distributions, correlation, boxplots)
3. **Descriptive Variables**
   - Statistical significance testing (t-tests, chi-square tests)
4. **Model Training**
   - One-Hot Encoding of categorical variables
   - Logistic Regression (baseline and balanced)
   - Decision Tree
   - Random Forest
   - Models Comparison (accuracy, recall, precision)
   - Decision Threshold Tuning (precision-recall trade-off, F1 optimization)
5. **Limitations**
6. **Conclusion**

## Key Findings

- **EDA:** Associations between the available clinical variables and dropout were generally weak. Among the numerical variables tested, only **Diastolic Blood Pressure** showed a statistically significant difference between groups (p = 0.027). Gender, Treatment Group, and enrollment month showed no significant association with dropout.
- **Model performance:** Three models were trained and compared, all using class weighting to address the class imbalance (~16% dropout rate):

  | Model | Accuracy | Recall (Dropout) | Precision (Dropout) |
  |---|---|---|---|
  | Logistic Regression | moderate | **0.57** | 0.18 |
  | Decision Tree | moderate | 0.25 | — |
  | Random Forest | **highest (0.86)** | 0.00 | — |

  Logistic Regression — the simplest model — achieved the best recall for detecting dropout cases, while Random Forest had the highest overall accuracy but failed to identify any dropout case at all. This reflects the weak underlying signal found in the EDA: with little real separation between classes, ensemble methods tend to default to the majority class, while a linear model can be pushed more directly toward the minority class through class weighting.
- **Balancing & threshold tuning:** `class_weight="balanced"` had a decisive effect on Logistic Regression (recall improved from 0.00 to 0.57) but little to no effect on the tree-based models. Threshold tuning showed that lowering the decision threshold below 0.5 degraded precision toward the dataset's baseline dropout rate (~14%), meaning the model stopped discriminating meaningfully — so the default threshold (0.5) was kept as the practical choice.

## Limitations

- The dataset is synthetic and not clinically validated.
- The weak signal found across statistical tests limits how well any model can discriminate between classes.
- Only 161 dropout cases and a single 80/20 train-test split were used (no cross-validation), so metrics may be sensitive to the specific split.
- Only one balancing strategy (`class_weight`) was tested, without systematic hyperparameter search.
- All associations identified are observational and do not imply causation.

## Conclusion

Overall, the available clinical variables showed only weak associations with dropout, which limited how well any model could discriminate between the two classes. Among the three models tested, Logistic Regression combined with class balancing offered the best practical trade-off — identifying over half of the dropout cases while remaining more interpretable than the tree-based alternatives. Random Forest, despite its higher accuracy, was not useful for this task, since it failed to flag a single dropout case. These results reinforce that accuracy alone is a misleading metric on imbalanced clinical data, and that recall, precision, and threshold choice must be evaluated together in light of the clinical cost of missing an at-risk patient.

## Author

**Laura Neto**
