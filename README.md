# Clinical Trial Dropout Risk Prediction

Predicting participant dropout risk in a clinical trial using exploratory data analysis, statistical testing, and machine learning.

## Objective

This project explores a synthetic clinical trial dataset to identify factors associated with participant dropout and to develop machine learning models that predict which participants are at risk of dropping out of the study. Participant retention was treated as the primary clinical objective throughout the analysis, from exploratory data analysis to model evaluation and decision threshold tuning.

## Dataset

The dataset contains 1,000 synthetic patient records simulating a Phase III hypertension clinical trial with the following variables: 

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

- **Python 3.14**
- **Pandas** / **NumPy** — data manipulation and numerical operations
- **Matplotlib** / **Seaborn** — data visualization
- **SciPy** (`scipy.stats`) — statistical hypothesis testing (t-test, chi-square test)
- **Scikit-learn** — machine learning:
  - `train_test_split`, `StandardScaler`
  - `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
  - `classification_report`, `ConfusionMatrixDisplay`, `accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `precision_recall_curve`

## Machine Learning Workflow

```
Dataset
   ↓
Exploratory Data Analysis
   ↓
Statistical Testing
   ↓
Data Preprocessing
- One-Hot Encoding
- StandardScaler
   ↓
Model Training
- Logistic Regression
- Decision Tree
- Random Forest
   ↓
Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
   ↓
Threshold Tuning
   ↓
Clinical Interpretation
```

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
- **Model performance:** Three models were trained and compared, all using class weighting to address the class imbalance (16.1% dropout rate):

  | Model | Accuracy | Recall (Dropout) | Precision (Dropout) |
  |---|---|---|---|
  | Logistic Regression | 0.570 | **0.571** | 0.178 |
  | Decision Tree | 0.775 | 0.250 | 0.226 |
  | Random Forest | **0.860** | 0.000 | 0.000 |

  Logistic Regression achieved the best recall for detecting dropout cases, while Random Forest had the highest overall accuracy but failed to identify any dropout case at all, even after testing the more granular `class_weight="balanced_subsample"`. This reflects the weak underlying signal found in the EDA: with little real separation between classes, ensemble methods tend to default to the majority class, while a linear model can be shifted more directly toward the minority class through class weighting. Decision Tree fell in between, with moderate recall but limited practical gain from balancing, since an unpruned tree can already separate the training data almost perfectly on its own.
- **Balancing:** `class_weight="balanced"` had a decisive effect on Logistic Regression (recall improved from 0.00 to 0.57) but little to no effect on the tree-based models.
- **Threshold tuning:** Lowering the decision threshold below 0.5 was explored to further prioritize recall, but precision dropped to around the dataset's baseline dropout rate (14%), indicating the model stopped discriminating meaningfully. The default threshold (0.5) was therefore kept as the practical choice, preserving reasonable recall (0.57) while still outperforming random guessing (precision 0.18). As an alternative, the F1-optimized threshold (0.558) trades recall for reliability, precision rises to 0.34 at the cost of recall dropping to 0.39, and may be preferable in contexts where minimizing false alarms matters more than catching every at-risk case.

## Limitations

- The dataset is synthetic and not clinically validated.
- The weak signal found across statistical tests limits how well any model can discriminate between classes.
- Only 161 dropout cases and a single 80/20 train-test split were used (no cross-validation), so metrics may be sensitive to the specific split.
- Only one balancing strategy (`class_weight`) was tested, without systematic hyperparameter search.
- All associations identified are observational and do not imply causation.

## Conclusion

Overall, the available clinical variables showed only weak associations with dropout, which limited how well any model could discriminate between the two classes. Among the three models tested, Logistic Regression combined with class balancing offered the best practical trade-off, identifying more than half of the dropout cases while remaining highly interpretable. Random Forest, despite its higher accuracy, was not useful for this task, since it failed to flag a single dropout case. These results reinforce that accuracy alone is a misleading metric on imbalanced clinical data, and that recall, precision, and threshold choice must be evaluated together in light of the clinical cost of missing an at-risk patient.

## Author

**Laura Neto**
