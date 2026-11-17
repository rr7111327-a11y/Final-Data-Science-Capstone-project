# Student Performance Analysis and Prediction — Task 6 Capstone

**Prepared by:** Ramesh Singh  
**Course:** BCA, 5th Semester  
**College:** Bhupal Nobles University, Udaipur  
**Internship:** Yuva Intern — Data Science  
**Project type:** Educational data analysis and regression modelling

## 1. Problem statement
Educational institutions need to identify patterns associated with academic performance and understand where students may benefit from timely support. This project explores student-related study and academic attributes and builds a regression model to estimate the final grade (`G3`, on a 0–20 scale).

## 2. Dataset and transparency
The bundled `data/student_performance_demo.csv` contains **500 reproducible synthetic demo records** created for this project. It is not real student information and must not be represented as real observations. This allows the project to run immediately without network access.

For a source-data version, download the Mathematics file (`student-mat.csv`) from the **UCI Student Performance dataset** and place it in `data/`. The script automatically prefers `student-mat.csv` when present. UCI describes data from two Portuguese secondary schools, with grades and demographic, social and school-related attributes. Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## 3. Methodology
1. Load and inspect data.
2. Check missing values, duplicates, and valid grade range.
3. Separate target `G3` from predictor variables.
4. Impute numeric and categorical missing values; scale numeric variables; one-hot encode categorical variables.
5. Split data into 80% training and 20% testing using a fixed random seed.
6. Train a Random Forest regression model.
7. Evaluate with MAE, RMSE and R².
8. Save predictions, metrics and visualizations.

## 4. Run the project
Open this folder in VS Code. Open Terminal and run:

```bash
python -m pip install -r requirements.txt
python main.py
```

On Windows, if `python` does not work, try `py main.py`.

## 5. Files generated
- `outputs/metrics.json` — evaluation metrics
- `outputs/predictions.csv` — actual and predicted test values
- `outputs/grade_distribution.png` — target distribution
- `outputs/actual_vs_predicted.png` — model comparison chart

## 6. Interpretation and recommendations
Use model outputs as a screening aid, not as a final judgement about any student. Teachers can review students with low estimated grades, check attendance and study-support needs, and offer tutoring or study-planning assistance. Recommendations should be based on individual conversations and additional context.

## 7. Limitations
- The included demo dataset is synthetic; findings from it are demonstrations, not evidence about real schools.
- A Random Forest can learn patterns but does not prove cause and effect.
- The dataset is small compared with production education systems.
- If `G1` and `G2` are included, the model is a late-term estimate; it is not an early-year prediction.
- Model accuracy can change when the original UCI dataset is used or when the train/test split changes.
- Predictions should never be used alone for high-stakes decisions.

## 8. Future scope
Use a verified real dataset; compare Linear Regression, Random Forest and Gradient Boosting; apply cross-validation and error analysis; build a dashboard; and validate recommendations with educators.

## 9. References
- Cortez, P. & Silva, A. (2008), *Using data mining to predict secondary school student performance*, cited by the UCI dataset record.
- UCI Machine Learning Repository, *Student Performance* dataset: https://archive.ics.uci.edu/dataset/320/student+performance

---
**Academic integrity note:** This is a working educational project template. Review the code, run it on your own computer, and ensure your submission follows your internship provider's rules. No project can guarantee acceptance or full marks.
