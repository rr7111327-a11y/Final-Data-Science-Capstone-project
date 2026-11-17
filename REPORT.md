# FINAL DATA SCIENCE CAPSTONE PROJECT
## Student Performance Analysis and Final Grade Prediction

**Submitted by:** Ramesh Singh  
**Course:** BCA 5th Semester  
**College:** Bhupal Nobles University, Udaipur  
**Internship:** Yuva Intern — Data Science  
**Project:** Final Data Science Capstone Project (Task 6)

## Abstract
This project demonstrates an end-to-end data science workflow for exploring student performance and estimating final grades. It includes data inspection, preprocessing, exploratory visualization, Random Forest regression, evaluation, interpretation, recommendations and limitations.

## 1. Problem statement and objectives
The project addresses the analytical question: *Can available study, attendance and academic-history variables help estimate a student's final course grade?* Objectives are to inspect the data, prepare features, train a regression model, evaluate test-set predictions, communicate results and recommend responsible educational support.

## 2. Dataset and source
The included file has 500 synthetic demonstration records. It was generated for reproducibility and is **not real student data**. The public reference dataset is the UCI Student Performance dataset, which contains student achievement information from two Portuguese schools, including Mathematics and Portuguese subject records. Source: https://archive.ics.uci.edu/dataset/320/student+performance. To run the project on the original Mathematics data, save UCI's `student-mat.csv` in the project's `data` folder; the script will prefer it over the demo file.

## 3. Data preparation
The workflow checks missing values and duplicate rows, removes duplicates, drops records without the target, validates the 0–20 target range, separates `G3`, imputes missing numeric values with the median and categorical values with the most frequent category, scales numeric features and one-hot encodes categorical features. The data is split into 80% training and 20% testing.

## 4. Techniques
A Random Forest Regressor with 300 trees and fixed random seed 42 is used. Random Forest combines multiple decision trees and averages their predictions, helping model non-linear relationships. The target is `G3`, the final grade. `G1` and `G2` are retained where available, so this should be interpreted as a late-term estimate rather than an early-warning prediction.

## 5. Results and evaluation
On the bundled synthetic demo data, the held-out test set produced:

| Metric | Result |
|---|---:|
| Records | 500 |
| Training records | 400 |
| Testing records | 100 |
| Mean Absolute Error (MAE) | 1.402 |
| Root Mean Squared Error (RMSE) | 1.756 |
| R² | 0.856 |
| Mean final grade | 10.54 |
| Median final grade | 11.0 |

MAE is the average absolute difference between actual and predicted grades. RMSE penalizes larger errors more strongly. R² describes the proportion of target variation explained by the model on this test split. These values are demonstration results only and must be recalculated when using the UCI source file.

## 6. Visualizations
The project creates a final-grade distribution chart and an actual-versus-predicted scatter plot. The diagonal reference line represents perfect predictions; points closer to it indicate smaller prediction errors.

## 7. Findings and practical recommendations
The model demonstrates how previous grades, study time, attendance and other available attributes can be explored together. A school could use such analysis to flag cases for supportive review, offer structured study plans, monitor attendance patterns and provide tutoring. These are recommendations for further investigation, not causal conclusions.

## 8. Limitations and ethical considerations
The bundled dataset is synthetic, so no real-world educational conclusion can be drawn from its metrics. The sample is limited, model results depend on the data and split, and correlation does not imply causation. Student information is sensitive: use authorized, anonymized data, restrict access, and never use automated predictions as the sole basis for academic decisions.

## 9. Future scope
Replace the demo data with the verified UCI dataset; compare several regression algorithms; use cross-validation; analyze feature importance and subgroup error; create an interactive dashboard; and seek educator feedback before operational use.

## 10. Conclusion
The capstone delivers a reproducible end-to-end workflow covering data loading, preprocessing, machine learning, evaluation, visualization and recommendations. It provides a foundation for responsible student-support analytics while clearly documenting that the bundled data is synthetic.

## References
1. UCI Machine Learning Repository, *Student Performance* dataset: https://archive.ics.uci.edu/dataset/320/student+performance
2. Cortez, P. & Silva, A. (2008), *Using data mining to predict secondary school student performance* (referenced in the UCI dataset record).

## Appendix: How to run
Install dependencies using `python -m pip install -r requirements.txt`, then run `python main.py` in the project folder. Generated files are saved in `outputs/`.
