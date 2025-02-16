Movie Box Office Revenue Prediction Using TMDB Dataset


This project is designed to predict movie box office revenue using a comprehensive dataset from TMDB. The workflow encompasses retrieving the data, cleaning and engineering features, conducting exploratory analyses, building a predictive regression model, tuning its hyperparameters, and finally preparing the model for deployment. The final prediction model is based on a Random Forest regressor that operates within a complete scikit-learn pipeline.

1. Data Acquisition and Loading

Dataset Download:
The project utilizes the KaggleHub library to download the latest version of the TMDB movies dataset automatically from Kaggle. This ensures that you are working with a current and comprehensive dataset.

Extraction and Loading:
Once downloaded, the dataset is contained in a ZIP archive. The ZIP file is extracted to reveal the CSV file, which is then loaded into a pandas DataFrame. This step verifies the success of the loading process by printing the dataset’s path and shape.

2. Dataset Verification

Column Check:
As an initial sanity check, the project prints out all the column names present in the DataFrame. This is essential for understanding which features are available, and it guides how to approach data preprocessing and feature engineering.

3. Data Preprocessing and Feature Engineering

Data Cleaning:
Essential rows missing critical values (such as revenue and budget) are removed. Additionally, missing runtime values are imputed using the median, and the release date is converted into datetime format for subsequent transformations.

Filtering:
The dataset is filtered to remove movies with extremely low values or those that have not been released. This focuses the analysis on movies that are more likely to have reliable production data.

Feature Engineering:

A helper function extracts the primary genre from a pipe-separated string in the genres column.

The release year is derived from the release date.

A profit ratio is calculated to gauge movie profitability while handling potential division-by-zero.

The average budget per primary genre is computed to capture genre-level spending trends.

These steps produce additional variables that can be useful predictors in the modeling process. Ultimately, the input features include budget, runtime, popularity, vote average, primary genre, release year, and budget by genre, while the target variable is the revenue.

4. Exploratory Data Analysis (EDA) and Visualization

Revenue Distribution:
A histogram with a kernel density estimate (using log-transformed revenue values) is generated to understand the distribution of revenue, which tends to be right-skewed.

Correlation Analysis:
A correlation heatmap is created to visualize the relationships between numerical features such as budget, revenue, popularity, and runtime.

Genre Analysis:
A boxplot shows the distribution of revenue per primary genre, offering insights into differences in box office performance across different movie genres.

5. Modeling Pipeline and Data Splitting

Pipeline Creation:
An end-to-end pipeline is set up using scikit-learn’s ColumnTransformer and Pipeline. Numerical features are standardized while categorical features (primary genre and release year) are one-hot encoded. This pipeline feeds directly into a RandomForestRegressor.

Train-Test Split:
The dataset is divided into training and testing sets. The revenue is log-transformed (using np.log1p) before modeling to stabilize variance and then reverted during evaluation.

6. Hyperparameter Tuning

GridSearchCV:
An exhaustive search is performed over a grid of hyperparameters (number of trees, max depth, and minimum samples per split) with 3-fold cross-validation. This identifies the parameters that minimize the mean absolute error.

RandomizedSearchCV:
A randomized search over a broader parameter space is also conducted. Although both searches are performed, the best model from GridSearchCV is ultimately selected for further evaluation.

7. Model Evaluation and Diagnostic Plots

Metrics Calculation:
The model’s performance is quantified using Mean Absolute Error (MAE) and the R² score. Since the predictions are on a log scale, the np.expm1 function is applied to revert them to their original scale.

Visual Diagnostics:

An Actual vs. Predicted Revenue scatter plot helps assess model accuracy.

A Residuals Distribution plot checks for bias and skew in the prediction errors.

A Learning Curve shows how performance scales with the number of training samples, indicating whether the model suffers from underfitting or overfitting.

These evaluations give a comprehensive picture of model performance and help pinpoint areas for further improvement.

8. Feature Importance Analysis

The importance of each feature is extracted from the trained RandomForestRegressor, and a bar plot is generated. This analysis highlights which features are most influential in predicting revenue, offering insights not only for model interpretability but also for potential future feature engineering.

9. Deployment Preparation

Model Saving:
The final chosen model is saved to disk using joblib, making it straightforward to deploy or use later without retraining.

Prediction Function:
A helper function is provided to load the saved model and predict revenue based on new input data. This function includes an inversion of the log transformation, ensuring predictions are returned in their original scale.

Sample Prediction:
A sample input DataFrame simulates a new movie’s attributes. The function returns a predicted revenue, demonstrating how the complete pipeline works—from data input through prediction.
