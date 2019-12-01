# Predict-Playtime-Game
Predict the playtime for a game given its features


## Approach
Given the nature of the target variable (almost 50% of data is 0), I decided to approach the prediction problem by handling it as a classification problem and then apply regression on the same test_data to predict continuous values and later merge them as the combined result.

## Phases
- Data Pre-processing
    - Filling Missing data
    - Converting `datetime` as `int`
    - Filtering out Outliers
    - Converting `Boolean` as `int`
    - Reducing the skewness
    -  Encoding `Categorical` data
    - Split dataset as `X` and `Y`

- Phase1: Predict as Classification problem
    - Convert `Y` into two class data
    - Splitting into `test` and `train` sets
    - Feature Selection (Wrapper Method)
    - KCrossfold Validation to get best performing model
    - Grid Search on the XGBClassifier model for tuning the hyperparameter
    - Predict on the test data

- Phase2: Predict as Regression problem
    - Extracting all non-zero `Y` data
    - Splitting into `test` and `train` sets
    - Feature Selection (Wrapper Method)
    - KCrossfold Validation to get best performing model
    - Grid Search on the RandomForestRegressor model for tuning the hyperparameter
    - Predict on the test data

- Combine Results
    - Element-wise multiplication on classifiaction and regression result


## Packages used
- numpy
- pandas
- sklearn
- matplotlib