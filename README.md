# Development Pipeline #ToDos

## 1. Get Data
## 2. Get Desired Pos
- Get desired pos change
- Get desired total pos
- Get a column of encoded values for classification models
    - For using predict_proba
    - using pd.qcut
## 3. Engineer Features
- Get all TALIB features
- Generate Time features
    - Is within open 10/15/20/30 minutes
    - Is within close 10/15/20/30 minutes
    - Is last day before holiday
    - Is first day after holiday
    - Is US holiday
- TODO other packages that generate features similarly
## 4. Select Features
- Recursive feature elimination using ExtraTreesClassifier
    - Using SelectFromModel
## 5. Train Model & Cross-Validation
- Use GridSearchCV for optimizing hyperparameters
- TODO: add DEAP functionality for faster computation
- Use TimeSeriesSplit in 5 parts. Also leaving .35 part out for final testing
- The model with the highest score is reported and saved. 
## 6. Explain Model
- Use SHAP
- Use sklearn's plotting function