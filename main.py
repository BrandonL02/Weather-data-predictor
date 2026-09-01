
# Import pandas to convert excel data to dataframe
import pandas as pd

# Import the train-test split function and the Decision Tree Regressor model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Import mean squared error and r2 evaluation metrics to assess algorithm performance
from sklearn.metrics import mean_squared_error, r2_score

# Read excel data and convert it into a dataframe then removed unnecessary date/time columns
df = pd.read_excel("Environmental conditions dataset.xlsx")
df = df.drop(columns=['datetimeEpoch', 'month', 'dayOfWeek', 'isWeekend'] )

# Separate the feature variables and the target variable
X = df.iloc[:, 0:30]
y = df.iloc[:, 30]

# Split the data between training (80%) and testing (20%) sets with random state set for consistency between runs 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size= 0.2)

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Use the trained model to predict health risk scores for the test data
y_pred = model.predict(X_test)

# Assess algorithm performance with evaluation metrics
print(f"(Baseline Decision Tree)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}\n"
      f"R² Score: {r2_score(y_test, y_pred)}\n")

# Import random forest regression and gradient boosting regression ensemble models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import grid search and randomized search optimization techniques
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Create and train the random forest regressor (Ensemble 1)
rf_model = RandomForestRegressor(random_state=17)

# Create and train gradient boosting regressor (Ensemble 2)
gb_model = GradientBoostingRegressor(random_state=17)

### Random Forest with Grid Search (Optimization 1) ###

## Apply max_depth regularization technique (Regularization 1) 
param_grid = [{
    'max_depth': [3, 5, 7, 10, 15, 20]
}]

rf_grid_depth = GridSearchCV(rf_model,
                           cv=2,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1
)

rf_grid_depth.fit(X_train, y_train)

rf_grid_depth_pred = rf_grid_depth.best_estimator_.predict(X_test)

# Apply min_samples_leaf regularization technique (Regularization 2) 
param_grid = [{
    'min_samples_leaf': [2, 4, 6, 8, 10]
}]

rf_grid_leaf = GridSearchCV(rf_model,
                           cv=2,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1
)

rf_grid_leaf.fit(X_train, y_train)

rf_grid_leaf.fit(X_train, y_train)

rf_grid_leaf_pred = rf_grid_leaf.best_estimator_.predict(X_test)

### Random Forest with Randomized Search (Optimization 2)###

## Apply max_depth regularization technique (Regularization 1)
random_parameters = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
}

rf_random_depth = RandomizedSearchCV(rf_model,
                                   random_parameters,
                                   cv=2,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   random_state=17)

rf_random_depth.fit(X_train, y_train)

rf_random_depth_pred = rf_random_depth.best_estimator_.predict(X_test)

## Apply min_samples_leaf regularization technique (Regularization 2)
random_parameters = {
    'min_samples_leaf': [2, 3, 4, 5, 6, 8, 10]
}

rf_random_leaf = RandomizedSearchCV(rf_model,
                                   random_parameters,
                                   n_iter=5,
                                   cv=2,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   random_state=17)

rf_random_leaf.fit(X_train, y_train)
rf_random_leaf_pred = rf_random_leaf.best_estimator_.predict(X_test)

### Gradient Boosting with Grid Search (Optimization 1)###

## Apply max_depth regularization technique (Regularization 1) 
param_grid = [{
    'max_depth': [3, 5, 7, 10, 15, 20]
}]

gb_grid_depth = GridSearchCV(gb_model,
                           cv=2,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1
)

gb_grid_depth.fit(X_train, y_train)
gb_grid_depth_pred = gb_grid_depth.best_estimator_.predict(X_test)

# Apply min_samples_leaf regularization technique (Regularization 2) 
param_grid = [{
    'min_samples_leaf': [2, 4, 6, 8, 10]
}]

gb_grid_leaf = GridSearchCV(gb_model,
                           cv=2,
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1
)

gb_grid_leaf.fit(X_train, y_train)
gb_grid_leaf_pred = gb_grid_leaf.best_estimator_.predict(X_test)

### Gradient Boosting with Randomized Search (Optimization 2)###

## Apply max_depth regularization technique (Regularization 1)
random_parameters = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
}

gb_random_depth = RandomizedSearchCV(gb_model,
                                   random_parameters,
                                   cv=2,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   random_state=17)

gb_random_depth.fit(X_train, y_train)

gb_random_depth_pred = gb_random_depth.best_estimator_.predict(X_test)

## Apply min_samples_leaf regularization technique (Regularization 2)
random_parameters = {
    'min_samples_leaf': [2, 3, 4, 5, 6, 8, 10]
}

gb_random_leaf = RandomizedSearchCV(gb_model,
                                   random_parameters,
                                   n_iter=5,
                                   cv=2,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,
                                   random_state=17)

gb_random_leaf.fit(X_train, y_train)

gb_random_leaf_pred = gb_random_leaf.best_estimator_.predict(X_test)

## Evaluate the models using mean squared error and R² Score metrics

print(f"(Random Forest + Grid Search + max_depth)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, rf_grid_depth_pred)}\n"
      f"R² Score: {r2_score(y_test, rf_grid_depth_pred)}")

print(f"\n(Random Forest + Grid Search + min_samples_leaf)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, rf_grid_leaf_pred)}\n"
      f"R² Score: {r2_score(y_test, rf_grid_leaf_pred)}")

print(f"\n(Random Forest + Randomized Search + max_depth)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, rf_random_depth_pred)}\n"
      f"R² Score: {r2_score(y_test, rf_random_depth_pred)}")

print(f"\n(Random Forest + Randomized Search + min_samples_leaf)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, rf_random_leaf_pred)}\n"
      f"R² Score: {r2_score(y_test, rf_random_leaf_pred)}")

print(f"\n(Gradient Boosting + Grid Search + max_depth)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, gb_grid_depth_pred)}\n"
      f"R² Score: {r2_score(y_test, gb_grid_depth_pred)}")

print(f"\n(Gradient Boosting + Grid Search + min_samples_leaf)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, gb_grid_leaf_pred)}\n"
      f"R² Score: {r2_score(y_test, gb_grid_leaf_pred)}")

print(f"\n(Gradient Boosting + Randomized Search + max_depth)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, gb_random_depth_pred)}\n"
      f"R² Score: {r2_score(y_test, gb_random_depth_pred)}")

print(f"\n(Gradient Boosting + Randomized Search + min_samples_leaf)\n"
      f"Mean Squared Error: {mean_squared_error(y_test, gb_random_leaf_pred)}\n"
      f"R² Score: {r2_score(y_test, gb_random_leaf_pred)}\n")
