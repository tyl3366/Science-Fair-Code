# Data import
import pandas as pd 
import numpy as np

# Data processing and scoring
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Import data
data_file_path = "clean_data_v2.csv"
data = pd.read_csv(data_file_path)

# Features to be used for prediction
features = ['Extraverted', 
            'Intuitive', 
            'Thinking', 
            'Judging',
            'Assertive',
            'SAT',
            '7',
            '8']

# Predicting targets
targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']

# Create x and y datasets
x = data[features]
y = data[targets]

# Split data
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.9, random_state=0)

# Neural Network Regressor
# MAE: 1.15
# model = MultiOutputRegressor(MLPRegressor(random_state=0, max_iter=500)).fit(x_train, y_train)

# Extra Trees Regressor
# MAE: 1.02
# model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=500, min_samples_split=3, random_state=0)).fit(x_train, y_train)

# Bagging Regressor
# MAE: 1.02
# model = MultiOutputRegressor(BaggingRegressor(n_estimators=100, random_state=0, max_samples=5)).fit(x_train, y_train)

# Gradient Boosting Regressor
# MAE: 1.02
# model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5, learning_rate=0.01, random_state=0)).fit(x_train, y_train)

# AdaBoost Regressor 
# MAE: 1.04
# model = MultiOutputRegressor(AdaBoostRegressor(learning_rate=0.001, random_state=0)).fit(x_train, y_train)

# Random Forest Model
# MAE: 1.05
# model = RandomForestRegressor(n_estimators=100, random_state=0).fit(x_train, y_train)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0)).fit(x_train, y_train)

# Math MAE: 1.09 = %78.2
# English MAE: 1.23 = %75.4
# Science MAE: 1.17 = %76.6
# Art MAE: 0.46 = %90.8
# History MAE: 1.29 = %74.2
# Overall MAE: 1.05 = %79

# Create pipline for cross validation
pipeline = Pipeline(steps=[('model', model)])

# Cross validation score
scores = -1 * cross_val_score(pipeline, x, y, cv=5, scoring='neg_mean_absolute_error')

# Print validation score
print("MAE: " + str(scores.mean()))