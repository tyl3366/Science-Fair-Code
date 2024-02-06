import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import AdaBoostRegressor

import eli5
from eli5.sklearn import PermutationImportance


# Import data
data_file_path = "clean_data.csv"
data = pd.read_csv(data_file_path)

# Features to be used for prediction
features = ['Extraverted', 
            'Intuitive', 
            'Thinking', 
            'Judging', 
            'Assertive',
            'SAT',  
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',]

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

model = MultiOutputRegressor(AdaBoostRegressor(learning_rate=0.001, random_state=0)).fit(x_train, y_train)

# Create pipline for cross validation
pipeline = Pipeline(steps=[('model', model)])

# Cross validation score
scores = -1 * cross_val_score(pipeline, x, y, cv=5, scoring='neg_mean_absolute_error')

perm = PermutationImportance(model, random_state=1).fit(x_valid, y_valid)
eli5.show_weights(perm, feature_names = x_valid.columns.tolist())