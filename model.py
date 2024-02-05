# Data import
import pandas as pd 

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

data_file_path = "clean_data.csv"
data = pd.read_csv(data_file_path)

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

targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']

x = data[features]
y = data[targets]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.9, random_state=0)

# Random Forest Model
# MAE: 4.76
# model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0)).fit(x_train, y_train)

# Extra Trees Regressor
# MAE: 4.64
# model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=500, min_samples_split=3, random_state=0)).fit(x_train, y_train)

# Bagging Regressor
# MAE: 4.62
# model = MultiOutputRegressor(BaggingRegressor(n_estimators=100, random_state=0, max_samples=5)).fit(x_train, y_train)

# Gradient Boosting Regressor
# MAE: 4.61
# model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5, learning_rate=0.01, random_state=0)).fit(x_train, y_train)

# AdaBoost Regressor 
# MAE: 4.52
# model = MultiOutputRegressor(AdaBoostRegressor(learning_rate=0.001, random_state=0)).fit(x_train, y_train)

# Neural Network Regressor
# MAE: 7.06
model = MultiOutputRegressor(MLPRegressor(random_state=0, max_iter=500)).fit(x_train, y_train)

pipeline = Pipeline(steps=[('model', model)])

scores = -1 * cross_val_score(pipeline, x, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE: " + str(scores.mean()))