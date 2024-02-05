import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

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
            # 'Age'

targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']

x = data[features]
y = data[targets]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.9, random_state=0)

# Random Forest Model
# Initial MAE: 4.76
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0)).fit(x_train, y_train)

# XGBoost Model 
# Cross-Val MAE: 4.90
# model = MultiOutputRegressor(XGBRegressor(n_estimators=500, random_state=0)).fit(x_train, y_train)

my_pipeline = Pipeline(steps=[('model', model)])

scores = -1 * cross_val_score(my_pipeline, x, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE: " + str(scores.mean()))