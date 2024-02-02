import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor

data_file_path = "clean_data.csv"
data = pd.read_csv(data_file_path)
print(data)

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
            '8']

targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']

x = data[features]
y = data.Math

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.9, random_state=0)

# model = RandomForestRegressor(n_estimators=100, random_state=0)

# model.fit(x_train, y_train)

model = XGBRegressor(n_estimators=1000, random_state=0)

model.fit(x_train, y_train,
           early_stopping_rounds=5, 
             eval_set=[(x_valid, y_valid)],
             verbose=False)

val_mae = mae(y_valid, model.predict(x_valid))

print("MAE: {}".format(val_mae))