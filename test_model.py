import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
import tensorflow as tf
from xgboost import XGBRegressor


data_file_path = "test_data.csv"
data = pd.read_csv(data_file_path)

# print(data.columns)

# bad_columns = ["Unnamed: 13", "Unnamed: 22", "Unnamed: 23", "Unnamed: 24", "Unnamed: 25", "Unnamed: 26", "Unnamed: 27"]
# for column in bad_columns:
#     data = data.drop(column, axis=1)

data = data.drop("Unnamed: 13", axis=1)

# features = ('Extraverted', 'Intuitive', 'Thinking', 'Judging', 'Assertive', 'Math',
#             'English', 'Science', 'Art', 'History', 'SAT', 'SAT Math',
#             'SAT Reading', 'ACT', 'ACT English', 'ACT Math', 'ACT Reading',
#             'ACT Science', 'Age', 'School Year', 'Sex')

testing_features = ['Extraverted', 'Intuitive', 'Thinking', 'Judging', 'Assertive', 'Age', 'School Year']

targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']

y = data.English
X = data[testing_features]

x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, random_state=0)

# Did better with Random Forest than XGBoost
# model = XGBRegressor(n_estimators=1000, random_state=0)

# model.fit(x_train, y_train,
#            early_stopping_rounds=5, 
#              eval_set=[(x_valid, y_valid)],
#              verbose=False)

model = RandomForestRegressor(n_estimators=500, random_state=0)

model.fit(x_train, y_train)

val_mae = mae(y_valid, model.predict(x_valid))

print("MAE: {}".format(val_mae))