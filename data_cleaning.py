import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


data_file_path = "raw_data.csv"
data = pd.read_csv(data_file_path)

features = ['Extraverted', 
            'Intuitive', 
            'Thinking', 
            'Judging', 
            'Assertive', 
            'SAT', 
            'Age', 
            'School Year', 
            'Sex']

targets = ['Math', 
           'English', 
           'Science', 
           'Art', 
           'History']


x = data[features]
y = data.Math

print(x.columns)
print(y)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.9, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)

obj = (x_train.dtypes == 'object')
num = (x_train.dtypes == 'float64')
object_cols = list(obj[obj].index)
number_cols = list(num[num].index)
print(object_cols)
print(number_cols)


# object_cols = [cname for cname in x_train.columns if
#                     x_train[cname].nunique() < 10 and 
#                     x_train[cname].dtype == "object"]

# number_cols = [cname for cname in x_train.columns if 
#                 x_train[cname].dtype in ['int64', 'float64']]

# my_cols = object_cols + number_cols
# X_train = x_train[my_cols].copy()
# X_valid = x_valid[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, number_cols),
        ('cat', categorical_transformer, object_cols)
    ])

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(x_train, y_train)

preds = my_pipeline.predict(x_valid)

score = mean_absolute_error(y_valid, preds)
print('MAE:', str(score))