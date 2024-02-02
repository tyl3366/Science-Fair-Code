import pandas as pd 

data_file_path = "clean_data.csv"
data = pd.read_csv(data_file_path)
print(data)

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