# Sahibrimpledeep Singh
# Assignment 2 for making a prediction model for CSV data
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

le = preprocessing.LabelEncoder()
filename = 'avocado.csv'
# Treat missing values
missing_values = ["No Data", ""]
data = pd.read_csv(filename, header=0, sep=",", skip_blank_lines=True, na_values=missing_values)
# Fill Null and empty values
filtered_data = data.fillna("0")
# Convert Strings to Numerical Data
filtered_data['Type'] = le.fit_transform(filtered_data['Type'])
filtered_data['Date'] = le.fit_transform(filtered_data['Date'])
# Set target column to verify predictions
cols = [col for col in filtered_data.columns if col not in ['Type']]
test_data = filtered_data[cols]
target = filtered_data['Type']

# Separate date for testing and training
data_train, data_test, target_train, target_test = train_test_split(test_data, target, test_size=0.30, random_state=530)
svc_model = LinearSVC(random_state=0)
print(data_test.isnull().sum())
# Make predictions for the SVC model
pred = svc_model.fit(data_train, target_train).predict(data_test)
# Get the accuracy
print("Accuracy : ", accuracy_score(target_test, pred, normalize=True))
