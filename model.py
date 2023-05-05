import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('powerconsumption1.csv')

# Define the features
features = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows','Zone1PowerConsumption','Zone2PowerConsumption','Zone3PowerConsumption']
# Compute the correlation matrix
corr = df[features].corr()
# Generate a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Define the features and target
X = df[['Temperature', 'Humidity']]
y = df['Zone1PowerConsumption']

# Create the Linear Regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X, y)

# Create the Support Vector Machine model
svm_model = SVR(kernel='linear')
svm_model.fit(X, y)

# Save the models
from joblib import dump
dump(lin_reg_model, 'linear_model.joblib')
dump(svm_model, 'svm_model.joblib')


#++++++++++++
# Saving models to disk
pickle.dump(lin_reg_model, open('lin_reg_model.pkl','wb'))
pickle.dump(svm_model, open('svm_model.pkl','wb'))

# Loading models to compare the results
# lin_reg_model = pickle.load(open('lin_reg_model.pkl','rb'))
# svm_model = pickle.load(open('svm_model.pkl','rb'))
# print(lin_reg_model.predict([[2, 9, 6]]))
# print(svm_model.predict([[2, 9, 6]]))