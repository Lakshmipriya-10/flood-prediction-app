import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle


df = pd.read_csv('flood_modified.csv')


df.fillna(df.mean(), inplace=True)


features = [
     'TopographyDrainage', 'RiverManagement', 
     'ClimateChange', 
    'Siltation', 'AgriculturalPractices', 'Encroachments', 
    'IneffectiveDisasterPreparedness',
     'Landslides',  'WetlandLoss', 
     'PoliticalFactors'
]
target = 'FloodProbability'  

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(random_state=42, n_estimators=100)



xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


with open('xgb_model.pkl', 'wb') as model_file:
    pickle.dump(xgb_model, model_file)

print("Model saved as 'xgb_model.pkl'")
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')