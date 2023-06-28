import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FIFA23.csv')
X = df['Age'].values.reshape(-1, 1)
y = df['Value(£)'].values

regressor = LinearRegression()
regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[151]]))