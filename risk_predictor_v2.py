
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv('synthetic_genomes_v2.csv')
X = df.drop(['has_leukemia'], axis=1)
y = df['has_leukemia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f'V2 Accuracy: {acc:.1%}')
print(classification_report(y_test, model.predict(X_test)))

with open('leukemia_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)
print('V2 model saved!')
