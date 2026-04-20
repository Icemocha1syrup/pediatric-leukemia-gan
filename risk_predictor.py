import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


#1.Load synthetic genomic data
print("Loading synthetic genomic data...")
df = pd.read_csv('synthetic_genomes.csv')
print(f"Loaded {len(df)} synthetic patients")


#2.Define features and target
X = df[['age', 'FLT3_ITD', 'BCR_ABL', 'TP53',
        'RUNX1', 'DNMT3A', 'NPM1', 'CEBPA',
        'WT1', 'chromosome', 'mutation_burden']]
y = df['has_leukemia']

print(f"Features: {X.columns.tolist()}")
print(f"Leukemia cases: {y.sum()} / {len(y)}")


#3.Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining: {len(X_train)} patients")
print(f"Testing:  {len(X_test)} patients")

#4.Train model
print("\nTraining risk predictor...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("Done!")


# STEP 5 — Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.1%}")
print("\nDetailed report:")
print(classification_report(y_test, predictions))


# STEP 6 — Feature importance
# Which mutations matter most?
importance = pd.DataFrame({
    'mutation': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Most important features:")
print(importance.to_string(index=False))

# STEP 7 — Save model for app.py
with open('leukemia_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to leukemia_model.pkl")

# STEP 8 — Test on a real child example
print("\n" + "="*50)
print("DEMO — Predict risk for a child:")
print("="*50)

new_child = pd.DataFrame([{
    'age':              6,
    'FLT3_ITD':         1,   # mutation present
    'BCR_ABL':          1,   # mutation present
    'TP53':             1,   # mutation present
    'RUNX1':            0,
    'DNMT3A':           0,
    'NPM1':             1,   # mutation present
    'CEBPA':            0,
    'WT1':              0,
    'chromosome':       17,
    'mutation_burden':  22.5
}])

risk = model.predict_proba(new_child)[0][1]
prediction = "⚠️ HIGH RISK — Recommend immediate testing!" if risk > 0.5 else "✅ LOW RISK"

print(f"Patient: 6 year old")
print(f"Mutations: FLT3, BCR-ABL, TP53, NPM1")
print(f"Mutation burden: 22.5")
print(f"\nLeukemia risk: {risk:.1%}")
print(f"Assessment: {prediction}")

print("\n" + "="*50)
print("SUCCESS — Risk predictor ready!")
print("="*50)