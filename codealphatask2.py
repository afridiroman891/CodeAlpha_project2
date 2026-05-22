
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_path = "D:\\heart_disease_dataset1.csv"
df = pd.read_csv(data_path)

print("--- Dataset Preview ---")
print(df.head())
print("\n--- Target Variable Distribution ---")
print(df['ExerciseAngina'].value_counts())



X = df.drop(columns=['ExerciseAngina'])
y = df['ExerciseAngina']


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()



preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_transformed, y_train)


y_pred = rf_model.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + encoded_cat_features

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- Top Feature Importances ---")
for f in range(len(all_features)):
    print(f"{f + 1}. {all_features[indices[f]]}: {importances[indices[f]]:.4f}")