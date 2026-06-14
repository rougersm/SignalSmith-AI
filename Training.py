import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb

df = pd.read_csv("Dataset.csv")

ber_cols = ['BER_BPSK', 'BER_QPSK', 'BER_16QAM', 'BER_64QAM']
ber_to_mod = {
    'BER_BPSK': 'BPSK',
    'BER_QPSK': 'QPSK',
    'BER_16QAM': '16QAM',
    'BER_64QAM': '64QAM'
}
min_ber_cols = df[ber_cols].idxmin(axis=1)
df['Best_Modulation'] = min_ber_cols.map(ber_to_mod)



target = 'Best_Modulation'

le = LabelEncoder()
y_encoded = le.fit_transform(df[target]) 

joblib.dump(le.classes_, 'modulation_classes.pkl')
print(f"Saved modulation classes map to 'modulation_classes.pkl'. Classes: {le.classes_}")

numerical_features = ['snr_db', 'Doppler_Hz']
categorical_features = ['ChannelType'] 

X = df[numerical_features + categorical_features].copy()
y = pd.Series(y_encoded, index=X.index)
print(f"Model is trained on only {len(X.columns)} features.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} entries.")


preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numerical_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' 
)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,  
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False, 
    eval_metric='mlogloss' 
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

print("Training the minimal feature pipeline using XGBoost...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred)

report = classification_report(y_test_decoded, y_pred_decoded, zero_division=0)

print(f"\nModel for Best_Modulation Accuracy (XGBoost, Minimal Features): {accuracy:.4f}")
print("Model for Best_Modulation Classification Report:\n", report)

joblib.dump(model_pipeline, 'random_forest_best_modulation_pipeline.pkl')
print("\nSaved the trained XGBoost pipeline to 'random_forest_best_modulation_pipeline.pkl'.")