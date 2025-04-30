# model_pipeline.py

# 1. Imports et installation des dépendances
pip install pandas scikit-learn xgboost matplotlib streamlit joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import joblib
import streamlit as st

# 2. Définition du chemin de données
DATA_PATH = r"C:\Users\moham\Desktop\data_filtre_7_augmented.xlsx"
TARGET_RAW = 'Recurrence'                   # nom de la colonne cible brute
TARGET_BIN = 'Recurrence_bin'               # nom de la colonne cible binaire

# 3. Chargement et prétraitement des données
def load_and_preprocess(path):
    df = pd.read_excel(path)
    # Conversion de la cible en binaire
    df[TARGET_BIN] = df[TARGET_RAW].map({'yes': 1, 'no': 0})
    # Séparation features / cible
    X_raw = df.drop(columns=[TARGET_RAW, TARGET_BIN])
    y = df[TARGET_BIN]
    # Encodage one-hot des variables catégorielles
    X = pd.get_dummies(X_raw, drop_first=True)
    return X, y

# Charger et transformer
X, y = load_and_preprocess(DATA_PATH)

# 4. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entraînement du modèle XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 6. Évaluation sur le jeu de test
#   - ROC AUC
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC sur test : {roc_auc:.4f}")
#   - Rapport de classification (seuil 0.5)
y_pred = (y_prob >= 0.5).astype(int)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))
#   - Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# 7. Importance des variables
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nTop 10 des features importantes :")
print(importance_df.head(10).to_string(index=False))
# Sauvegarde de l'importance
importance_df.to_csv('feature_importances.csv', index=False)

# 8. Sauvegarde du modèle et des colonnes
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(X.columns.tolist(), 'X_columns.pkl')

# 9. Fonction de prédiction pour un nouveau patient
def predict_recurrence(patient_data: dict) -> float:
    """
    Retourne la probabilité de récidive à partir d'un dict patient_data.
    """
    X_cols = joblib.load('X_columns.pkl')
    df_new = pd.DataFrame([patient_data])
    df_enc = pd.get_dummies(df_new).reindex(columns=X_cols, fill_value=0)
    model_loaded = joblib.load('xgb_model.pkl')
    prob = model_loaded.predict_proba(df_enc)[:,1][0]
    return prob

# 10. Application Streamlit
st.title("Prédiction de récidive du cancer ")

# Chargement du DataFrame brut pour extraire les colonnes
orig_df = pd.read_excel(DATA_PATH)
# Initialisation du dict d'inputs
patient_input = {}
# Création dynamique du formulaire en excluant uniquement la colonne cible brute
for raw_col in orig_df.drop(columns=[TARGET_RAW]).columns:
    if pd.api.types.is_numeric_dtype(orig_df[raw_col]):
        # Valeur par défaut : moyenne
        default_val = float(orig_df[raw_col].dropna().mean())
        val = st.number_input(label=raw_col, value=default_val)
    else:
        options = sorted(orig_df[raw_col].dropna().unique())
        val = st.selectbox(label=raw_col, options=options)
    patient_input[raw_col] = val

if st.button("Prédire récidive"):
    prob = predict_recurrence(patient_input)
    st.success(f"Probabilité de récidive : {prob:.2%}")
    if prob >= 0.5:
        st.warning("Risque élevé : envisager un suivi renforcé et un protocole adapté.")
    else:
        st.info("Risque modéré/faible : suivre protocole standard.")

# Fin du pipeline complet
