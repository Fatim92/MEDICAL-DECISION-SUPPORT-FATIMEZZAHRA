import streamlit as st
import pandas as pd
import os
import joblib

# --- CSS pour le style ---
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Helvetica', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
    }
    .stTable table {
        background-color: #ecf0f1;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Charger le modèle ---
MODEL_PATH = "src/model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- Titre et description ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🩺 Medical Decision Support - Obesity Risk")
st.write("Remplissez les informations du patient pour prédire le risque d'obésité.")

# --- Formulaire utilisateur ---
age = st.number_input("Âge", min_value=0, max_value=120, value=22)
height = st.number_input("Taille (cm)", min_value=50, max_value=250, value=158)
weight = st.number_input("Poids (kg)", min_value=10, max_value=300, value=56)
family_history = st.selectbox("Antécédents familiaux d'obésité ?", ["Non", "Oui"])
activity_level = st.selectbox("Niveau d'activité physique", ["Faible", "Moyen", "Élevé"])

# --- Bouton de prédiction ---
submit_button = st.button("Prédire le risque d'obésité")

# --- Prédiction ---
if submit_button:
    if model is None:
        st.warning("⚠️ Le modèle n'est pas disponible.")
    else:
        input_dict = {
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "Family_History": [family_history],
            "Activity_Level": [activity_level]
        }
        input_df = pd.DataFrame(input_dict)
        input_df = pd.get_dummies(input_df)
        # Assurer que toutes les colonnes du modèle existent
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # Faire la prédiction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        class_names = model.classes_

        # Affichage avec style
        st.markdown(f"<h2 style='color:#e74c3c;'>⚠️ Risque prédite : {prediction}</h2>", unsafe_allow_html=True)

        st.subheader("Probabilités par classe :")
        prob_df = pd.DataFrame({
            "Classe": class_names,
            "Probabilité (%)": [round(p*100, 2) for p in probabilities]
        }).sort_values(by="Probabilité (%)", ascending=False)

        # Ajouter des barres colorées pour chaque classe
        for idx, row in prob_df.iterrows():
            st.markdown(f"{row['Classe']} : {row['Probabilité (%)']}%")
            st.progress(int(row['Probabilité (%)']))

st.markdown("</div>", unsafe_allow_html=True)
