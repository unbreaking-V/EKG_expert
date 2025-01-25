import random
import streamlit as st
import pandas as pd
import pickle
from sklearn import preprocessing
import torch
import numpy as np
from collections import Counter
from resnet1d.resnet1d import ResNet1D
import matplotlib.pyplot as plt


# Wczytanie modelu do analizy EKG
model_file = "model.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_file, map_location=device)
model.eval()  

def preprocess(np_input):
    np_prep = np_input[np_input.size % 360:]
    np_prep = np_prep.reshape(-1, 360)
    np_prep = preprocessing.scale(np_prep, axis=1)
    np_prep = np.expand_dims(np_prep, 1)
    return np_prep

def predict(np_input):
    np_prep = preprocess(np_input)
    model_input = torch.tensor(np_prep, dtype=torch.float).to(device)
    batch_size = 64
    predictions = []
    for i in range(0, len(model_input), batch_size):
        batch = model_input[i:i+batch_size]
        output = model(batch)
        predictions.extend(output.argmax(dim=1).cpu().numpy().tolist())
    return Counter(predictions)

@st.cache_data
def analyze_ekg(file):
    ekg_data = np.loadtxt(file)
    predictions = predict(ekg_data)
    return ekg_data, predictions


# Funkcja sprawdzająca, czy należy udać się do lekarza
def evaluate_health(predictions):
    if predictions[1] > 0: # Jeśli model wykrył arytmie
        return "⚠️ Based on the analysis, it's recommended to consult a doctor.", True
    else: # Jeśli model nie wykrył arytmii
        return "✅ Everything seems fine, no immediate medical attention is needed.", False

# Funkcja do rysowania wykresu
def plot_ekg(data):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(data)), data, label="EKG Signal ('MLII')", color='blue', linewidth=1)
    plt.title("EKG Signal Over Time")
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal Value ('MLII')")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Interfejs Streamlit
st.markdown("<h1 style='text-align: center; color: white;'>HeartVision ❤️‍🩹</h1>", unsafe_allow_html=True)
st.image("https://media.giphy.com/media/Dpz3px6erCFeU/giphy.gif?cid=ecf05e47nx56xpyy8ao9ggrq269893494clj4lloq1tiqp09&ep=v1_gifs_related&rid=giphy.gif&ct=g", use_container_width=True)
st.header('Upload an EKG file for analysis')

with st.form("ekg_form"):
    uploaded_file = st.file_uploader("Choose an EKG file", type=['csv', 'txt'], help="Upload your EKG file in CSV or TXT format.")
    submit = st.form_submit_button("Analyze EKG")

if submit:
    if uploaded_file is not None:
        try:
            ekg_data, result = analyze_ekg(uploaded_file)
            recommendation, alert = evaluate_health(result)

            st.markdown("<h2 style='font-size: 30px;'>EKG Analysis Results</h2>", unsafe_allow_html=True)
        
            # Rysowanie wykresu
            plot_ekg(ekg_data)

            st.markdown("<h3 style='font-size: 26px;'>Recommendation</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 22px;'>{recommendation}</h4>", unsafe_allow_html=True)

            if alert:
                st.warning("Consult a doctor for further analysis.")
            else:
                st.success("No medical attention is required.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
    else:
        st.error("Please upload a valid EKG file.")
