import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Configuracion de la pagina
st.set_page_config(page_title="AI Trading Bot Pro", layout="wide")
st.title("Sistema de Trading IA - Alta Precision")

# ==========================================
# PANEL LATERAL (Sidebar)
# ==========================================
st.sidebar.header("Configuracion")

monedas = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Binance Coin": "BNB-USD",
    "XRP": "XRP-USD",
    "Dogecoin": "DOGE-USD"
}

seleccion = st.sidebar.selectbox("Moneda", list(monedas.keys()))
ticker = monedas[seleccion]

if st.sidebar.button("Recargar Datos"):
    st.cache_data.clear()

dias_futuros = st.sidebar.slider("Dias a predecir", 1, 7, 3)
sensibilidad = st.sidebar.slider("Sensibilidad de señal (%)", 0.5, 5.0, 2.0)

# ==========================================
# OBTENCION DE DATOS
# ==========================================
@st.cache_data
def get_data(symbol):
    df = yf.Ticker(symbol).history(period="4y", interval="1d")
    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)
    return df

data = get_data(ticker)

# INDICADORES
data['ma7'] = data['price'].rolling(7).mean()
data['ma30'] = data['price'].rolling(30).mean()
data['ma90'] = data['price'].rolling(90).mean()
delta = data['price'].diff()
ganancia = (delta.where(delta > 0, 0)).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = ganancia / (perdida + 0.000001) # Evita division por cero
data['rsi'] = 100 - (100 / (1 + rs))
data['std'] = data['price'].rolling(7).std()

# OBJETIVO
data['cambio_futuro'] = ((data['price'].shift(-dias_futuros) - data['price']) / data['price']) * 100

def definir_target(cambio):
    if cambio > sensibilidad: return 1
    elif cambio < -sensibilidad: return -1
    else: return 0

data['target'] = data['cambio_futuro'].apply(definir_target)
data.dropna(inplace=True)

# MODELO
columnas_ia = ['ma7', 'ma30', 'ma90', 'rsi', 'std']
X = data[columnas_ia]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train
