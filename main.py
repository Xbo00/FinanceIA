import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configuración básica
st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("Sistema de Trading IA - Version Estable")

# 2. Panel Lateral
st.sidebar.header("Panel de Control")

monedas_disponibles = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Binance Coin": "BNB-USD",
    "XRP": "XRP-USD"
}
nombre_moneda = st.sidebar.selectbox("Selecciona la Criptomoneda", list(monedas_disponibles.keys()))
ticker_seleccionado = monedas_disponibles[nombre_moneda]

if st.sidebar.button("Actualizar Datos"):
    st.cache_data.clear()

dias_prediccion = st.sidebar.slider("Dias hacia el futuro", 1, 7, 1)

# 3. Carga de datos
@st.cache_data
def load_data(ticker):
    data = yf.Ticker(ticker).history(period="4y", interval="1d")
    data = data[['Close']].copy()
    data.rename(columns={'Close': 'price'}, inplace=True)
    return data

data = load_data(ticker_seleccionado)

# 4. Variables matemáticas simples (Features)
data['ma_7'] = data['price'].rolling(window=7).mean()
data['volatility_7'] = data['price'].rolling(window=7).std()
data['ma_30'] = data['price'].rolling(window=30).mean()
data['ma_90'] = data['price'].rolling(window=90).mean()
data['tendencia'] = data['ma_30'] - data['ma_90'] 

# 5. Objetivo (Target)
data['future_pct'] = (data['price'].shift(-dias_prediccion) - data['price']) / data['price']

def create_signal(pct):
    umbral = 0.02 # 2% de movimiento
    if pct > umbral: return 1
    elif pct < -umbral: return -1
    else: return 0

data['target'] = data['future_pct'].apply(create_signal)
data.dropna(inplace=True)

# 6. Entrenamiento de la IA
features = ['ma_7', 'volatility_7', 'ma_30', 'ma_90', 'tendencia']
X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. Prediccion y Resultados
precio_actual = data['price'].iloc[-1]
precision = accuracy_score(y_test, model.predict(X_test)) * 100

# Prediccion de hoy (sacamos el valor simple para evitar errores)
prediccion_raw = model.predict(data[features].iloc[[-1]])
prediccion_hoy = int(prediccion_raw)

# Interfaz
col1, col2, col3 = st.columns(3)
col1.metric(f"Precio {nombre_moneda}", f"${precio_actual:,.2f}")
col2.metric("Precision", f"{precision:.2f}%")

res_map = {1: "COMPRAR", -1: "VENDER", 0: "HOLD / ESPERAR"}
col3.metric("Decision IA", res_map[prediccion_hoy])

# 8. Grafico Unico
st.subheader("Analisis Visual")
plot_data = data.iloc[-365:].copy()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_data.index, plot_data['price'], color='black', label="Precio")
ax.plot(plot_data.index, plot_data['ma_30'], color='blue', alpha=0.5, label="Media 30d")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
