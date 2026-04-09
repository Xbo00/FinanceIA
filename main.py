import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuracion de la pagina web
st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("Sistema de Trading IA - Multi-Activos")

# ==========================================
# PANEL LATERAL (Sidebar)
# ==========================================
st.sidebar.header("Panel de Control")

# 1. Selector de Monedas
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

# 2. Boton de Refresh
if st.sidebar.button("Actualizar Datos Ahora"):
    st.cache_data.clear() # Limpia la memoria para forzar una nueva descarga

dias_prediccion = st.sidebar.slider("Dias hacia el futuro a predecir", 1, 7, 1)

# ==========================================
# NUCLEO DEL MODELO
# ==========================================
# Descarga de datos con cache (acepta el ticker como parametro)
@st.cache_data
def load_data(ticker):
    data = yf.Ticker(ticker).history(period="4y", interval="1d")
    data = data[['Close']].copy()
    data.rename(columns={'Close': 'price'}, inplace=True)
    return data

data = load_data(ticker_seleccionado)

# Variables (Features)
data['ma_7'] = data['price'].rolling(window=7).mean()
data['volatility_7'] = data['price'].rolling(window=7).std()
data['ma_30'] = data['price'].rolling(window=30).mean()
data['ma_90'] = data['price'].rolling(window=90).mean()
data['tendencia_larga'] = data['ma_30'] - data['ma_90'] 

# Objetivo (Target) basado en PORCENTAJES
data['future_pct_change'] = (data['price'].shift(-dias_prediccion) - data['price']) / data['price']

def create_signal(pct_change):
    # Umbral de 2% diario de movimiento para tomar accion
    umbral = 0.01 * dias_prediccion
    if pct_change > umbral: return 1
    elif pct_change < -umbral: return -1
    else: return 0

data['target'] = data['future_pct_change'].apply(create_signal)
data.dropna(inplace=True)

# Entrenamiento
features = ['ma_7', 'volatility_7', 'ma_30', 'ma_90', 'tendencia_larga']
X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
precision = accuracy_score(y_test, preds) * 100

# ==========================================
# INTERFAZ Y RESULTADOS
# ==========================================
precio_actual = data['price'].iloc[-1]
ultima_fila = data.iloc[[-1]][features]
prediccion_hoy = model.predict(ultima_fila)

col1, col2, col3 = st.columns(3)
col1.metric(f"Precio Actual {nombre_moneda}", f"${precio_actual:,.4f}")
col2.metric("Precision Historica", f"{precision:.2f}%")

if prediccion_hoy == 1:
    estado = "COMPRAR"
elif prediccion_hoy == -1:
    estado = "VENDER"
else:
    estado = "ESPERAR / HOLD"

col3.metric("Decision de la IA", estado)

st.subheader(f"Grafico Historico y Senales ({nombre_moneda})")
plot_data = data.iloc[-365:].copy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_data.index, plot_data['price'], label=f"Precio {nombre_moneda}", color='black', alpha=0.7)
ax.plot(plot_data.index, plot_data['ma_30'], label="Tendencia 30 dias", color='blue', alpha=0.4, linestyle='--')
ax.plot(plot_data.index, plot_data['ma_90'], label="Tendencia 90 dias", color='orange', alpha=0.4, linestyle='--')

ax.set_title(f"Evolucion del precio de {nombre_moneda} e indicadores")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
