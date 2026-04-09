import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. CONFIGURACION
st.set_page_config(page_title="AI Trading Bot Final", layout="wide")
st.title("Sistema de Trading IA - Version Estable")

# 2. PANEL LATERAL
st.sidebar.header("Configuracion")
monedas = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD"
}
seleccion = st.sidebar.selectbox("Moneda", list(monedas.keys()))
ticker = monedas[seleccion]

if st.sidebar.button("Actualizar Datos"):
    st.cache_data.clear()

dias_futuros = st.sidebar.slider("Dias a predecir", 1, 7, 3)
sensibilidad = st.sidebar.slider("Sensibilidad (%)", 0.5, 5.0, 2.0)

# 3. DATOS
@st.cache_data
def get_data(symbol):
    df = yf.Ticker(symbol).history(period="4y", interval="1d")
    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)
    return df

data = get_data(ticker)

# 4. INDICADORES
data['ma7'] = data['price'].rolling(7).mean()
data['ma30'] = data['price'].rolling(30).mean()
data['ma90'] = data['price'].rolling(90).mean()
delta = data['price'].diff()
ganancia = (delta.where(delta > 0, 0)).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = ganancia / (perdida + 0.000001)
data['rsi'] = 100 - (100 / (1 + rs))
data['std'] = data['price'].rolling(7).std()

# 5. IA
data['cambio_futuro'] = ((data['price'].shift(-dias_futuros) - data['price']) / data['price']) * 100
def definir_target(cambio):
    if cambio > sensibilidad: return 1
    elif cambio < -sensibilidad: return -1
    else: return 0

data['target'] = data['cambio_futuro'].apply(definir_target)
data.dropna(inplace=True)

features = ['ma7', 'ma30', 'ma90', 'rsi', 'std']
X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 6. PREDICCION (AQUI ESTABA EL ERROR)
acc = metrics.accuracy_score(y_test, model.predict(X_test)) * 100
ultima_fila = X.iloc[[-1]]

# Extraemos el valor de forma ultra-segura
prediccion_raw = model.predict(ultima_fila)
pred_hoy = int(prediccion_raw) 

# Probabilidades
probs_raw = model.predict_proba(ultima_fila)
probabilidades = probs_raw
clases = list(model.classes_)
indice = clases.index(pred_hoy)
confianza_num = float(probabilidades[indice]) * 100

# 7. INTERFAZ
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio", f"${data['price'].iloc[-1]:,.2f}")
col2.metric("Precision", f"{acc:.2f}%")

res_map = {1: "COMPRAR", -1: "VENDER", 0: "ESPERAR"}
col3.metric("Decision IA", res_map.get(pred_hoy, "ESPERAR"))
col4.metric("Confianza", f"{confianza_num:.1f}%")

st.write("---")

# 8. GRAFICOS
st.subheader("Graficos de Analisis")
hist = data.iloc[-300:]

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(hist.index, hist['price'], color='black', label='Precio')
ax1.plot(hist.index, hist['ma30'], color='blue', alpha=0.5, label='Media 30')
ax1.grid(True, alpha=0.2)
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 2))
ax2.plot(hist.index, hist['rsi'], color='purple', label='RSI')
ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
ax2.set_ylim(0, 100)
st.pyplot(fig2)
