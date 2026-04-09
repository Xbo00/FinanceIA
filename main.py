import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. CONFIGURACION
st.set_page_config(page_title="AI Trading Bot Pro", layout="wide")
st.title("Sistema de Trading IA - Alta Precision")

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

if st.sidebar.button("Recargar Datos"):
    st.cache_data.clear()

dias_futuros = st.sidebar.slider("Dias a predecir", 1, 7, 3)
sensibilidad = st.sidebar.slider("Sensibilidad de señal (%)", 0.5, 5.0, 2.0)

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

columnas_ia = ['ma7', 'ma30', 'ma90', 'rsi', 'std']
X = data[columnas_ia]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 6. PREDICCION Y METRICAS
acc = metrics.accuracy_score(y_test, model.predict(X_test)) * 100
ultima_fila = X.iloc[[-1]]

# PREDICCION LIMPIA
pred_hoy = int(model.predict(ultima_fila))
probs = model.predict_proba(ultima_fila)
clases = list(model.classes_)
# Sacamos el valor de confianza y lo forzamos a ser un numero flotante puro
confianza_valor = float(probs[clases.index(pred_hoy)]) * 100

# 7. INTERFAZ
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio Actual", f"${data['price'].iloc[-1]:,.2f}")
c2.metric("Precision", f"{acc:.2f}%")

res_map = {1: "COMPRAR", -1: "VENDER", 0: "ESPERAR"}
c3.metric("Decision IA", res_map.get(pred_hoy, "ESPERAR"))
# Formateamos la confianza de forma mas sencilla para evitar errores
c4.metric("Confianza", str(round(confianza_valor, 1)) + "%")

st.write("---")

# 8. GRAFICOS
st.subheader("Visualizacion de Tendencias")
hist_grafico = data.iloc[-365:]

fig_p, ax_p = plt.subplots(figsize=(12, 4))
ax_p.plot(hist_grafico.index, hist_grafico['price'], color='black', label='Precio')
ax_p.plot(hist_grafico.index, hist_grafico['ma30'], color='blue', alpha=0.5, label='Media 30')
ax_p.legend()
ax_p.grid(True, alpha=0.2)
st.pyplot(fig_p)

st.write("Fuerza del Mercado (RSI)")
fig_r, ax_r = plt.subplots(figsize=(12, 2))
ax_r.plot(hist_grafico.index, hist_grafico['rsi'], color='purple')
ax_r.axhline(70, color='red', linestyle='--', alpha=0.3)
ax_r.axhline(30, color='green', linestyle='--', alpha=0.3)
ax_r.set_ylim(0, 100)
st.pyplot(fig_r)
