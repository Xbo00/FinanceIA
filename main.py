import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Se importa metrics para evitar errores de referencia
from sklearn import metrics

# Configuración de la página
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

# ==========================================
# INDICADORES TECNICOS (Features)
# ==========================================
# Medias moviles
data['ma7'] = data['price'].rolling(7).mean()
data['ma30'] = data['price'].rolling(30).mean()
data['ma90'] = data['price'].rolling(90).mean()

# RSI
delta = data['price'].diff()
ganancia = (delta.where(delta > 0, 0)).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = ganancia / perdida
data['rsi'] = 100 - (100 / (1 + rs))

# Volatilidad
data['std'] = data['price'].rolling(7).std()

# ==========================================
# OBJETIVO Y ENTRENAMIENTO
# ==========================================
# Calculo de cambio porcentual futuro
data['cambio_futuro'] = ((data['price'].shift(-dias_futuros) - data['price']) / data['price']) * 100

def definir_target(cambio):
    if cambio > sensibilidad: return 1
    elif cambio < -sensibilidad: return -1
    else: return 0

data['target'] = data['cambio_futuro'].apply(definir_target)
data.dropna(inplace=True)

# Preparar variables para la IA
columnas_ia = ['ma7', 'ma30', 'ma90', 'rsi', 'std']
X = data[columnas_ia]
y = data['target']

# Split temporal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modelo Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Evaluacion
predicciones_test = model.predict(X_test)
acc = metrics.accuracy_score(y_test, predicciones_test) * 100

# ==========================================
# RESULTADOS ACTUALES
# ==========================================
ultima_fila = X.iloc[[-1]]
pred_hoy = model.predict(ultima_fila)
probabilidades = model.predict_proba(ultima_fila)

# Confianza de la prediccion
clases = list(model.classes_)
idx = clases.index(pred_hoy)
confianza = probabilidades[idx] * 100

# Interfaz de usuario
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio Actual", f"${data['price'].iloc[-1]:,.2f}")
c2.metric("Precision", f"{acc:.2f}%")

res_map = {1: "COMPRAR", -1: "VENDER", 0: "ESPERAR"}
c3.metric("Decision IA", res_map[pred_hoy])
c4.metric("Confianza", f"{confianza:.1f}%")

st.write("---")

# ==========================================
# GRAFICOS (Version Simplificada sin errores)
# ==========================================
st.subheader("Visualizacion de Tendencias")

# Solo mostramos el ultimo año para claridad
hist_grafico = data.iloc[-365:]

# Grafico de precios
fig_price, ax_price = plt.subplots(figsize=(12, 4))
ax_price.plot(hist_grafico.index, hist_grafico['price'], color='black', label='Precio')
ax_price.plot(hist_grafico.index, hist_grafico['ma30'], color='blue', alpha=0.5, label='Media 30')
ax_price.legend()
ax_price.grid(True, alpha=0.2)
st.pyplot(fig_price)

# Grafico de RSI
st.write("Fuerza del Mercado (RSI)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 2))
ax_rsi.plot(hist_grafico.index, hist_grafico['rsi'], color='purple')
ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.3)
ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.3)
ax_rsi.set_ylim(0, 100)
st.pyplot(fig_rsi)
