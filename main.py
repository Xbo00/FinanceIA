import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. CONFIGURACION Y TITULO
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

# 3. OBTENCION DE DATOS
@st.cache_data
def get_data(symbol):
    df = yf.Ticker(symbol).history(period="4y", interval="1d")
    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)
    return df

data = get_data(ticker)

# 4. INDICADORES TECNICOS
data['ma7'] = data['price'].rolling(7).mean()
data['ma30'] = data['price'].rolling(30).mean()
data['ma90'] = data['price'].rolling(90).mean()

delta = data['price'].diff()
ganancia = (delta.where(delta > 0, 0)).rolling(14).mean()
perdida = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = ganancia / (perdida + 0.000001)
data['rsi'] = 100 - (100 / (1 + rs))
data['std'] = data['price'].rolling(7).std()

# 5. ETIQUETADO Y ENTRENAMIENTO
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

# --- BLOQUE CRITICO CORREGIDO ---
resultado_raw = model.predict(ultima_fila)
# Usamos .item() que es la forma mas segura de sacar un solo valor de un array de numpy
pred_hoy = int(resultado_raw.item()) 

probabilidades = model.predict_proba(ultima_fila)
clases = list(model.classes_)
idx = clases.index(pred_hoy)
confianza = probabilidades[idx] * 100
# --------------------------------

# 7. INTERFAZ DE USUARIO
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio Actual", f"${data['price'].iloc[-1]:,.2f}")
c2.metric("Precision", f"{acc:.2f}%")

res_map = {1: "COMPRAR", -1: "VENDER", 0: "ESPERAR"}
decision_texto = res_map.get(pred_hoy, "ESPERAR")

c3.metric("Decision IA", decision_texto)
c4.metric("Confianza", f"{confianza:.1f}%")

st.write("---")

# 8. GRAFICOS
st.subheader("Visualizacion de Tendencias")
hist_grafico = data.iloc[-365:]

fig_price, ax_price = plt.subplots(figsize=(12, 4))
ax_price.plot(hist_grafico.index, hist_grafico['price'], color='black', label='Precio')
ax_price.plot(hist_grafico.index, hist_grafico['ma30'], color='blue', alpha=0.5, label='Media 30')
ax_price.legend()
ax_price.grid(True, alpha=0.2)
st.pyplot(fig_price)

st.write("Fuerza del Mercado (RSI)")
fig_rsi, ax_rsi = plt.subplots(figsize=(12, 2))
ax_rsi.plot(hist_grafico.index, hist_grafico['rsi'], color='purple')
ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.3)
ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.3)
ax_rsi.set_ylim(0, 100)
st.pyplot(fig_rsi)
