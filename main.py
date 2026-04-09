import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuracion general
st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("Sistema de Trading IA - Alta Precision")

# ==========================================
# PANEL LATERAL (Sidebar)
# ==========================================
st.sidebar.header("Panel de Control")

monedas_disponibles = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Binance Coin": "BNB-USD",
    "XRP": "XRP-USD",
    "Dogecoin": "DOGE-USD",
    "Polkadot": "DOT-USD"
}

nombre_moneda = st.sidebar.selectbox("Selecciona la Criptomoneda", list(monedas_disponibles.keys()))
ticker_seleccionado = monedas_disponibles[nombre_moneda]

if st.sidebar.button("Actualizar Datos Ahora"):
    st.cache_data.clear()

dias_prediccion = st.sidebar.slider("Dias hacia el futuro a predecir", 1, 7, 1)
umbral_porcentaje = st.sidebar.number_input("Umbral de movimiento (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

# ==========================================
# EXTRACCION Y PROCESAMIENTO DE DATOS
# ==========================================
@st.cache_data
def load_data(ticker):
    data = yf.Ticker(ticker).history(period="4y", interval="1d")
    data = data[['Close']].copy()
    data.rename(columns={'Close': 'price'}, inplace=True)
    return data

data = load_data(ticker_seleccionado)

# 1. Indicadores de Tendencia
data['ma_7'] = data['price'].rolling(window=7).mean()
data['volatility_7'] = data['price'].rolling(window=7).std()
data['ma_30'] = data['price'].rolling(window=30).mean()
data['ma_90'] = data['price'].rolling(window=90).mean()
data['tendencia_larga'] = data['ma_30'] - data['ma_90']

# 2. RSI (Indice de Fuerza Relativa) - Detecta sobrecompra/sobreventa
delta = data['price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['rsi_14'] = 100 - (100 / (1 + rs))

# 3. MACD (Convergencia/Divergencia de Medias Moviles)
ema_12 = data['price'].ewm(span=12, adjust=False).mean()
ema_26 = data['price'].ewm(span=26, adjust=False).mean()
data['macd'] = ema_12 - ema_26
data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

# ==========================================
# CREACION DEL OBJETIVO (Target)
# ==========================================
# Convertimos el cambio a un porcentaje estandar
data['future_pct_change'] = ((data['price'].shift(-dias_prediccion) - data['price']) / data['price']) * 100

def create_signal(pct_change):
    if pct_change >= umbral_porcentaje: 
        return 1
    elif pct_change <= -umbral_porcentaje: 
        return -1
    else: 
        return 0

data['target'] = data['future_pct_change'].apply(create_signal)

# Eliminamos cualquier fila con datos incompletos para evitar bugs en el entrenamiento
data.dropna(inplace=True)

# ==========================================
# ENTRENAMIENTO DEL MODELO DE IA
# ==========================================
features = ['ma_7', 'volatility_7', 'ma_30', 'ma_90', 'tendencia_larga', 'rsi_14', 'macd', 'macd_signal']
X = data[features]
y = data['target']

# Mantenemos el orden cronologico estricto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Aumentamos la complejidad del modelo para mayor precision
model = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
precision = accuracy_score(y_test, preds) * 100

# ==========================================
# PREDICCION Y CALCULO DE CONFIANZA
# ==========================================
precio_actual = data['price'].iloc[-1]
ultima_fila = data.iloc[[-1]][features]

# Extraemos el valor exacto con para evitar errores de listas
prediccion_hoy = model.predict(ultima_fila)
probabilidades = model.predict_proba(ultima_fila)

# Extraemos el porcentaje de seguridad que tiene la IA sobre su propia decision
clases_modelo = list(model.classes_)
indice_prediccion = clases_modelo.index(prediccion_hoy)
nivel_confianza = probabilidades[indice_prediccion] * 100

# ==========================================
# INTERFAZ GRAFICA Y RESULTADOS
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Precio {nombre_moneda}", f"${precio_actual:,.4f}")
col2.metric("Precision Test", f"{precision:.2f}%")

if prediccion_hoy == 1:
    estado = "COMPRAR"
elif prediccion_hoy == -1:
    estado = "VENDER"
else:
    estado = "ESPERAR"

col3.metric("Decision IA", estado)
col4.metric("Nivel de Confianza", f"{nivel_confianza:.2f}%")

st.write("---")

# Graficos divididos para analisis profesional
st.subheader("Analisis Tecnico del Modelo")

# Mostramos solo los ultimos 200 dias para que el grafico no se comprima
plot_data = data.iloc[-200:].copy()

# AQUI ESTA LA LINEA DE SINTAXIS COMPLETAMENTE ARREGLADA
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios':})

# Grafico 1: Precios y Medias
ax1.plot(plot_data.index, plot_data['price'], label="Precio", color='black', alpha=0.8)
ax1.plot(plot_data.index, plot_data['ma_30'], label="Tendencia 30 dias", color='blue', alpha=0.5)
ax1.plot(plot_data.index, plot_data['ma_90'], label="Tendencia 90 dias", color='orange', alpha=0.5)
ax1.set_title(f"Evolucion del precio - {nombre_moneda}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Grafico 2: RSI
ax2.plot(plot_data.index, plot_data['rsi_14'], label="RSI (14)", color='purple', alpha=0.8)
ax2.axhline(70, color='red', linestyle='--', alpha=0.5) # Linea de sobrecompra
ax2.axhline(30, color='green', linestyle='--', alpha=0.5) # Linea de sobreventa
ax2.set_title("Indice de Fuerza Relativa (RSI)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
