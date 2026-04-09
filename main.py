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
st.title("Sistema de Trading IA - Analisis Historico")

# 1. FUNCION PARA DESCARGAR DATOS (Con cache para no recargar siempre)
@st.cache_data
def load_data():
    ticker = yf.Ticker("BTC-USD")
    # Ampliamos a 4 anos para tener mas contexto historico
    data = ticker.history(period="4y", interval="1d")
    data = data[['Close']].copy()
    data.rename(columns={'Close': 'price'}, inplace=True)
    return data

data = load_data()

# 2. INGENIERIA DE VARIABLES (Memoria a corto y largo plazo)
st.sidebar.header("Configuracion del Modelo")
dias_prediccion = st.sidebar.slider("Dias hacia el futuro a predecir", 1, 7, 1)

# Corto plazo
data['ma_7'] = data['price'].rolling(window=7).mean()
data['volatility_7'] = data['price'].rolling(window=7).std()

# Largo plazo (Aqui el modelo lee como subia/bajaba antiguamente)
data['ma_30'] = data['price'].rolling(window=30).mean()
data['ma_90'] = data['price'].rolling(window=90).mean()
data['tendencia_larga'] = data['ma_30'] - data['ma_90'] 

# 3. CREAR EL OBJETIVO (Target)
data['future_change'] = data['price'].shift(-dias_prediccion) - data['price']

def create_signal(change):
    umbral = 200 * dias_prediccion
    if change > umbral: return 1
    elif change < -umbral: return -1
    else: return 0

data['target'] = data['future_change'].apply(create_signal)
data.dropna(inplace=True)

# 4. PREPARAR Y ENTRENAR
features = ['ma_7', 'volatility_7', 'ma_30', 'ma_90', 'tendencia_larga']
X = data[features]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
precision = accuracy_score(y_test, preds) * 100

# 5. PREDICCION ACTUAL
ultima_fila = data.iloc[[-1]][features]
prediccion_hoy = model.predict(ultima_fila)
precio_actual = data['price'].iloc[-1]

# 6. INTERFAZ WEB (Frontend)
col1, col2, col3 = st.columns(3)
col1.metric("Precio Actual BTC", f"${precio_actual:,.2f}")
col2.metric("Precision Historica del Modelo", f"{precision:.2f}%")

if prediccion_hoy == 1:
    estado = "COMPRAR"
elif prediccion_hoy == -1:
    estado = "VENDER"
else:
    estado = "ESPERAR / HOLD"

col3.metric("Decision de la IA", estado)

st.subheader("Grafico Historico y Senales (Ultimo año)")
# Filtramos para mostrar solo el ultimo año y que el grafico sea legible
plot_data = data.iloc[-365:].copy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_data.index, plot_data['price'], label="Precio BTC", color='black', alpha=0.7)
ax.plot(plot_data.index, plot_data['ma_30'], label="Tendencia 30 dias", color='blue', alpha=0.4, linestyle='--')
ax.plot(plot_data.index, plot_data['ma_90'], label="Tendencia 90 dias", color='orange', alpha=0.4, linestyle='--')

ax.set_title("Evolucion del precio e indicadores de largo plazo")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.write("---")
st.write("Nota: El modelo ahora analiza el cruce de medias de 30 y 90 dias para entender el ciclo historico en el que se encuentra el mercado antes de tomar una decision.")
