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

# 1. Indicadores de Tend
