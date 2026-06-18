# FinanceIA

FinanceIA es una app de Streamlit que analiza criptomonedas con datos de Yahoo Finance y genera señales simples de compra, espera o venta usando un modelo Random Forest.

## Funciones

- Selección de varias criptomonedas: Bitcoin, Ethereum, Solana, Cardano, Binance Coin y XRP.
- Descarga de datos históricos con `yfinance`.
- Indicadores técnicos básicos: medias móviles, volatilidad y tendencia larga.
- Modelo de clasificación para señales `COMPRAR`, `ESPERAR` y `VENDER`.
- Evaluación con precisión, base simple, métricas por clase y matriz de confusión.
- Backtest básico sobre el tramo de test.
- Simulación de cartera por sesión con historial de compras y ventas.
- Gráfico histórico con medias móviles y señales del modelo.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run main.py
```

## Cómo funciona

La app descarga cuatro años de datos diarios para el activo seleccionado. Después calcula indicadores técnicos y entrena un modelo `RandomForestClassifier` para clasificar el movimiento esperado según el número de días elegido en el panel lateral.

La predicción actual se calcula con el último dato real disponible. El entrenamiento elimina solamente las filas que no tienen objetivo futuro conocido, evitando usar una fila antigua como si fuera la predicción más reciente.

## Ideas de mejora

- Añadir más indicadores técnicos, como RSI, MACD o volumen.
- Guardar operaciones en una base de datos o archivo.
- Incluir comisiones, spreads y slippage en el backtest.
- Separar el código en módulos para datos, modelo, cartera e interfaz.
- Desplegar la app en Streamlit Community Cloud.
