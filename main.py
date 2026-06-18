from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ASSET_OPTIONS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Binance Coin": "BNB-USD",
    "XRP": "XRP-USD",
}

FEATURES = ["ma_7", "volatility_7", "ma_30", "ma_90", "long_trend"]
SIGNAL_LABELS = {-1: "VENDER", 0: "ESPERAR", 1: "COMPRAR"}


st.set_page_config(page_title="FinanceIA", layout="wide")
st.title("FinanceIA - Sistema de Trading IA")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="4y", interval="1d")
    except Exception as exc:
        raise RuntimeError(f"No se pudieron descargar datos para {ticker}: {exc}") from exc

    if data.empty:
        return pd.DataFrame()

    if "Close" not in data.columns:
        return pd.DataFrame()

    data = data[["Close"]].copy()
    data.rename(columns={"Close": "price"}, inplace=True)
    data.dropna(inplace=True)
    return data


def add_indicators(data):
    data = data.copy()
    data["ma_7"] = data["price"].rolling(window=7).mean()
    data["volatility_7"] = data["price"].rolling(window=7).std()
    data["ma_30"] = data["price"].rolling(window=30).mean()
    data["ma_90"] = data["price"].rolling(window=90).mean()
    data["long_trend"] = data["ma_30"] - data["ma_90"]
    return data


def create_signal(pct_change, prediction_days):
    if pd.isna(pct_change):
        return None

    threshold = 0.01 * prediction_days
    if pct_change > threshold:
        return 1
    if pct_change < -threshold:
        return -1
    return 0


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_market(ticker, prediction_days):
    data = load_data(ticker)
    if data.empty:
        return {"error": f"No hay datos válidos para {ticker}. Prueba a actualizar más tarde."}

    indicator_data = add_indicators(data)
    usable_data = indicator_data.dropna(subset=FEATURES).copy()
    if len(usable_data) < 120:
        return {"error": "No hay suficientes datos para calcular indicadores y entrenar el modelo."}

    model_data = usable_data.copy()
    model_data["future_pct_change"] = (
        model_data["price"].shift(-prediction_days) - model_data["price"]
    ) / model_data["price"]
    model_data["target"] = model_data["future_pct_change"].apply(
        lambda value: create_signal(value, prediction_days)
    )

    training_data = model_data.dropna(subset=FEATURES + ["future_pct_change", "target"]).copy()
    training_data["target"] = training_data["target"].astype(int)

    if len(training_data) < 100:
        return {"error": "No hay suficientes filas de entrenamiento después de limpiar los datos."}

    split_index = int(len(training_data) * 0.8)
    train_data = training_data.iloc[:split_index]
    test_data = training_data.iloc[split_index:]

    if train_data["target"].nunique() < 2:
        return {"error": "El tramo de entrenamiento no tiene suficientes tipos de señal."}

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(train_data[FEATURES], train_data["target"])

    y_test = test_data["target"]
    predictions = model.predict(test_data[FEATURES])
    accuracy = accuracy_score(y_test, predictions) * 100

    baseline_label = int(train_data["target"].mode().iloc[0])
    baseline_predictions = [baseline_label] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions) * 100

    report = classification_report(
        y_test,
        predictions,
        labels=[-1, 0, 1],
        target_names=[SIGNAL_LABELS[-1], SIGNAL_LABELS[0], SIGNAL_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(
        {label: report[label] for label in [SIGNAL_LABELS[-1], SIGNAL_LABELS[0], SIGNAL_LABELS[1]]}
    ).T
    report_df = report_df[["precision", "recall", "f1-score", "support"]]

    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[-1, 0, 1]),
        index=[f"Real {SIGNAL_LABELS[value]}" for value in [-1, 0, 1]],
        columns=[f"Pred {SIGNAL_LABELS[value]}" for value in [-1, 0, 1]],
    )

    signals = usable_data.copy()
    signals["signal"] = model.predict(signals[FEATURES])

    test_eval = test_data.copy()
    test_eval["prediction"] = predictions
    test_eval["strategy_return"] = test_eval["future_pct_change"].where(
        test_eval["prediction"] == 1, 0.0
    )
    strategy_return = ((1 + test_eval["strategy_return"]).prod() - 1) * 100
    market_return = ((test_eval["price"].iloc[-1] / test_eval["price"].iloc[0]) - 1) * 100

    return {
        "error": None,
        "signals": signals,
        "training_rows": len(training_data),
        "test_rows": len(test_data),
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "report_df": report_df,
        "confusion_df": confusion_df,
        "strategy_return": strategy_return,
        "market_return": market_return,
        "buy_signals": int((test_eval["prediction"] == 1).sum()),
        "sell_signals": int((test_eval["prediction"] == -1).sum()),
        "latest_price": float(signals["price"].iloc[-1]),
        "latest_signal": int(signals["signal"].iloc[-1]),
        "latest_date": signals.index[-1].strftime("%Y-%m-%d"),
    }


def ensure_session_state():
    if "portfolio_units" not in st.session_state:
        st.session_state.portfolio_units = {ticker: 0.0 for ticker in ASSET_OPTIONS.values()}

    if "trade_history" not in st.session_state:
        st.session_state.trade_history = []


def register_trade(ticker, action, quantity, price):
    st.session_state.trade_history.append(
        {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "activo": ticker,
            "tipo": action,
            "cantidad": round(float(quantity), 6),
            "precio_usd": round(float(price), 6),
            "valor_usd": round(float(quantity) * float(price), 2),
        }
    )


def get_ai_decision(signal, has_position):
    if signal == 1:
        return "COMPRAR"
    if signal == -1 and has_position:
        return "VENDER"
    if signal == -1:
        return "ESPERAR (SIN POSICIÓN PARA VENDER)"
    return "ESPERAR"


st.sidebar.header("Panel de control")
asset_name = st.sidebar.selectbox("Selecciona la criptomoneda", list(ASSET_OPTIONS.keys()))
selected_ticker = ASSET_OPTIONS[asset_name]
prediction_days = st.sidebar.slider("Días hacia el futuro a predecir", 1, 7, 1)

if st.sidebar.button("Actualizar datos ahora"):
    st.cache_data.clear()

ensure_session_state()
if selected_ticker not in st.session_state.portfolio_units:
    st.session_state.portfolio_units[selected_ticker] = 0.0

with st.spinner("Descargando datos y entrenando el modelo..."):
    analysis = analyze_market(selected_ticker, prediction_days)

if analysis["error"]:
    st.error(analysis["error"])
    st.stop()

latest_price = analysis["latest_price"]
current_units = st.session_state.portfolio_units[selected_ticker]
has_position = current_units > 0
ai_decision = get_ai_decision(analysis["latest_signal"], has_position)

st.sidebar.subheader("Cartera (simulación)")
quantity = st.sidebar.number_input(
    "Cantidad para registrar (unidades)", min_value=0.0, value=0.01, step=0.01, format="%.4f"
)

buy_col, sell_col = st.sidebar.columns(2)
if buy_col.button("Registrar compra", use_container_width=True):
    if quantity <= 0:
        st.sidebar.error("La cantidad debe ser mayor que cero.")
    else:
        st.session_state.portfolio_units[selected_ticker] += float(quantity)
        register_trade(selected_ticker, "COMPRA", quantity, latest_price)
        st.sidebar.success("Compra registrada.")

if sell_col.button("Registrar venta", use_container_width=True):
    if quantity <= 0:
        st.sidebar.error("La cantidad debe ser mayor que cero.")
    elif quantity <= st.session_state.portfolio_units[selected_ticker]:
        st.session_state.portfolio_units[selected_ticker] -= float(quantity)
        register_trade(selected_ticker, "VENTA", quantity, latest_price)
        st.sidebar.success("Venta registrada.")
    else:
        st.sidebar.error("No puedes vender más unidades de las que tienes en cartera.")

current_units = st.session_state.portfolio_units[selected_ticker]
st.sidebar.caption(f"Unidades actuales de {asset_name}: {current_units:.4f}")

metric_cols = st.columns(4)
metric_cols[0].metric(f"Precio actual {asset_name}", f"${latest_price:,.4f}")
metric_cols[1].metric("Precisión test", f"{analysis['accuracy']:.2f}%")
metric_cols[2].metric("Base simple", f"{analysis['baseline_accuracy']:.2f}%")
metric_cols[3].metric("Decisión de la IA", ai_decision)

st.caption(
    f"Último dato disponible: {analysis['latest_date']} | "
    f"Posición actual en {asset_name}: {current_units:.4f} unidades"
)

if st.session_state.trade_history:
    st.subheader("Historial de operaciones (sesión)")
    history_df = pd.DataFrame(st.session_state.trade_history).tail(10)
    st.dataframe(history_df, use_container_width=True)

st.subheader(f"Gráfico histórico y señales ({asset_name})")
plot_data = analysis["signals"].iloc[-365:].copy()
buy_points = plot_data[plot_data["signal"] == 1]
sell_points = plot_data[plot_data["signal"] == -1]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_data.index, plot_data["price"], label=f"Precio {asset_name}", color="black", alpha=0.75)
ax.plot(plot_data.index, plot_data["ma_30"], label="Media 30 días", color="blue", alpha=0.45, linestyle="--")
ax.plot(plot_data.index, plot_data["ma_90"], label="Media 90 días", color="orange", alpha=0.45, linestyle="--")
ax.scatter(buy_points.index, buy_points["price"], label="Señal comprar", color="green", marker="^", s=55)
ax.scatter(sell_points.index, sell_points["price"], label="Señal vender", color="red", marker="v", s=55)
ax.set_title(f"Evolución del precio de {asset_name} e indicadores")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

with st.expander("Evaluación del modelo"):
    st.write(
        "La evaluación usa el 20% final de los datos como test, respetando el orden temporal. "
        "La base simple predice siempre la clase más común del entrenamiento."
    )

    eval_cols = st.columns(4)
    eval_cols[0].metric("Filas de entrenamiento", analysis["training_rows"] - analysis["test_rows"])
    eval_cols[1].metric("Filas de test", analysis["test_rows"])
    eval_cols[2].metric("Señales compra test", analysis["buy_signals"])
    eval_cols[3].metric("Señales venta test", analysis["sell_signals"])

    st.write("Métricas por clase")
    st.dataframe(analysis["report_df"].round(3), use_container_width=True)

    st.write("Matriz de confusión")
    st.dataframe(analysis["confusion_df"], use_container_width=True)

with st.expander("Backtest básico"):
    st.write("Backtest simple calculado sobre el tramo de test.")
    backtest_cols = st.columns(2)
    backtest_cols[0].metric("Retorno estrategia", f"{analysis['strategy_return']:.2f}%")
    backtest_cols[1].metric("Retorno mercado", f"{analysis['market_return']:.2f}%")
