import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import fetch_history, run_backtest
from config import STRATEGIES

st.set_page_config(page_title="Triple Lock Backtester", layout="wide")
st.title("🔒 Triple Lock – 3x ETF Backtester")


@st.cache_data(ttl=3600)
def load_cached_series(symbol: str):
    return fetch_history(symbol, start="1980-01-01")


# --- SIDEBAR ---
st.sidebar.header("1. Strategie & Daten")
strat_map = {f"{s['ticker']} ({s['long_asset']})": s for s in STRATEGIES}
selected_label = st.sidebar.selectbox("Strategie wählen", list(strat_map.keys()))
cfg = strat_map[selected_label]

use_proxy = st.sidebar.checkbox(
    f"Langer Index-Proxy ({cfg['backtest_ticker']}) statt {cfg['ticker']}",
    value=True,
)
ticker_to_load = cfg["backtest_ticker"] if use_proxy else cfg["ticker"]

start_year = st.sidebar.slider("Startjahr", 1980, 2024, 1990)

st.sidebar.header("2. Locks ein-/ausschalten")
use_sma = st.sidebar.checkbox("SMA-Lock aktiv", value=True)
use_ema = st.sidebar.checkbox("EMA-Lock aktiv", value=True)
has_vix = cfg.get("vol_ticker") is not None
use_vol = st.sidebar.checkbox("VIX-Lock aktiv", value=has_vix)
has_rvol = cfg.get("realized_vol_window") is not None
use_rvol = st.sidebar.checkbox("Realized-Vol-Lock aktiv", value=has_rvol)

st.sidebar.header("3. Parameter-Tuning")
with st.sidebar.expander("SMA-Parameter", expanded=True):
    sma_period = st.number_input("Entry SMA", 20, 300, cfg["sma_period"], step=10)
    exit_sma_period = st.number_input("Exit SMA", 20, 300, cfg.get("exit_sma_period", cfg["sma_period"]), step=10)
    sma_entry_mult = st.slider("SMA Entry Buffer (%)", -5.0, 5.0, (cfg["sma_entry_mult"] - 1) * 100, 0.25) / 100 + 1
    sma_exit_mult = st.slider("SMA Exit Buffer (%)", -5.0, 5.0, (cfg["sma_exit_mult"] - 1) * 100, 0.25) / 100 + 1

with st.sidebar.expander("EMA-Parameter"):
    ema_fast = st.number_input("EMA Fast", 5, 100, cfg["ema_fast"])
    ema_slow = st.number_input("EMA Slow", 10, 200, cfg["ema_slow"])
    ema_entry_mult = st.slider("EMA Entry Hysterese (%)", -2.0, 2.0, (cfg["ema_entry_mult"] - 1) * 100, 0.1) / 100 + 1
    ema_exit_mult = st.slider("EMA Exit Hysterese (%)", -2.0, 2.0, (cfg["ema_exit_mult"] - 1) * 100, 0.1) / 100 + 1

with st.sidebar.expander("Volatilitäts-Parameter"):
    vol_entry_max = st.slider("VIX Entry Max", 10.0, 50.0, float(cfg.get("vol_entry_max", 25.0)), 1.0)
    vol_exit_min = st.slider("VIX Exit Min", 10.0, 60.0, float(cfg.get("vol_exit_min", 27.0)), 1.0)
    rvol_entry_max = st.slider("Realized Vol Entry Max (%)", 15.0, 80.0, float(cfg.get("realized_vol_entry_max", 45.0)), 1.0)
    rvol_exit_min = st.slider("Realized Vol Exit Min (%)", 15.0, 100.0, float(cfg.get("realized_vol_exit_min", 55.0)), 1.0)

st.sidebar.header("4. Kosten & Hebel")
leverage = st.sidebar.selectbox("Hebel", [1, 2, 3], index=2)
fee_pct = st.sidebar.slider("Fee + Slippage pro Trade (%)", 0.0, 1.0, 0.15, 0.05)
ter_3x_pct = st.sidebar.slider("3x ETP TER (% p.a.)", 0.0, 2.0, 0.75, 0.05)
borrow_spread_pct = st.sidebar.slider("Finanzierungs-Spread über T-Bill (% p.a.)", 0.0, 2.0, 0.50, 0.10)

# --- DATEN LADEN & BACKTEST AUSFÜHREN ---
with st.spinner(f"Lade historische Daten für {ticker_to_load}..."):
    price_series = load_cached_series(ticker_to_load)
    rf_series = load_cached_series("^IRX")
    vol_series = load_cached_series("^VIX") if use_vol else None

params = {
    "start_date": f"{start_year}-01-01",
    "use_sma": use_sma,
    "use_ema": use_ema,
    "use_vol": use_vol,
    "use_rvol": use_rvol,
    "sma_period": sma_period,
    "exit_sma_period": exit_sma_period,
    "sma_entry_mult": sma_entry_mult,
    "sma_exit_mult": sma_exit_mult,
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "ema_entry_mult": ema_entry_mult,
    "ema_exit_mult": ema_exit_mult,
    "vol_entry_max": vol_entry_max,
    "vol_exit_min": vol_exit_min,
    "realized_vol_window": cfg.get("realized_vol_window", 20),
    "rvol_entry_max": rvol_entry_max,
    "rvol_exit_min": rvol_exit_min,
    "leverage": leverage,
    "fee_per_trade": fee_pct / 100.0,
    "ter_3x": ter_3x_pct / 100.0,
    "borrow_spread": borrow_spread_pct / 100.0,
    "cash_ter": 0.0010,  # 0.10% für XEON
}

df, trades_df, metrics = run_backtest(price_series, rf_series, vol_series, params)

st.caption(
    f"Zeitraum: **{df.index[0].strftime('%Y-%m-%d')}** bis **{df.index[-1].strftime('%Y-%m-%d')}** "
    f"({len(df):,} Handelstage) | Underlying: **{ticker_to_load}** | Cash-Proxy: **^IRX**"
)

# --- KENNZAHLEN-TABELLE ---
st.subheader("Performance-Vergleich")
st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)

# --- EQUITY CURVE ---
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Equity Curve (Wachstum von 1 $)")
with col2:
    log_scale = st.checkbox("Logarithmische Skala", value=True)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=df.index, y=df["eq_strat"], name="Triple Lock", line=dict(color="#00CC96", width=2)))
fig_eq.add_trace(go.Scatter(x=df.index, y=df["eq_3x_bh"], name=f"{leverage}x Buy & Hold", line=dict(color="#EF553B", width=1.5, dash="dot")))
fig_eq.add_trace(go.Scatter(x=df.index, y=df["eq_1x_bh"], name="1x Buy & Hold", line=dict(color="#636EFA", width=1.5)))
fig_eq.update_layout(
    yaxis_type="log" if log_scale else "linear",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
    height=450,
)
st.plotly_chart(fig_eq, use_container_width=True)

# --- DRAWDOWN CHART ---
st.subheader("Underwater / Drawdown (%)")
dd_strat = (df["eq_strat"] / df["eq_strat"].cummax() - 1) * 100
dd_3x = (df["eq_3x_bh"] / df["eq_3x_bh"].cummax() - 1) * 100

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=df.index, y=dd_strat, name="Triple Lock DD", fill="tozeroy", line=dict(color="#00CC96")))
fig_dd.add_trace(go.Scatter(x=df.index, y=dd_3x, name=f"{leverage}x B&H DD", line=dict(color="#EF553B", width=1)))
fig_dd.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), height=300)
st.plotly_chart(fig_dd, use_container_width=True)

# --- TRADE LOG ---
with st.expander(f"Trade-Historie anzeigen ({len(trades_df)} Signale)"):
    st.dataframe(trades_df.sort_values("Date", ascending=False), use_container_width=True)
