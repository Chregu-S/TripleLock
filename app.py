import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import compute_decade_breakdown, fetch_history, run_backtest, run_grid_search
from config import STRATEGIES

st.set_page_config(page_title="Triple Lock Backtester", layout="wide")
st.title("🔒 Triple Lock – 3x ETF Backtester & Optimierer")


@st.cache_data(ttl=3600)
def load_cached_series(symbol: str):
    return fetch_history(symbol, start="1980-01-01")


# --- SIDEBAR ---
st.sidebar.header("1. Strategie & Daten")
strat_map = {f"{s['ticker']} ({s['long_asset']})": s for s in STRATEGIES}
selected_label = st.sidebar.selectbox("Strategie wählen", list(strat_map.keys()))
cfg = strat_map[selected_label]
tk = cfg["ticker"]

use_proxy = st.sidebar.checkbox(
    f"Langer Index-Proxy ({cfg['backtest_ticker']}) statt {tk}",
    value=True,
    key=f"proxy_{tk}",
)
ticker_to_load = cfg["backtest_ticker"] if use_proxy else tk

start_year = st.sidebar.slider("Startjahr", 1980, 2024, 1980)

default_div = float(cfg.get("proxy_div_yield", 0.0)) if use_proxy else 0.0
div_yield_pct = st.sidebar.slider(
    "Dividenden-Korrektur (% p.a.)",
    0.0,
    4.0,
    default_div,
    0.1,
    key=f"div_{tk}_{use_proxy}",
    help="Bei Kursindizes (^GSPC, ^NDX, ^SOX) fehlen Dividenden. Bei echten ETFs (SPY, QQQ) auf 0.0 % lassen.",
)

st.sidebar.header("2. Locks ein-/ausschalten")
use_sma = st.sidebar.checkbox("SMA-Lock aktiv", value=True, key=f"use_sma_{tk}")
use_ema = st.sidebar.checkbox("EMA-Lock aktiv", value=True, key=f"use_ema_{tk}")
has_vix = cfg.get("vol_ticker") is not None
use_vol = st.sidebar.checkbox("VIX-Lock aktiv", value=has_vix, key=f"use_vol_{tk}")
has_rvol = cfg.get("realized_vol_window") is not None
use_rvol = st.sidebar.checkbox("Realized-Vol-Lock aktiv", value=has_rvol, key=f"use_rvol_{tk}")

st.sidebar.header("3. Parameter-Tuning")
with st.sidebar.expander("SMA-Parameter", expanded=True):
    sma_period = st.number_input("Entry SMA", 20, 300, cfg["sma_period"], step=10, key=f"sma_en_{tk}")
    exit_sma_period = st.number_input(
        "Exit SMA", 20, 300, cfg.get("exit_sma_period", cfg["sma_period"]), step=10, key=f"sma_ex_{tk}"
    )
    sma_entry_mult = (
        st.slider("SMA Entry Buffer (%)", -5.0, 5.0, (cfg["sma_entry_mult"] - 1) * 100, 0.25, key=f"smab_en_{tk}") / 100
        + 1
    )
    sma_exit_mult = (
        st.slider("SMA Exit Buffer (%)", -5.0, 5.0, (cfg["sma_exit_mult"] - 1) * 100, 0.25, key=f"smab_ex_{tk}") / 100
        + 1
    )

with st.sidebar.expander("EMA-Parameter"):
    ema_fast = st.number_input("EMA Fast", 5, 100, cfg["ema_fast"], key=f"ema_f_{tk}")
    ema_slow = st.number_input("EMA Slow", 10, 200, cfg["ema_slow"], key=f"ema_s_{tk}")
    ema_entry_mult = (
        st.slider("EMA Entry Hysterese (%)", -2.0, 2.0, (cfg["ema_entry_mult"] - 1) * 100, 0.1, key=f"emab_en_{tk}")
        / 100
        + 1
    )
    ema_exit_mult = (
        st.slider("EMA Exit Hysterese (%)", -2.0, 2.0, (cfg["ema_exit_mult"] - 1) * 100, 0.1, key=f"emab_ex_{tk}") / 100
        + 1
    )

with st.sidebar.expander("Volatilitäts-Parameter"):
    vol_entry_max = st.slider(
        "VIX Entry Max", 10.0, 50.0, float(cfg.get("vol_entry_max", 25.0)), 1.0, key=f"vix_en_{tk}"
    )
    vol_exit_min = st.slider(
        "VIX Exit Min", 10.0, 60.0, float(cfg.get("vol_exit_min", 27.0)), 1.0, key=f"vix_ex_{tk}"
    )
    rvol_entry_max = st.slider(
        "Realized Vol Entry Max (%)",
        15.0,
        80.0,
        float(cfg.get("realized_vol_entry_max", 45.0)),
        1.0,
        key=f"rvol_en_{tk}",
    )
    rvol_exit_min = st.slider(
        "Realized Vol Exit Min (%)",
        15.0,
        100.0,
        float(cfg.get("realized_vol_exit_min", 55.0)),
        1.0,
        key=f"rvol_ex_{tk}",
    )

st.sidebar.header("4. Kosten & Hebel")
leverage = st.sidebar.selectbox("Hebel", [1, 2, 3], index=2)
fee_pct = st.sidebar.slider("Fee + Slippage pro Trade (%)", 0.0, 1.0, 0.15, 0.05)
ter_3x_pct = st.sidebar.slider("3x ETP TER (% p.a.)", 0.0, 2.0, 0.75, 0.05)
borrow_spread_pct = st.sidebar.slider("Finanzierungs-Spread über T-Bill (% p.a.)", 0.0, 2.0, 0.50, 0.10)

# --- DATEN LADEN ---
with st.spinner(f"Lade historische Daten für {ticker_to_load}..."):
    price_series = load_cached_series(ticker_to_load)
    rf_series = load_cached_series("^IRX")
    vol_series = load_cached_series("^VIX") if use_vol else None

params = {
    "start_date": f"{start_year}-01-01",
    "div_yield": div_yield_pct / 100.0,
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
    "cash_ter": 0.0010,  # 0.10 % für XEON
}

df, trades_df, metrics = run_backtest(price_series, rf_series, vol_series, params)

st.caption(
    f"Zeitraum: **{df.index[0].strftime('%Y-%m-%d')}** bis **{df.index[-1].strftime('%Y-%m-%d')}** "
    f"({len(df):,} Handelstage) | Underlying: **{ticker_to_load}** | "
    f"Dividenden-Korrektur: **+{div_yield_pct:.1f} % p.a.** | Cash-Proxy: **^IRX**"
)

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Einzel-Backtest & Dekaden", "🔍 Parameter-Optimierer & Heatmap"])

with tab1:
    st.subheader("Performance-Vergleich")
    st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Equity Curve (Wachstum von 1 $)")
    with col2:
        log_scale = st.checkbox("Logarithmische Skala", value=True)

    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(x=df.index, y=df["eq_strat"], name="Triple Lock", line=dict(color="#00CC96", width=2))
    )
    fig_eq.add_trace(
        go.Scatter(
            x=df.index,
            y=df["eq_3x_bh"],
            name=f"{leverage}x Buy & Hold",
            line=dict(color="#EF553B", width=1.5, dash="dot"),
        )
    )
    fig_eq.add_trace(
        go.Scatter(x=df.index, y=df["eq_1x_bh"], name="1x Buy & Hold", line=dict(color="#636EFA", width=1.5))
    )
    fig_eq.update_layout(
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Underwater / Drawdown (%)")
    dd_strat = (df["eq_strat"] / df["eq_strat"].cummax() - 1) * 100
    dd_3x = (df["eq_3x_bh"] / df["eq_3x_bh"].cummax() - 1) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=df.index, y=dd_strat, name="Triple Lock DD", fill="tozeroy", line=dict(color="#00CC96"))
    )
    fig_dd.add_trace(
        go.Scatter(x=df.index, y=dd_3x, name=f"{leverage}x B&H DD", line=dict(color="#EF553B", width=1))
    )
    fig_dd.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), height=300)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Dekaden-Check (Regime-Stabilität)")
    dec_df = compute_decade_breakdown(df)
    st.dataframe(dec_df, use_container_width=True, hide_index=True)

    with st.expander(f"Trade-Historie anzeigen ({len(trades_df)} Signale)"):
        if not trades_df.empty:
            st.dataframe(trades_df.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.info("Keine Trades im ausgewählten Zeitraum.")

with tab2:
    st.subheader("Automatische Parameter-Suche (Train / Test Split & Plateau-Heatmap)")
    st.markdown(
        "Achte darauf, dass **Train Sortino / Calmar** und **Test Sortino / Calmar** beide stark sind "
        "und in der Heatmap ein **breites grünes Plateau** entsteht (kein einzelner Ausreißer)."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sma_list = st.multiselect(
            "SMA Perioden", [100, 125, 150, 175, 200, 225, 250, 275], default=[150, 175, 200, 225, 250]
        )
        test_ema_toggle = st.checkbox("EMA-Lock (An vs. Aus) testen", value=False)
    with c2:
        rvol_exit_list = st.multiselect(
            "Realized-Vol Exit-Schwellen (%)", [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 65.0], default=[35.0, 45.0, 55.0, 65.0]
        )
        test_rvol_toggle = st.checkbox("Realized-Vol-Lock (An vs. Aus) testen", value=True)
        vix_exit_list = st.multiselect("VIX Exit-Schwellen", [27.0, 30.0, 35.0, 40.0], default=[35.0, 40.0])
        test_vix_toggle = st.checkbox("VIX-Lock (An vs. Aus) testen", value=False)
        ema_options = {"12/26": (12, 26), "19/39": (19, 39), "20/50": (20, 50), "50/100": (50, 100), "50/200": (50, 200)}
        ema_selected = st.multiselect("EMA-Paare", list(ema_options.keys()), default=["19/39", "20/50", "50/100"])
    with c3:
        split_year = st.number_input("Train/Test Split-Jahr", 1995, 2023, 2005)
        sort_metric = st.selectbox(
            "Sortieren nach",
            ["Test Calmar", "Test Sortino", "Train Sortino", "Train Calmar", "Calmar", "Sortino", "CAGR (%)"],
        )

    if st.button("🚀 Optimierung starten", type="primary"):
        grid = {
            "sma_period": sma_list or [200],
            "use_ema": [True, False] if test_ema_toggle else [False],
            "use_vol": [True, False] if test_vix_toggle else [False],
            "vol_exit_min": vix_exit_list or [35.0],
            "use_rvol": [True, False] if test_rvol_toggle else [True],
            "rvol_exit_min": rvol_exit_list or [55.0],
            "ema_pair": [ema_options[k] for k in ema_selected] if (test_ema_toggle and ema_selected) else [(ema_fast, ema_slow)],
        }

        with st.spinner(f"Berechne Grid Search für {ticker_to_load} (ab {start_year})..."):
            v_series = load_cached_series("^VIX") if True in grid["use_vol"] else None
            res_df = run_grid_search(price_series, rf_series, v_series, params, grid, split_year=split_year)

        if not res_df.empty:
            sort_col = sort_metric if sort_metric in res_df.columns else "Calmar"
            res_df = res_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
            st.success(f"Fertig! {len(res_df)} eindeutige Kombinationen ausgewertet.")

            # --- 2D PLATEAU-HEATMAP (SMA vs. Realized Vol Exit) ---
            hm_df = res_df.dropna(subset=["rvol_exit_min"])
            if len(hm_df["sma_period"].unique()) > 1 and len(hm_df["rvol_exit_min"].unique()) > 1:
                st.subheader(f"2D Plateau-Heatmap: SMA vs. Realized-Vol Exit ({sort_col})")
                pivot = hm_df.pivot_table(
                    index="rvol_exit_min", columns="sma_period", values=sort_col, aggfunc="max"
                ).sort_index(ascending=False)

                fig_hm = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=[str(c) for c in pivot.columns],
                        y=[f"{y:.0f}%" for y in pivot.index],
                        colorscale="RdYlGn",
                        text=pivot.values,
                        texttemplate="%{z:.2f}",
                        hovertemplate="SMA: %{x}<br>RVol Exit: %{y}<br>" + sort_col + ": %{z:.2f}<extra></extra>",
                    )
                )
                fig_hm.update_layout(
                    xaxis_title="SMA Periode",
                    yaxis_title="Realized-Vol Exit-Schwelle",
                    height=380,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

            st.subheader("Top 30 Kombinationen")
            st.dataframe(res_df.head(30), use_container_width=True)

            st.download_button(
                "📥 Alle Ergebnisse als CSV herunterladen",
                res_df.to_csv(index=False).encode("utf-8"),
                file_name=f"grid_search_{tk}_{start_year}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Keine gültigen Ergebnisse — prüfe das Startjahr oder die Parameter.")
