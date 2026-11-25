import itertools

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_history(symbol: str, start: str = "1980-01-01") -> pd.Series:
    """Lädt bereinigte Schlusskurse mit normalisiertem Datums-Index (ohne Timezone)."""
    df = yf.Ticker(symbol).history(start=start, auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Keine historischen Daten für {symbol} gefunden.")
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def run_backtest(
    price: pd.Series,
    rf_series: pd.Series,
    vol_series: pd.Series | None,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Führt die Triple-Lock-Strategie historisch aus (Execution am Folgetag t+1).
    Inkl. optionaler Dividenden-Korrektur für Kursindizes (^GSPC, ^NDX, ^SOX).
    """
    df = pd.DataFrame({"raw_price": price})
    raw_ret = df["raw_price"].pct_change().fillna(0.0)
    div_daily = params.get("div_yield", 0.0) / 252.0

    # Total-Return-Rendite & synthetischer Total-Return-Kurs für konsistente SMAs/EMAs
    df["ret_1x"] = raw_ret + div_daily
    df["price"] = (1.0 + df["ret_1x"]).cumprod() * df["raw_price"].iloc[0]

    # Risikofreier Zins (^IRX in % p.a. -> Dezimal)
    rf_aligned = rf_series.reindex(df.index).ffill().bfill().fillna(0.0) / 100.0
    df["rf_annual"] = rf_aligned.clip(lower=0.0)

    if vol_series is not None:
        df["vol_idx"] = vol_series.reindex(df.index).ffill()
    else:
        df["vol_idx"] = np.nan

    # Indikatoren berechnen
    df["sma_entry"] = df["price"].rolling(params["sma_period"]).mean()
    df["sma_exit"] = df["price"].rolling(params["exit_sma_period"]).mean()
    df["ema_fast"] = df["price"].ewm(span=params["ema_fast"], adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=params["ema_slow"], adjust=False).mean()

    rvol_win = params.get("realized_vol_window", 20)
    df["rvol"] = raw_ret.rolling(rvol_win).std() * np.sqrt(252) * 100.0

    # Auf Startdatum und gültige Indikatoren filtern
    df = df.dropna(subset=["sma_entry", "sma_exit", "ema_fast", "ema_slow"]).copy()
    if params.get("start_date"):
        df = df.loc[df.index >= pd.Timestamp(params["start_date"])].copy()

    # Synthetische 3x-Rendite & Cash-Rendite (XEON-Proxy)
    leverage = params["leverage"]
    ter_3x = params["ter_3x"]
    borrow_spread = params["borrow_spread"]
    cash_ter = params["cash_ter"]

    daily_drag_3x = (ter_3x + (leverage - 1) * (df["rf_annual"] + borrow_spread)) / 252.0
    df["ret_3x"] = (leverage * df["ret_1x"] - daily_drag_3x).clip(lower=-0.9999)
    df["ret_cash"] = ((df["rf_annual"] - cash_ter) / 252.0).clip(lower=0.0)

    # State-Machine (Signal am Tag t -> Position gilt für Tag t+1)
    n = len(df)
    positions = np.zeros(n, dtype=int)  # 0 = CASH, 1 = LONG
    trades = []

    use_sma = params["use_sma"]
    use_ema = params["use_ema"]
    use_vol = params["use_vol"]
    use_rvol = params["use_rvol"]

    prices = df["price"].to_numpy()
    sma_en = df["sma_entry"].to_numpy() * params["sma_entry_mult"]
    sma_ex = df["sma_exit"].to_numpy() * params["sma_exit_mult"]
    ema_f = df["ema_fast"].to_numpy()
    ema_en = df["ema_slow"].to_numpy() * params["ema_entry_mult"]
    ema_ex = df["ema_slow"].to_numpy() * params["ema_exit_mult"]
    vol_arr = df["vol_idx"].to_numpy()
    rvol_arr = df["rvol"].to_numpy()
    dates = df.index

    curr_pos = 0
    for i in range(n - 1):
        if curr_pos == 0:  # CASH -> Prüfe Entry
            c_sma = (not use_sma) or (prices[i] > sma_en[i])
            c_ema = (not use_ema) or (ema_f[i] > ema_en[i])
            c_vol = (not use_vol) or np.isnan(vol_arr[i]) or (vol_arr[i] < params["vol_entry_max"])
            c_rvol = (not use_rvol) or np.isnan(rvol_arr[i]) or (rvol_arr[i] < params["rvol_entry_max"])

            if c_sma and c_ema and c_vol and c_rvol:
                curr_pos = 1
                trades.append({"Date": dates[i + 1], "Action": "BUY", "Reason": "Entry-Locks erfüllt"})
        else:  # LONG -> Prüfe Exit
            reasons = []
            if use_sma and prices[i] < sma_ex[i]:
                reasons.append("SMA-Exit")
            if use_ema and ema_f[i] < ema_ex[i]:
                reasons.append("EMA-Exit")
            if use_vol and not np.isnan(vol_arr[i]) and vol_arr[i] > params["vol_exit_min"]:
                reasons.append("VIX-Exit")
            if use_rvol and not np.isnan(rvol_arr[i]) and rvol_arr[i] > params["rvol_exit_min"]:
                reasons.append("RealizedVol-Exit")

            if reasons:
                curr_pos = 0
                trades.append({"Date": dates[i + 1], "Action": "SELL", "Reason": ", ".join(reasons)})

        positions[i + 1] = curr_pos

    df["position"] = positions
    df["trade_flag"] = df["position"].diff().abs().fillna(0.0)

    # Strategie-Return inkl. Fees/Slippage bei jedem Wechsel
    fee = params["fee_per_trade"]
    raw_strat_ret = np.where(df["position"] == 1, df["ret_3x"], df["ret_cash"])
    df["ret_strat"] = (1.0 + raw_strat_ret) * (1.0 - df["trade_flag"] * fee) - 1.0

    # Equity Curves (Basis 1.0)
    df["eq_strat"] = (1.0 + df["ret_strat"]).cumprod()
    df["eq_3x_bh"] = (1.0 + df["ret_3x"]).cumprod()
    df["eq_1x_bh"] = (1.0 + df["ret_1x"]).cumprod()

    metrics = {
        "Triple Lock": compute_metrics(df["ret_strat"], df["eq_strat"], df["rf_annual"], df["position"]),
        "3x Buy & Hold": compute_metrics(df["ret_3x"], df["eq_3x_bh"], df["rf_annual"]),
        "1x Buy & Hold": compute_metrics(df["ret_1x"], df["eq_1x_bh"], df["rf_annual"]),
    }

    trades_df = pd.DataFrame(trades)
    return df, trades_df, metrics


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    rf_annual: pd.Series,
    position: pd.Series | None = None,
) -> dict:
    years = len(returns) / 252.0
    total_return = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0

    max_dd = ((equity / equity.cummax()) - 1.0).min()
    excess_ret = returns - (rf_annual / 252.0)
    vol_annual = returns.std() * np.sqrt(252)
    sharpe = (excess_ret.mean() * 252.0) / vol_annual if vol_annual > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    res = {
        "Total Return": f"{total_return * 100:,.1f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Vola (p.a.)": f"{vol_annual * 100:.1f}%",
    }
    if position is not None:
        res["Time in Market"] = f"{position.mean() * 100:.1f}%"
        res["Trades (Wechsel)"] = int(position.diff().abs().sum())
    else:
        res["Time in Market"] = "100.0%"
        res["Trades (Wechsel)"] = 1
    return res


def compute_numeric_metrics(
    returns: pd.Series,
    equity: pd.Series,
    rf_annual: pd.Series,
    position: pd.Series,
) -> dict:
    """Reine Zahlen-Metriken zum Sortieren im Grid Search."""
    years = len(returns) / 252.0
    cagr = (equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0
    max_dd = ((equity / equity.cummax()) - 1.0).min()
    excess_ret = returns - (rf_annual / 252.0)
    vol = returns.std() * np.sqrt(252)
    sharpe = (excess_ret.mean() * 252.0) / vol if vol > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    trades = int(position.diff().abs().sum())
    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Max DD (%)": round(max_dd * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Calmar": round(calmar, 2),
        "Trades": trades,
        "Time in Mkt (%)": round(position.mean() * 100, 1),
    }


def run_grid_search(
    price: pd.Series,
    rf_series: pd.Series,
    vol_series: pd.Series | None,
    base_params: dict,
    grid: dict,
    split_year: int = 2016,
) -> pd.DataFrame:
    """
    Testet alle sinnvollen Parameter-Kombinationen im Grid und teilt in Train (< split_year) & Test (>= split_year).
    """
    keys = list(grid.keys())
    raw_combinations = list(itertools.product(*grid.values()))

    # Duplikate entfernen (z. B. VIX-Schwellen nicht variieren, wenn use_vol=False)
    seen = set()
    combinations = []
    for combo in raw_combinations:
        d = dict(zip(keys, combo))
        if not d.get("use_vol", True) and "vol_exit_min" in d:
            d["vol_exit_min"] = None
        if not d.get("use_rvol", True) and "rvol_exit_min" in d:
            d["rvol_exit_min"] = None
        sig = tuple(d.items())
        if sig not in seen:
            seen.add(sig)
            combinations.append(d)

    results = []
    split_ts = pd.Timestamp(f"{split_year}-01-01")

    for combo_dict in combinations:
        p = base_params.copy()
        for k, v in combo_dict.items():
            if v is not None:
                p[k] = v

        # Exit-SMA an Entry-SMA koppeln, wenn nicht separat im Grid
        if "exit_sma_period" not in combo_dict and "sma_period" in combo_dict:
            p["exit_sma_period"] = p["sma_period"]
        # VIX Entry-Max sinnvoll 2 Punkte unter Exit-Min halten, wenn VIX-Exit variiert wird
        if combo_dict.get("vol_exit_min") is not None:
            p["vol_entry_max"] = min(p["vol_entry_max"], combo_dict["vol_exit_min"] - 2.0)

        df, _, _ = run_backtest(price, rf_series, vol_series, p)
        if len(df) < 252:
            continue

        m_all = compute_numeric_metrics(df["ret_strat"], df["eq_strat"], df["rf_annual"], df["position"])

        df_train = df.loc[df.index < split_ts]
        df_test = df.loc[df.index >= split_ts]

        row = {**combo_dict, **m_all}

        if len(df_train) > 100 and len(df_test) > 100:
            eq_tr = (1.0 + df_train["ret_strat"]).cumprod()
            eq_te = (1.0 + df_test["ret_strat"]).cumprod()
            m_tr = compute_numeric_metrics(df_train["ret_strat"], eq_tr, df_train["rf_annual"], df_train["position"])
            m_te = compute_numeric_metrics(df_test["ret_strat"], eq_te, df_test["rf_annual"], df_test["position"])
            row["Train Sharpe"] = m_tr["Sharpe"]
            row["Train Calmar"] = m_tr["Calmar"]
            row["Test Sharpe"] = m_te["Sharpe"]
            row["Test Calmar"] = m_te["Calmar"]
            row["Test CAGR (%)"] = m_te["CAGR (%)"]
            row["Test Max DD (%)"] = m_te["Max DD (%)"]

        results.append(row)

    return pd.DataFrame(results)
