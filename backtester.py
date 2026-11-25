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
    """
    df = pd.DataFrame({"price": price})
    df["ret_1x"] = df["price"].pct_change().fillna(0.0)

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
    df["rvol"] = df["ret_1x"].rolling(rvol_win).std() * np.sqrt(252) * 100.0

    # Auf Startdatum und gültige Indikatoren filtern
    df = df.dropna(subset=["sma_entry", "sma_exit", "ema_fast", "ema_slow"]).copy()
    if params.get("start_date"):
        df = df.loc[df.index >= pd.Timestamp(params["start_date"])].copy()

    # Synthetische 3x-Rendite & Cash-Rendite
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

        # Position gilt für den Folgetag i+1
        positions[i + 1] = curr_pos

    df["position"] = positions
    df["trade_flag"] = df["position"].diff().abs().fillna(0.0)

    # Strategie-Return inkl. Fees/Slippage bei jedem Wechsel
    fee = params["fee_per_trade"]
    raw_ret = np.where(df["position"] == 1, df["ret_3x"], df["ret_cash"])
    df["ret_strat"] = (1.0 + raw_ret) * (1.0 - df["trade_flag"] * fee) - 1.0

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
    days = len(returns)
    years = days / 252.0
    total_return = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()

    excess_ret = returns - (rf_annual / 252.0)
    vol_annual = returns.std() * np.sqrt(252)
    sharpe = (excess_ret.mean() * 252.0) / vol_annual if vol_annual > 0 else 0.0

    res = {
        "Total Return": f"{total_return * 100:,.1f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Vola (p.a.)": f"{vol_annual * 100:.1f}%",
    }
    if position is not None:
        res["Time in Market"] = f"{position.mean() * 100:.1f}%"
        res["Trades (Wechsel)"] = int(position.diff().abs().sum())
    else:
        res["Time in Market"] = "100.0%"
        res["Trades (Wechsel)"] = 1
    return res
