import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

STATE_FILE = Path("state.json")

# Modulare Strategie-Konfiguration
STRATEGIES = [
    {
        "ticker": "QQQ",
        "long_asset": "3QQQ",
        "cash_asset": "XEON",
        "sma_period": 200,
        "exit_sma_period": 200,
        "sma_entry_mult": 1.02,   # +2,0 % Buffer
        "sma_exit_mult": 0.99,    # -1,0 % Buffer
        "ema_fast": 19,
        "ema_slow": 39,
        "ema_entry_mult": 1.002,  # +0,20 % Hysterese
        "ema_exit_mult": 0.995,   # -0,50 % Hysterese (weniger Whipsaws)
        "vol_ticker": "^VIX",
        "vol_entry_max": 25.0,
        "vol_exit_min": 27.0,
    },
    {
        "ticker": "SPY",
        "long_asset": "3USL",
        "cash_asset": "XEON",
        "sma_period": 200,
        "exit_sma_period": 200,
        "sma_entry_mult": 1.02,
        "sma_exit_mult": 0.99,
        "ema_fast": 19,
        "ema_slow": 39,
        "ema_entry_mult": 1.002,
        "ema_exit_mult": 0.995,   # -0,50 % Hysterese
        "vol_ticker": "^VIX",
        "vol_entry_max": 25.0,
        "vol_exit_min": 27.0,
    },
    {
        "ticker": "GLD",
        "long_asset": "3GOL",
        "cash_asset": "XEON",
        "sma_period": 200,
        "exit_sma_period": 200,
        "sma_entry_mult": 1.02,
        "sma_exit_mult": 0.99,
        "ema_fast": 19,
        "ema_slow": 39,
        "ema_entry_mult": 1.002,
        "ema_exit_mult": 0.995,   # -0,50 % Hysterese
        "vol_ticker": None,       # Kein Vola-Lock bei Gold (GVZ bremst Krisen-Rallyes)
    },
    {
        "ticker": "SOXX",
        "long_asset": "3SEM",
        "cash_asset": "XEON",
        "sma_period": 200,
        "exit_sma_period": 100,   # Kompromiss zwischen trägem SMA200 und nervösem SMA50
        "sma_entry_mult": 1.02,
        "sma_exit_mult": 0.99,
        "ema_fast": 19,
        "ema_slow": 39,
        "ema_entry_mult": 1.002,
        "ema_exit_mult": 0.995,   # -0,50 % Hysterese
        "vol_ticker": None,       # Kein VIX, Fokus auf eigene Sektor-Vola
        "realized_vol_window": 20,
        "realized_vol_entry_max": 45.0,
        "realized_vol_exit_min": 55.0,
    },
]


def load_state() -> dict:
    """Lädt die bestehende state.json oder erstellt ein leeres Dict."""
    if not STATE_FILE.exists():
        return {}
    with STATE_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Speichert den aktuellen Zustand sauber formatiert in state.json."""
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.write("\n")


def fetch_close_series(symbol: str, min_rows: int) -> pd.Series:
    """Lädt tägliche Schlusskurse via yfinance und prüft auf Vollständigkeit."""
    df = yf.Ticker(symbol).history(period="2y", auto_adjust=True)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Keine Daten von yfinance für {symbol} erhalten.")

    close = df["Close"].dropna()
    if len(close) < min_rows:
        raise ValueError(
            f"Zu wenige Handelstage für {symbol}: {len(close)} gefunden, mind. {min_rows} benötigt."
        )
    return close


def send_telegram_message(text: str) -> None:
    """Sendet eine Nachricht via Telegram Bot API (HTML-Format)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID ist nicht als Umgebungsvariable gesetzt."
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    response = requests.post(url, json=payload, timeout=15)
    response.raise_for_status()


def evaluate_strategy(cfg: dict, current_pos: str) -> tuple[str, dict, list[str]]:
    """Berechnet Indikatoren und prüft Entry-/Exit-Bedingungen."""
    ticker = cfg["ticker"]
    sma_period = cfg["sma_period"]
    exit_sma_period = cfg.get("exit_sma_period", sma_period)

    close = fetch_close_series(ticker, min_rows=250)

    sma_entry_series = close.rolling(window=sma_period).mean()
    sma_exit_series = close.rolling(window=exit_sma_period).mean()
    ema_fast_series = close.ewm(span=cfg["ema_fast"], adjust=False).mean()
    ema_slow_series = close.ewm(span=cfg["ema_slow"], adjust=False).mean()

    price = float(close.iloc[-1])
    sma_entry = float(sma_entry_series.iloc[-1])
    sma_exit = float(sma_exit_series.iloc[-1])
    ema_fast = float(ema_fast_series.iloc[-1])
    ema_slow = float(ema_slow_series.iloc[-1])
    data_date = close.index[-1].strftime("%Y-%m-%d")

    if any(pd.isna(val) for val in (price, sma_entry, sma_exit, ema_fast, ema_slow)):
        raise ValueError(f"NaN-Wert in den berechneten Indikatoren für {ticker}.")

    # Optional: Externer Volatilitäts-Index (^VIX, ^GVZ)
    vol_ticker = cfg.get("vol_ticker")
    vol_index = None
    if vol_ticker:
        vol_close = fetch_close_series(vol_ticker, min_rows=5)
        vol_index = float(vol_close.iloc[-1])
        if pd.isna(vol_index):
            raise ValueError(f"NaN-Wert für Volatilitäts-Index {vol_ticker}.")

    # Optional: Realisierte Volatilität (annualisiert in %)
    rvol_window = cfg.get("realized_vol_window")
    realized_vol = None
    if rvol_window:
        returns = close.pct_change().dropna()
        realized_vol = float(returns.tail(rvol_window).std() * (252 ** 0.5) * 100)
        if pd.isna(realized_vol):
            raise ValueError(f"NaN-Wert für realisierte Volatilität ({ticker}).")

    sma_entry_lvl = sma_entry * cfg["sma_entry_mult"]
    sma_exit_lvl = sma_exit * cfg["sma_exit_mult"]
    ema_entry_lvl = ema_slow * cfg["ema_entry_mult"]
    ema_exit_lvl = ema_slow * cfg["ema_exit_mult"]

    new_pos = current_pos
    triggers = []

    if current_pos == "CASH":
        cond_sma = price > sma_entry_lvl
        cond_ema = ema_fast > ema_entry_lvl
        cond_vol = True if vol_index is None else (vol_index < cfg["vol_entry_max"])
        cond_rvol = (
            True
            if realized_vol is None
            else (realized_vol < cfg["realized_vol_entry_max"])
        )

        if cond_sma and cond_ema and cond_vol and cond_rvol:
            new_pos = "LONG"
            triggers.append("Alle Entry-Locks erfüllt")

    elif current_pos == "LONG":
        if price < sma_exit_lvl:
            triggers.append(
                f"{ticker} &lt; SMA{exit_sma_period}-Exit ({price:.2f} &lt; {sma_exit_lvl:.2f})"
            )
        if ema_fast < ema_exit_lvl:
            triggers.append(
                f"EMA{cfg['ema_fast']} &lt; EMA{cfg['ema_slow']}-Exit ({ema_fast:.2f} &lt; {ema_exit_lvl:.2f})"
            )
        if vol_index is not None and vol_index > cfg["vol_exit_min"]:
            vol_name = vol_ticker.lstrip("^")
            triggers.append(f"{vol_name} &gt; {cfg['vol_exit_min']:.1f} ({vol_index:.2f})")
        if realized_vol is not None and realized_vol > cfg["realized_vol_exit_min"]:
            triggers.append(
                f"Vol{rvol_window}d &gt; {cfg['realized_vol_exit_min']:.0f}% ({realized_vol:.1f}%)"
            )

        if triggers:
            new_pos = "CASH"

    # Abstand zum relevanten SMA für die kompakte Übersichtstabelle
    ref_sma = sma_exit if new_pos == "LONG" else sma_entry
    sma_dist_pct = ((price / ref_sma) - 1.0) * 100.0

    metrics = {
        "data_date": data_date,
        "price": round(price, 2),
        f"sma{sma_period}": round(sma_entry, 2),
        "sma_entry_lvl": round(sma_entry_lvl, 2),
        "sma_exit_lvl": round(sma_exit_lvl, 2),
        "sma_dist_pct": round(sma_dist_pct, 1),
        "ema19": round(ema_fast, 2),
        "ema39": round(ema_slow, 2),
        "ema_entry_lvl": round(ema_entry_lvl, 2),
        "ema_exit_lvl": round(ema_exit_lvl, 2),
    }
    if exit_sma_period != sma_period:
        metrics[f"sma{exit_sma_period}"] = round(sma_exit, 2)
    if vol_index is not None:
        metrics[vol_ticker.lstrip("^").lower()] = round(vol_index, 2)
    if realized_vol is not None:
        metrics[f"vol{rvol_window}d_pct"] = round(realized_vol, 1)

    return new_pos, metrics, triggers


def format_summary_message(
    results: list[dict], is_test: bool = False, is_weekly: bool = False
) -> str:
    """Erstellt EINE übersichtliche Sammel-Nachricht mit Aktionen + Status-Tabelle."""
    data_date = results[0]["metrics"]["data_date"] if results else "-"
    changes = [r for r in results if r["changed"]]
    readonly = is_test or is_weekly

    if is_test:
        header = f"🧪 <b>[TRIPLE LOCK - TEST] ({data_date})</b>"
    elif is_weekly:
        header = f"📊 <b>[TRIPLE LOCK] WOCHEN-STATUS ({data_date})</b>"
    else:
        header = f"🚨 <b>[TRIPLE LOCK] SIGNAL-UPDATE ({data_date})</b>"

    lines = [header, ""]

    # 1. Aktionsblock (bei echten Wechseln oder Hinweis im Test-/Wochenmodus)
    if changes and not readonly:
        lines.append("<b>AKTION ERFORDERLICH:</b>")
        for r in changes:
            cfg = r["cfg"]
            ticker = cfg["ticker"]
            reason = ", ".join(r["triggers"])
            if r["new_pos"] == "LONG":
                lines.append(
                    f"🟢 <b>KAUFEN: {ticker}</b> → Wechsel <code>{cfg['cash_asset']}</code> in <code>{cfg['long_asset']}</code>"
                )
            else:
                lines.append(
                    f"🔴 <b>VERKAUFEN: {ticker}</b> → Wechsel <code>{cfg['long_asset']}</code> in <code>{cfg['cash_asset']}</code>"
                )
            lines.append(f"   <i>Grund: {reason}</i>")
        lines.append("")
    elif is_test:
        lines.append("<i>Testlauf: Kein State-Wechsel gespeichert.</i>\n")
    elif is_weekly:
        lines.append("<i>Wöchentlicher Status-Report (kein Signalwechsel).</i>\n")

    # 2. Kompakte Übersichtstabelle für alle Assets
    lines.append("<b>PORTFOLIO-ÜBERSICHT:</b>")
    table_rows = ["   Asset Ziel  Kurs   vsSMA   Vola"]

    for r in results:
        cfg = r["cfg"]
        m = r["metrics"]
        pos = r["old_pos"] if readonly else r["new_pos"]
        icon = "🟢" if pos == "LONG" else "🔴"
        target = cfg["long_asset"] if pos == "LONG" else cfg["cash_asset"]

        # Kompakte Vola-Spalte: bei SOXX die 20d-Vola, sonst VIX/GVZ
        if cfg.get("realized_vol_window"):
            rvol_key = f"vol{cfg['realized_vol_window']}d_pct"
            vol_str = f"{m[rvol_key]:.1f}%"
        elif cfg.get("vol_ticker"):
            v_key = cfg["vol_ticker"].lstrip("^").lower()
            vol_str = f"{m[v_key]:.1f}"
        else:
            vol_str = "-"

        row = (
            f"{icon} {cfg['ticker']:<4}  {target:<4} "
            f"{m['price']:>6.1f} {m['sma_dist_pct']:>+6.1f}% {vol_str:>5}"
        )
        table_rows.append(row)

    lines.append("<pre>" + "\n".join(table_rows) + "</pre>")
    lines.append(
        "<i>🟢 = Investiert (3x) | 🔴 = Cash (XEON)\n"
        "vsSMA = Abstand zum aktiven SMA (SOXX Exit: SMA100)</i>"
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple Lock Trading Strategy Checker")
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="Sendet eine Test-Sammelnachricht mit Live-Daten, ohne state.json zu verändern.",
    )
    parser.add_argument(
        "--weekly-summary",
        action="store_true",
        help="Sendet den wöchentlichen Status-Report, ohne state.json zu verändern.",
    )
    args = parser.parse_args()
    readonly_mode = args.test_notify or args.weekly_summary

    state = load_state()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []
    had_error = False

    for cfg in STRATEGIES:
        ticker = cfg["ticker"]
        current_entry = state.get(ticker, {})
        old_pos = current_entry.get("position", "CASH")

        try:
            new_pos, metrics, triggers = evaluate_strategy(cfg, old_pos)
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}", file=sys.stderr)
            had_error = True
            continue

        changed = new_pos != old_pos
        print(f"[{ticker}] Status: {old_pos} -> {new_pos} | Metrics: {metrics}")

        results.append(
            {
                "cfg": cfg,
                "old_pos": old_pos,
                "new_pos": new_pos,
                "metrics": metrics,
                "triggers": triggers,
                "changed": changed,
            }
        )

        if not readonly_mode:
            state[ticker] = {
                "position": new_pos,
                "last_check": now_utc,
                "indicators": metrics,
            }

    # Eine konsolidierte Nachricht senden bei Wechsel, Test oder Wochen-Report
    if results and (readonly_mode or any(r["changed"] for r in results)):
        msg = format_summary_message(
            results, is_test=args.test_notify, is_weekly=args.weekly_summary
        )
        send_telegram_message(msg)
        print("Nachricht erfolgreich an Telegram gesendet.")

    if not readonly_mode and results:
        save_state(state)
        print("state.json erfolgreich aktualisiert.")

    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
