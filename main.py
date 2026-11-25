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

# Modulare Strategie-Konfiguration: Für weitere Ticker einfach ein neues Dict hinzufügen
STRATEGIES = [
    {
        "ticker": "QQQ",
        "vix_ticker": "^VIX",
        "long_asset": "3QQQ",
        "cash_asset": "XEON",
        "sma_period": 200,
        "ema_fast": 19,
        "ema_slow": 39,
        "sma_entry_mult": 1.02,   # +2,0 % Buffer
        "sma_exit_mult": 0.99,    # -1,0 % Buffer
        "vix_entry_max": 25.0,
        "vix_exit_min": 27.0,
        "ema_entry_mult": 1.002,  # +0,20 % Hysterese
        "ema_exit_mult": 0.998,   # -0,20 % Hysterese
    }
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
    vix_ticker = cfg["vix_ticker"]

    # Mind. 250 Handelstage für sauberen SMA200 + EMA-Einschwingvorgang
    close = fetch_close_series(ticker, min_rows=250)
    vix_close = fetch_close_series(vix_ticker, min_rows=5)

    # Indikatoren berechnen
    sma200_series = close.rolling(window=cfg["sma_period"]).mean()
    ema_fast_series = close.ewm(span=cfg["ema_fast"], adjust=False).mean()
    ema_slow_series = close.ewm(span=cfg["ema_slow"], adjust=False).mean()

    price = float(close.iloc[-1])
    sma200 = float(sma200_series.iloc[-1])
    ema_fast = float(ema_fast_series.iloc[-1])
    ema_slow = float(ema_slow_series.iloc[-1])
    vix = float(vix_close.iloc[-1])
    data_date = close.index[-1].strftime("%Y-%m-%d")

    if any(pd.isna(val) for val in (price, sma200, ema_fast, ema_slow, vix)):
        raise ValueError(f"NaN-Wert in den berechneten Indikatoren für {ticker}.")

    # Schwellenwerte
    sma_entry_lvl = sma200 * cfg["sma_entry_mult"]
    sma_exit_lvl = sma200 * cfg["sma_exit_mult"]
    ema_entry_lvl = ema_slow * cfg["ema_entry_mult"]
    ema_exit_lvl = ema_slow * cfg["ema_exit_mult"]

    new_pos = current_pos
    triggers = []

    if current_pos == "CASH":
        cond_sma = price > sma_entry_lvl
        cond_vix = vix < cfg["vix_entry_max"]
        cond_ema = ema_fast > ema_entry_lvl

        if cond_sma and cond_vix and cond_ema:
            new_pos = "LONG"
            triggers.append("Alle 3 Entry-Locks erfüllt")

    elif current_pos == "LONG":
        if price < sma_exit_lvl:
            triggers.append(f"{ticker} &lt; SMA200-Exit ({price:.2f} &lt; {sma_exit_lvl:.2f})")
        if vix > cfg["vix_exit_min"]:
            triggers.append(f"VIX &gt; {cfg['vix_exit_min']:.1f} ({vix:.2f})")
        if ema_fast < ema_exit_lvl:
            triggers.append(f"EMA19 &lt; EMA39-Exit ({ema_fast:.2f} &lt; {ema_exit_lvl:.2f})")

        if triggers:
            new_pos = "CASH"

    metrics = {
        "data_date": data_date,
        "price": round(price, 2),
        "sma200": round(sma200, 2),
        "sma_entry_lvl": round(sma_entry_lvl, 2),
        "sma_exit_lvl": round(sma_exit_lvl, 2),
        "vix": round(vix, 2),
        "ema19": round(ema_fast, 2),
        "ema39": round(ema_slow, 2),
        "ema_entry_lvl": round(ema_entry_lvl, 2),
        "ema_exit_lvl": round(ema_exit_lvl, 2),
    }
    return new_pos, metrics, triggers


def format_message(
    cfg: dict,
    old_pos: str,
    new_pos: str,
    m: dict,
    triggers: list[str],
    is_test: bool = False,
) -> str:
    """Erstellt die formatierte Telegram-Nachricht."""
    ticker = cfg["ticker"]

    if is_test:
        header = f"<b>[TRIPLE LOCK - TEST] {ticker}</b>"
        action = f"Aktueller Status: <b>{old_pos}</b> (kein State-Wechsel gespeichert)"
    elif new_pos == "LONG":
        header = f"🟢 <b>[TRIPLE LOCK] {ticker}: CASH → LONG</b>"
        action = f"<b>Aktion:</b> Wechsel von <code>{cfg['cash_asset']}</code> in <code>{cfg['long_asset']}</code>"
    else:
        header = f"🔴 <b>[TRIPLE LOCK] {ticker}: LONG → CASH</b>"
        action = f"<b>Aktion:</b> Wechsel von <code>{cfg['long_asset']}</code> in <code>{cfg['cash_asset']}</code>"

    trigger_text = f"\n<b>Auslöser:</b> {', '.join(triggers)}" if triggers else ""

    return (
        f"{header}\n"
        f"{action}{trigger_text}\n\n"
        f"<b>Marktdaten ({m['data_date']}):</b>\n"
        f"• {ticker}: <code>{m['price']:.2f}</code>\n"
        f"• SMA200: <code>{m['sma200']:.2f}</code> "
        f"(Entry &gt; {m['sma_entry_lvl']:.2f} | Exit &lt; {m['sma_exit_lvl']:.2f})\n"
        f"• VIX: <code>{m['vix']:.2f}</code> "
        f"(Entry &lt; {cfg['vix_entry_max']:.0f} | Exit &gt; {cfg['vix_exit_min']:.0f})\n"
        f"• EMA19 / EMA39: <code>{m['ema19']:.2f}</code> / <code>{m['ema39']:.2f}</code> "
        f"(Entry &gt; {m['ema_entry_lvl']:.2f} | Exit &lt; {m['ema_exit_lvl']:.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple Lock Trading Strategy Checker")
    parser.add_argument(
        "--test-notify",
        action="store_true",
        help="Sendet eine Test-Nachricht mit Live-Daten, ohne state.json zu verändern.",
    )
    args = parser.parse_args()

    state = load_state()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for cfg in STRATEGIES:
        ticker = cfg["ticker"]
        current_entry = state.get(ticker, {})
        old_pos = current_entry.get("position", "CASH")

        try:
            new_pos, metrics, triggers = evaluate_strategy(cfg, old_pos)
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}", file=sys.stderr)
            sys.exit(1)

        print(
            f"[{ticker}] Status: {old_pos} -> {new_pos} | "
            f"Close: {metrics['price']} | SMA200: {metrics['sma200']} | "
            f"VIX: {metrics['vix']} | EMA19/39: {metrics['ema19']}/{metrics['ema39']}"
        )

        if args.test_notify:
            msg = format_message(cfg, old_pos, new_pos, metrics, triggers, is_test=True)
            send_telegram_message(msg)
            print(f"[{ticker}] Test-Benachrichtigung erfolgreich gesendet.")
            continue

        # Stateful Push: Nur bei echtem Zustandswechsel benachrichtigen
        if new_pos != old_pos:
            msg = format_message(cfg, old_pos, new_pos, metrics, triggers, is_test=False)
            send_telegram_message(msg)
            print(f"[{ticker}] Signal-Benachrichtigung ({old_pos} -> {new_pos}) gesendet.")

        # State bei jedem regulären Lauf aktualisieren
        state[ticker] = {
            "position": new_pos,
            "last_check": now_utc,
            "indicators": metrics,
        }

    if not args.test_notify:
        save_state(state)
        print("state.json erfolgreich aktualisiert.")


if __name__ == "__main__":
    main()