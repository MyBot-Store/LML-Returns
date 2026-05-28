"""
performance_sync.py
--------------------
Pulls trade data from IBKR Flex Query API, calculates equity curve
and key performance metrics, then saves to data/performance_data.json
for the Chart.js dashboard to read.

Required environment variables (set as GitHub Secrets):
  IBKR_FLEX_TOKEN   - Your Flex Web Service token from IBKR Client Portal
  IBKR_QUERY_ID     - Your Activity Flex Query ID from IBKR Client Portal
  STARTING_BALANCE  - Your account starting balance (e.g. "10000.00")
"""

import os
import json
import time
import math
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, date
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

FLEX_TOKEN      = os.environ["IBKR_FLEX_TOKEN"]
FLEX_QUERY_ID   = os.environ["IBKR_QUERY_ID"]
STARTING_BALANCE = float(os.environ.get("STARTING_BALANCE", "10000.00"))

SEND_URL  = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
FETCH_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"

OUTPUT_FILE = "data/performance_data.json"


# ── Step 1: Pull data from IBKR Flex Query ────────────────────────────────────

def request_flex_report():
    """Step 1 of 2: Request IBKR to generate the report. Returns a reference code."""
    print("Requesting Flex report from IBKR...")
    resp = requests.get(SEND_URL, params={"t": FLEX_TOKEN, "q": FLEX_QUERY_ID, "v": "3"}, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    status = root.findtext("Status")
    if status != "Success":
        raise RuntimeError(f"IBKR Flex SendRequest failed: {root.findtext('ErrorMessage')}")

    ref_code = root.findtext("ReferenceCode")
    print(f"Report requested. Reference code: {ref_code}")
    return ref_code


def fetch_flex_report(ref_code):
    """Step 2 of 2: Fetch the generated report using the reference code."""
    print("Fetching Flex report...")
    # IBKR recommends waiting a few seconds before fetching
    time.sleep(5)

    for attempt in range(5):
        resp = requests.get(FETCH_URL, params={"t": FLEX_TOKEN, "q": ref_code, "v": "3"}, timeout=30)
        resp.raise_for_status()

        # If report not ready yet, IBKR returns a status XML
        if "<FlexQueryResponse" in resp.text or "<FlexStatements" in resp.text:
            print("Report received.")
            return resp.text

        # Check if it's a "not ready" response
        try:
            root = ET.fromstring(resp.text)
            if root.findtext("Status") == "Warn":
                print(f"  Report not ready (attempt {attempt+1}/5), waiting 5s...")
                time.sleep(5)
                continue
        except ET.ParseError:
            pass

        raise RuntimeError(f"Unexpected IBKR response: {resp.text[:300]}")

    raise RuntimeError("IBKR report not ready after 5 attempts.")


# ── Step 2: Parse XML and extract trades ──────────────────────────────────────

def parse_trades(xml_text):
    """
    Parses IBKR Flex Query XML and returns a list of closed trade dicts.
    Only closed trades (openCloseIndicator == 'C') have realized P&L.
    """
    root = ET.fromstring(xml_text)
    trades = []

    for trade in root.iter("Trade"):
        indicator = trade.get("openCloseIndicator", "")
        # Only process closed trades — these have realized P&L
        if "C" not in indicator:
            continue

        try:
            trade_date_str = trade.get("tradeDate", "")
            if not trade_date_str:
                continue

            # IBKR date format is YYYYMMDD
            trade_date = datetime.strptime(trade_date_str, "%Y%m%d").date()

            realized_pnl  = float(trade.get("fifoPnlRealized", 0) or 0)
            commission     = float(trade.get("ibCommission", 0) or 0)
            net_pnl        = realized_pnl + commission  # commission is already negative in IBKR data

            buy_sell       = trade.get("buySell", "")
            symbol         = trade.get("symbol", "")

            trades.append({
                "date":        trade_date,
                "symbol":      symbol,
                "buy_sell":    buy_sell,
                "realized_pnl": realized_pnl,
                "commission":  commission,
                "net_pnl":     net_pnl,
            })

        except (ValueError, TypeError) as e:
            print(f"  Skipping trade row due to parse error: {e}")
            continue

    print(f"Parsed {len(trades)} closed trades.")
    return trades


# ── Step 3: Calculate equity curve ───────────────────────────────────────────

def calculate_equity_curve(trades, starting_balance):
    """
    Builds a day-by-day equity curve from closed trade P&L.
    Returns a sorted list of {date, balance, daily_pnl} dicts.
    """
    # Group net P&L by date
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl"]

    if not daily_pnl:
        print("Warning: No trade data found. Returning empty equity curve.")
        return []

    # Build equity curve in chronological order
    sorted_dates = sorted(daily_pnl.keys())
    curve = []
    balance = starting_balance

    for d in sorted_dates:
        pnl = daily_pnl[d]
        balance += pnl
        curve.append({
            "date":      d.strftime("%Y-%m-%d"),
            "balance":   round(balance, 2),
            "daily_pnl": round(pnl, 2),
        })

    return curve


# ── Step 4: Calculate performance metrics ─────────────────────────────────────

def calculate_metrics(trades, equity_curve, starting_balance):
    """
    Calculates key statistics investors care about:
    Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor.
    """
    if not equity_curve or not trades:
        return {}

    ending_balance = equity_curve[-1]["balance"]
    total_return_pct = ((ending_balance - starting_balance) / starting_balance) * 100

    # Daily returns (as decimals) for Sharpe calculation
    balances = [starting_balance] + [row["balance"] for row in equity_curve]
    daily_returns = [
        (balances[i] - balances[i-1]) / balances[i-1]
        for i in range(1, len(balances))
        if balances[i-1] != 0
    ]

    # Sharpe Ratio (annualised, assumes 252 trading days, risk-free rate ~0)
    if len(daily_returns) >= 2:
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        sharpe = (mean_return / std_dev) * math.sqrt(252) if std_dev > 0 else 0
    else:
        sharpe = 0

    # Maximum Drawdown
    peak = starting_balance
    max_dd = 0
    for row in equity_curve:
        b = row["balance"]
        if b > peak:
            peak = b
        dd = (peak - b) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Win Rate & Profit Factor (trade-level)
    closed_trades = [t for t in trades if t["net_pnl"] != 0]
    winners = [t for t in closed_trades if t["net_pnl"] > 0]
    losers  = [t for t in closed_trades if t["net_pnl"] < 0]

    win_rate = (len(winners) / len(closed_trades) * 100) if closed_trades else 0

    gross_profit = sum(t["net_pnl"] for t in winners)
    gross_loss   = abs(sum(t["net_pnl"] for t in losers))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    return {
        "starting_balance":  round(starting_balance, 2),
        "ending_balance":    round(ending_balance, 2),
        "total_return_pct":  round(total_return_pct, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "win_rate_pct":      round(win_rate, 1),
        "profit_factor":     round(profit_factor, 2),
        "total_trades":      len(closed_trades),
        "winning_trades":    len(winners),
        "losing_trades":     len(losers),
    }


# ── Step 5: Save output JSON ───────────────────────────────────────────────────

def save_output(equity_curve, metrics):
    """Saves equity curve + metrics to the JSON file read by Chart.js."""
    os.makedirs("data", exist_ok=True)

    output = {
        "last_updated":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "metrics":       metrics,
        "equity_curve":  equity_curve,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(equity_curve)} data points to {OUTPUT_FILE}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("IBKR Performance Sync")
    print(f"Starting balance: ${STARTING_BALANCE:,.2f}")
    print("=" * 50)

    try:
        # 1. Pull data from IBKR
        ref_code = request_flex_report()
        xml_data = fetch_flex_report(ref_code)

        # 2. Parse trades
        trades = parse_trades(xml_data)

        if not trades:
            print("No closed trades found. Check your Flex Query date range.")
            # Save empty output so dashboard shows gracefully
            save_output([], {})
        else:
            # 3. Build equity curve
            equity_curve = calculate_equity_curve(trades, STARTING_BALANCE)

            # 4. Calculate metrics
            metrics = calculate_metrics(trades, equity_curve, STARTING_BALANCE)

            # 5. Save output
            save_output(equity_curve, metrics)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        raise
