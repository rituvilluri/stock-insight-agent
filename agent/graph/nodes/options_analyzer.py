"""
Node 8: Options Analyzer

Reads:  ticker, price_data (for current stock price), next_earnings_date,
        days_until_earnings
Writes: options_data, options_error

Fetches the options chain from yfinance and produces a structured summary:
  - Put/call ratio (total put volume / total call volume)
  - Top-volume call and put strikes
  - Max Pain strike (where total option buyer loss is maximized at expiry)
  - Black-Scholes Greeks for the at-the-money option (nearest expiry)
  - Average implied volatility across the chain

Greeks are calculated via Black-Scholes using the implied volatility already
embedded in the yfinance chain — no paid data source required. The risk-free
rate defaults to 5% (approximate 10-year treasury rate).

No LLM calls in this node — pure data retrieval and arithmetic.
"""

import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from agent.graph.nodes.state import AgentState

logger = logging.getLogger(__name__)

# How many top-volume strikes to include in the summary
TOP_N_STRIKES = 5

# Risk-free rate used in Black-Scholes (approximate 10-year treasury)
RISK_FREE_RATE = 0.05


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF computed via erfc — no scipy dependency."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str = "call",
    r: float = RISK_FREE_RATE,
) -> dict:
    """
    Calculate Delta, Gamma, Theta, and Vega for a European option.

    Args:
        S:           Current stock price
        K:           Strike price
        T:           Time to expiration in years
        sigma:       Implied volatility (annualized, e.g. 0.30 for 30%)
        option_type: "call" or "put"
        r:           Risk-free rate (annualized, default RISK_FREE_RATE)

    Returns dict with keys delta, gamma, theta, vega (all floats).
    Returns None values if inputs are invalid (T=0, sigma=0, etc.).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or math.isnan(sigma) or math.isnan(S):
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    sqrt_T = math.sqrt(T)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = _normal_cdf(d1)
    nd2 = _normal_cdf(d2)
    # Standard normal PDF at d1
    npd1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)

    if option_type == "call":
        delta = nd1
        theta_rhs = r * K * math.exp(-r * T) * nd2
    else:
        delta = nd1 - 1.0
        theta_rhs = r * K * math.exp(-r * T) * (1 - nd2)

    gamma = npd1 / (S * sigma * sqrt_T)
    # Theta: daily decay (divide annualized figure by 365)
    theta = ((-S * npd1 * sigma) / (2 * sqrt_T) - theta_rhs) / 365
    # Vega: per 1% change in IV (divide by 100)
    vega = S * npd1 * sqrt_T / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


# ---------------------------------------------------------------------------
# Max Pain helper
# ---------------------------------------------------------------------------

def _calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> Optional[float]:
    """
    Calculate the Max Pain strike — the expiry price at which total option
    buyer gains are minimized (equivalently, where sellers retain the most
    premium).

    For each candidate strike P:
      - Sum (P - K) * OI for all calls where P > K  (in-the-money calls)
      - Sum (K - P) * OI for all puts  where P < K  (in-the-money puts)
    The strike that minimizes the combined total is Max Pain.

    Returns the Max Pain strike as a float, or None if data is insufficient.
    """
    try:
        all_strikes = sorted(
            set(list(calls_df["strike"].dropna())) |
            set(list(puts_df["strike"].dropna()))
        )
        if not all_strikes:
            return None

        call_strikes = calls_df["strike"].tolist()
        call_oi = calls_df["openInterest"].tolist()
        put_strikes = puts_df["strike"].tolist()
        put_oi = puts_df["openInterest"].tolist()

        min_total = float("inf")
        max_pain_strike = None

        for P in all_strikes:
            call_buyer_gain = sum(
                max(0.0, P - K) * oi
                for K, oi in zip(call_strikes, call_oi)
                if oi and not math.isnan(oi)
            )
            put_buyer_gain = sum(
                max(0.0, K - P) * oi
                for K, oi in zip(put_strikes, put_oi)
                if oi and not math.isnan(oi)
            )
            total = call_buyer_gain + put_buyer_gain
            if total < min_total:
                min_total = total
                max_pain_strike = P

        return float(max_pain_strike) if max_pain_strike is not None else None

    except Exception as e:
        logger.warning("_calculate_max_pain failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def analyze_options(state: AgentState) -> AgentState:
    """
    Fetch the options chain for the resolved ticker and compute a trader-grade
    options summary: put/call ratio, top strikes, Max Pain, and ATM Greeks.
    """
    ticker = state.get("ticker", "")
    if not ticker:
        msg = "analyze_options: ticker missing from state"
        logger.error(msg)
        return {**state, "options_data": None, "options_error": msg}

    try:
        stock = yf.Ticker(ticker)
        expiry_dates = stock.options

        if not expiry_dates:
            msg = f"No options expiry dates available for {ticker}"
            logger.warning(msg)
            return {**state, "options_data": None, "options_error": msg}

        # Use nearest expiry for Greeks and Max Pain; aggregate all for totals
        target_expiry = expiry_dates[0]
        chain = stock.option_chain(target_expiry)
        calls = chain.calls
        puts = chain.puts

        # ------------------------------------------------------------------
        # Volume and put/call ratio
        # ------------------------------------------------------------------
        total_call_volume = int(calls["volume"].fillna(0).sum())
        total_put_volume = int(puts["volume"].fillna(0).sum())
        put_call_ratio = (
            round(total_put_volume / total_call_volume, 3)
            if total_call_volume > 0
            else None
        )

        # ------------------------------------------------------------------
        # Top strikes by volume
        # ------------------------------------------------------------------
        vol_cols = ["strike", "volume", "openInterest", "impliedVolatility"]
        top_calls = (
            calls.nlargest(TOP_N_STRIKES, "volume")[vol_cols]
            .fillna(0)
            .to_dict("records")
        )
        top_puts = (
            puts.nlargest(TOP_N_STRIKES, "volume")[vol_cols]
            .fillna(0)
            .to_dict("records")
        )

        # ------------------------------------------------------------------
        # Average implied volatility
        # ------------------------------------------------------------------
        all_iv = pd.concat([
            calls["impliedVolatility"],
            puts["impliedVolatility"],
        ]).dropna()
        avg_iv = round(float(all_iv.mean()), 4) if not all_iv.empty else None

        # ------------------------------------------------------------------
        # Max Pain
        # ------------------------------------------------------------------
        max_pain = _calculate_max_pain(calls, puts)

        # ------------------------------------------------------------------
        # ATM Greeks (nearest strike to current price)
        # ------------------------------------------------------------------
        greeks_sample = None
        price_data = state.get("price_data") or {}
        S = price_data.get("close_price") if isinstance(price_data, dict) else None

        if S is not None and not calls.empty:
            atm_row = calls.iloc[(calls["strike"] - S).abs().argsort().iloc[:1]]
            if not atm_row.empty:
                K = float(atm_row.iloc[0]["strike"])
                sigma = float(atm_row.iloc[0]["impliedVolatility"])

                try:
                    expiry_dt = datetime.strptime(target_expiry, "%Y-%m-%d")
                    T = max((expiry_dt - datetime.now()).days / 365.0, 0.001)
                except Exception:
                    T = 0.0

                if sigma > 0 and T > 0:
                    greeks_sample = {
                        "strike": K,
                        "expiry": target_expiry,
                        "call": _black_scholes_greeks(S, K, T, sigma, "call"),
                        "put": _black_scholes_greeks(S, K, T, sigma, "put"),
                    }

        # ------------------------------------------------------------------
        # Assemble options_data
        # ------------------------------------------------------------------
        options_data = {
            "expiration_dates": list(expiry_dates[:5]),
            "put_call_ratio": put_call_ratio,
            "highest_volume_calls": top_calls,
            "highest_volume_puts": top_puts,
            "total_call_volume": total_call_volume,
            "total_put_volume": total_put_volume,
            "average_implied_volatility": avg_iv,
            "max_pain": max_pain,
            "greeks_sample": greeks_sample,
            "notable_positions": [],
        }

        logger.info(
            "analyze_options [%s] → PCR=%.2f max_pain=%s expiry=%s",
            ticker,
            put_call_ratio or 0,
            max_pain,
            target_expiry,
        )
        return {**state, "options_data": options_data, "options_error": None}

    except Exception as e:
        logger.error("analyze_options failed for %s: %s", ticker, e)
        return {**state, "options_data": None, "options_error": str(e)}
