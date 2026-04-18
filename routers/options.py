"""
routers/options.py — MONARCH PRO Options Intelligence Engine
FastAPI router ported from the Streamlit option.py.
All computation lives here; the HTML page fetches via fetch().
"""

import os, json, math, time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

# ── shared STATE injected by main.py ─────────────────────────────────────────
_STATE: dict = {}

def init_state(state_dict: dict):
    global _STATE
    _STATE = state_dict

router = APIRouter(prefix="/api/options", tags=["options"])

# ═══════════════════════════════════════════════════════════════════════════════
# CFG
# ═══════════════════════════════════════════════════════════════════════════════
LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 75, "SENSEX": 10,
    "RELIANCE": 250, "HDFCBANK": 550, "ICICIBANK": 700, "INFY": 400,
    "TCS": 150, "LT": 150, "SBIN": 1500, "AXISBANK": 625,
    "KOTAKBANK": 400, "BHARTIARTL": 500, "ITC": 3200,
    "BAJFINANCE": 125, "WIPRO": 1500, "HCLTECH": 350,
    "TATAMOTORS": 1425, "MARUTI": 100, "SUNPHARMA": 350,
    "TITAN": 175, "ADANIENT": 400, "ONGC": 1925,
    "NTPC": 2250, "JSWSTEEL": 600, "TATASTEEL": 5500,
    "HINDALCO": 1075, "DRREDDY": 125, "CIPLA": 650, "DIVISLAB": 200,
}

_CHAIN_CACHE: dict = {}   # {symbol: {data, ts}}
_EXPIRY_CACHE: dict = {}  # {symbol: {expiries, ts}}
_MASTER_INSTRUMENTS: dict = {}
_MASTER_TS: float = 0.0

RFR_DEFAULT = 6.5  # India repo rate %

# ═══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES ENGINE (no external lib)
# ═══════════════════════════════════════════════════════════════════════════════
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_price(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(0, S - K) if opt == "call" else max(0, K - S)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_greeks(S, K, T, r, sigma, opt="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0, "iv": sigma}
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = _norm_pdf(d1)
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega  = S * nd1 * math.sqrt(T) / 100
    if opt == "call":
        delta = _norm_cdf(d1)
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 252
        rho   = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
    else:
        delta = _norm_cdf(d1) - 1
        theta = (-(S * nd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 252
        rho   = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100
    return {"delta": round(delta, 4), "gamma": round(gamma, 6),
            "theta": round(theta, 4), "vega": round(vega, 4),
            "rho": round(rho, 4), "iv": round(sigma * 100, 2)}

def implied_vol(market_price, S, K, T, r, opt="call", tol=1e-5, max_iter=100):
    if T <= 0 or market_price <= 0:
        return None
    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p   = bs_price(S, K, T, r, mid, opt)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ═══════════════════════════════════════════════════════════════════════════════
# UPSTOX HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _headers():
    tok = _STATE.get("token", "")
    if not tok:
        raise HTTPException(401, "Not authenticated — login first")
    return {"Authorization": f"Bearer {tok}", "Accept": "application/json"}

def _get_instrument_master():
    global _MASTER_INSTRUMENTS, _MASTER_TS
    if time.time() - _MASTER_TS < 3600 and _MASTER_INSTRUMENTS:
        return _MASTER_INSTRUMENTS
    try:
        import gzip, io
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r = requests.get(url, timeout=20)
        data = json.loads(gzip.decompress(r.content))
        _MASTER_INSTRUMENTS = {d["trading_symbol"]: d for d in data if isinstance(d, dict)}
        _MASTER_TS = time.time()
    except Exception as e:
        raise HTTPException(500, f"Instrument master fetch failed: {e}")
    return _MASTER_INSTRUMENTS

def _get_expiries(symbol: str) -> list:
    cached = _EXPIRY_CACHE.get(symbol)
    if cached and time.time() - cached["ts"] < 300:
        return cached["expiries"]
    master = _get_instrument_master()
    sym_up = symbol.upper()
    expiries = set()
    for k, v in master.items():
        if (v.get("underlying_symbol") == sym_up and
                v.get("instrument_type") in ("CE", "PE")):
            exp = v.get("expiry")
            if exp:
                expiries.add(exp)
    expiries = sorted(expiries)
    _EXPIRY_CACHE[symbol] = {"expiries": expiries, "ts": time.time()}
    return expiries

def _fetch_chain(symbol: str, expiry: str) -> list:
    cache_key = f"{symbol}:{expiry}"
    cached = _CHAIN_CACHE.get(cache_key)
    if cached and time.time() - cached["ts"] < 30:
        return cached["data"]
    hdrs = _headers()
    url = "https://api.upstox.com/v2/option/chain"
    params = {"instrument_key": f"NSE_INDEX|{symbol}", "expiry_date": expiry}
    r = requests.get(url, headers=hdrs, params=params, timeout=15)
    data = r.json()
    if data.get("status") != "success":
        # Try EQ for stock options
        params["instrument_key"] = f"NSE_EQ|{symbol}"
        r = requests.get(url, headers=hdrs, params=params, timeout=15)
        data = r.json()
    if data.get("status") != "success":
        raise HTTPException(500, f"Chain fetch failed: {data.get('message', data)}")
    chain = data.get("data", [])
    _CHAIN_CACHE[cache_key] = {"data": chain, "ts": time.time()}
    return chain

def _fetch_ltp(symbol: str) -> float | None:
    try:
        hdrs = _headers()
        # Try index first
        ikey = requests.utils.quote(f"NSE_INDEX|{symbol}", safe="")
        r = requests.get(f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={ikey}",
                         headers=hdrs, timeout=8)
        d = r.json()
        if d.get("status") == "success":
            vals = list(d.get("data", {}).values())
            if vals:
                return float(vals[0].get("last_price", 0))
        # Try EQ
        ikey = requests.utils.quote(f"NSE_EQ|{symbol}", safe="")
        r = requests.get(f"https://api.upstox.com/v2/market-quote/ltp?instrument_key={ikey}",
                         headers=hdrs, timeout=8)
        d = r.json()
        if d.get("status") == "success":
            vals = list(d.get("data", {}).values())
            if vals:
                return float(vals[0].get("last_price", 0))
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def _historical_vol(symbol: str, window: int = 20) -> float:
    try:
        import yfinance as yf
        yf_sym = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(yf_sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < window + 2:
            return 0.0
        rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        return float(rets.tail(window).std() * math.sqrt(252))
    except Exception:
        return 0.0

def _compute_max_pain(chain: list, spot: float) -> float:
    """Max-pain: strike at which total option seller loss is minimised."""
    strikes = sorted({r.get("strike_price", 0) for r in chain if r.get("strike_price")})
    if not strikes:
        return spot
    best_strike, min_pain = strikes[0], float("inf")
    for K in strikes:
        pain = 0.0
        for r in chain:
            sk = r.get("strike_price", 0)
            ce = (r.get("call_options") or {}).get("market_data", {}).get("oi", 0) or 0
            pe = (r.get("put_options")  or {}).get("market_data", {}).get("oi", 0) or 0
            pain += max(0, K - sk) * pe + max(0, sk - K) * ce
        if pain < min_pain:
            min_pain = pain
            best_strike = K
    return best_strike

def _process_chain(chain: list, spot: float, expiry: str, rfr: float) -> dict:
    """Full chain processing: greeks, IV, OI analysis, max pain, PCR."""
    today = date.today()
    try:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
    except Exception:
        exp_dt = today
    T = max((exp_dt - today).days, 0) / 365.0
    r = rfr / 100

    rows = []
    total_ce_oi = total_pe_oi = total_ce_vol = total_pe_vol = 0.0

    for item in chain:
        K = item.get("strike_price", 0)
        if not K:
            continue
        ce_md = (item.get("call_options") or {}).get("market_data", {})
        pe_md = (item.get("put_options")  or {}).get("market_data", {})

        ce_ltp = float(ce_md.get("ltp") or 0)
        pe_ltp = float(pe_md.get("ltp") or 0)
        ce_oi  = float(ce_md.get("oi") or 0)
        pe_oi  = float(pe_md.get("oi") or 0)
        ce_vol = float(ce_md.get("volume") or 0)
        pe_vol = float(pe_md.get("volume") or 0)
        ce_bid = float(ce_md.get("bid_price") or ce_ltp * 0.98)
        ce_ask = float(ce_md.get("ask_price") or ce_ltp * 1.02)
        pe_bid = float(pe_md.get("bid_price") or pe_ltp * 0.98)
        pe_ask = float(pe_md.get("ask_price") or pe_ltp * 1.02)

        total_ce_oi  += ce_oi
        total_pe_oi  += pe_oi
        total_ce_vol += ce_vol
        total_pe_vol += pe_vol

        # IV
        ce_iv = implied_vol(ce_ltp, spot, K, T, r, "call") if ce_ltp > 0 else None
        pe_iv = implied_vol(pe_ltp, spot, K, T, r, "put")  if pe_ltp > 0 else None
        atm_iv = ce_iv or pe_iv or 0.15

        # Greeks
        ce_g = bs_greeks(spot, K, T, r, atm_iv, "call") if atm_iv else {}
        pe_g = bs_greeks(spot, K, T, r, atm_iv, "put")  if atm_iv else {}

        moneyness = (spot - K) / spot * 100
        itm_ce = K < spot
        itm_pe = K > spot

        rows.append({
            "strike":   K,
            "moneyness": round(moneyness, 2),
            "ce_ltp":   ce_ltp, "pe_ltp": pe_ltp,
            "ce_oi":    ce_oi,  "pe_oi":  pe_oi,
            "ce_vol":   ce_vol, "pe_vol": pe_vol,
            "ce_iv":    round((ce_iv or 0) * 100, 2),
            "pe_iv":    round((pe_iv or 0) * 100, 2),
            "ce_delta": ce_g.get("delta", 0), "pe_delta": pe_g.get("delta", 0),
            "ce_gamma": ce_g.get("gamma", 0), "pe_gamma": pe_g.get("gamma", 0),
            "ce_theta": ce_g.get("theta", 0), "pe_theta": pe_g.get("theta", 0),
            "ce_vega":  ce_g.get("vega", 0),  "pe_vega":  pe_g.get("vega", 0),
            "ce_bid":   round(ce_bid, 2),      "ce_ask":   round(ce_ask, 2),
            "pe_bid":   round(pe_bid, 2),      "pe_ask":   round(pe_ask, 2),
            "ce_spread_pct": round((ce_ask - ce_bid) / max(ce_ltp, 0.01) * 100, 2) if ce_ltp else 0,
            "pe_spread_pct": round((pe_ask - pe_bid) / max(pe_ltp, 0.01) * 100, 2) if pe_ltp else 0,
            "itm_ce": itm_ce, "itm_pe": itm_pe,
        })

    if not rows:
        return {"error": "No chain data"}

    df = pd.DataFrame(rows).sort_values("strike")
    pcr_oi  = total_pe_oi  / max(total_ce_oi, 1)
    pcr_vol = total_pe_vol / max(total_ce_vol, 1)
    max_pain = _compute_max_pain(chain, spot)

    # ATM IV (closest to spot)
    df["dist"] = (df["strike"] - spot).abs()
    atm_row = df.loc[df["dist"].idxmin()]
    atm_iv_pct = float((atm_row["ce_iv"] + atm_row["pe_iv"]) / 2) if atm_row["ce_iv"] and atm_row["pe_iv"] else float(atm_row["ce_iv"] or atm_row["pe_iv"])

    # Implied move
    impl_move_pct = atm_iv_pct / 100 * math.sqrt(T) * 100 if T > 0 else 0

    # OI skew: (total CE OI - total PE OI) / total
    oi_total = total_ce_oi + total_pe_oi
    oi_skew  = (total_ce_oi - total_pe_oi) / max(oi_total, 1)

    # Max-pain distance
    mp_dist_pct = (max_pain - spot) / spot * 100

    return {
        "chain":         df.to_dict(orient="records"),
        "spot":          round(spot, 2),
        "expiry":        expiry,
        "dte":           int(max((exp_dt - today).days, 0)),
        "atm_iv_pct":    round(atm_iv_pct, 2),
        "impl_move_pct": round(impl_move_pct, 2),
        "pcr_oi":        round(pcr_oi, 3),
        "pcr_vol":       round(pcr_vol, 3),
        "max_pain":      round(max_pain, 2),
        "mp_dist_pct":   round(mp_dist_pct, 2),
        "oi_skew":       round(oi_skew, 3),
        "total_ce_oi":   int(total_ce_oi),
        "total_pe_oi":   int(total_pe_oi),
        "total_ce_vol":  int(total_ce_vol),
        "total_pe_vol":  int(total_pe_vol),
    }

def _directional_bias(symbol: str, spot: float, chain_data: dict) -> dict:
    """7-factor technical directional model."""
    try:
        import yfinance as yf
        yf_sym = symbol + ".NS" if symbol not in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY") else {
            "NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^NSEI", "MIDCPNIFTY": "^NSEI"
        }.get(symbol, symbol)
        df = yf.download(yf_sym, period="6mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < 20:
            return {"score": 0, "bias": "NEUTRAL", "prob_up": 0.5, "factors": {}}

        close = df["Close"].squeeze()
        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        # ADX
        high = df["High"].squeeze(); low = df["Low"].squeeze()
        tr   = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        up_move   = high - high.shift(); dn_move = low.shift() - low
        pos_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
        neg_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
        atr14  = tr.ewm(span=14, adjust=False).mean()
        pdi = 100 * pos_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, 1e-9)
        mdi = 100 * neg_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, 1e-9)
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
        adx = dx.ewm(span=14, adjust=False).mean()

        # 52-week position
        hi52 = close.rolling(252, min_periods=50).max()
        lo52 = close.rolling(252, min_periods=50).min()
        pos52 = (close - lo52) / (hi52 - lo52).replace(0, 1e-9)

        # Factors
        lv = close.iloc[-1]
        f_trend = 1 if (ema9.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]) else (-1 if ema9.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1] else 0)
        f_mom   = 1 if rsi.iloc[-1] > 55 else (-1 if rsi.iloc[-1] < 45 else 0)
        f_adx   = (1 if pdi.iloc[-1] > mdi.iloc[-1] else -1) * min(adx.iloc[-1] / 25, 1)
        f_52w   = 1 if pos52.iloc[-1] > 0.7 else (-1 if pos52.iloc[-1] < 0.3 else 0)
        # PCR factor
        pcr = chain_data.get("pcr_oi", 1.0)
        f_pcr = 1 if pcr > 1.2 else (-1 if pcr < 0.8 else 0)
        # Max-pain factor
        mp_dist = chain_data.get("mp_dist_pct", 0)
        f_mp = 1 if mp_dist > 1 else (-1 if mp_dist < -1 else 0)
        # Vol factor
        hv = _historical_vol(symbol, 20)
        atm_iv = chain_data.get("atm_iv_pct", 0) / 100
        f_vol = -1 if atm_iv > hv * 1.2 else (1 if atm_iv < hv * 0.85 else 0)

        factors = {
            "trend":   round(float(f_trend), 2),
            "momentum": round(float(f_mom), 2),
            "adx":     round(float(f_adx), 2),
            "pos_52w": round(float(f_52w), 2),
            "pcr":     round(float(f_pcr), 2),
            "max_pain": round(float(f_mp), 2),
            "vol_regime": round(float(f_vol), 2),
        }
        raw_score = sum(factors.values()) / (2.5 * len(factors))
        prob_up   = 1 / (1 + math.exp(-7 * raw_score))
        bias = "BULLISH" if prob_up > 0.55 else ("BEARISH" if prob_up < 0.45 else "NEUTRAL")
        return {
            "score":   round(raw_score, 3),
            "prob_up": round(prob_up, 3),
            "bias":    bias,
            "factors": factors,
            "rsi":     round(float(rsi.iloc[-1]), 1),
            "adx_val": round(float(adx.iloc[-1]), 1),
            "pos52w":  round(float(pos52.iloc[-1]), 3),
            "hv20":    round(hv * 100, 2),
            "atm_iv":  round(atm_iv * 100, 2),
            "iv_hv_ratio": round(atm_iv / hv, 3) if hv else None,
        }
    except Exception as e:
        return {"score": 0, "bias": "NEUTRAL", "prob_up": 0.5, "factors": {}, "error": str(e)}

def _strategy_recommendations(bias: dict, chain_data: dict, dte: int) -> list:
    """EV-ranked strategy recommendations."""
    prob_up  = bias.get("prob_up", 0.5)
    iv_hv    = bias.get("iv_hv_ratio") or 1.0
    atm_iv   = chain_data.get("atm_iv_pct", 15) / 100
    impl_move = chain_data.get("impl_move_pct", 1) / 100

    is_bull    = prob_up > 0.55
    is_bear    = prob_up < 0.45
    is_neutral = not is_bull and not is_bear
    sell_vol   = iv_hv > 1.2
    buy_vol    = iv_hv < 0.85

    candidates = []
    if is_bull and buy_vol:
        candidates.append(("BUY CALLS", 0.75, 0.7 * prob_up, "Directional + cheap vol"))
    if is_bull and sell_vol:
        candidates.append(("BULL PUT SPREAD", 0.72, 0.68 * prob_up, "Defined risk, sell premium"))
    if is_bull:
        candidates.append(("BUY CALLS", 0.60, 0.6 * prob_up, "Pure directional play"))
    if is_bear and buy_vol:
        candidates.append(("BUY PUTS", 0.75, 0.7 * (1 - prob_up), "Directional + cheap vol"))
    if is_bear and sell_vol:
        candidates.append(("BEAR CALL SPREAD", 0.72, 0.68 * (1 - prob_up), "Defined risk, sell premium"))
    if is_bear:
        candidates.append(("BUY PUTS", 0.60, 0.6 * (1 - prob_up), "Pure directional play"))
    if is_neutral and sell_vol:
        candidates.append(("SHORT STRANGLE", 0.70, 0.65, "Sell expensive premium, range bound"))
        candidates.append(("IRON CONDOR", 0.68, 0.62, "Defined risk neutral"))
    if is_neutral and buy_vol:
        candidates.append(("LONG STRADDLE", 0.55, 0.52, "Buy cheap vol, expect move"))
    if sell_vol:
        candidates.append(("COVERED CALL", 0.65, 0.60, "Yield enhancement"))

    # Deduplicate and score
    seen = set()
    out  = []
    for strat, ev_base, pop, rationale in candidates:
        if strat in seen:
            continue
        seen.add(strat)
        # DTE fit: short-term favours directional, long-term favours spreads
        dte_fit = 1.0 if dte <= 7 else (0.85 if dte <= 21 else 0.70)
        score = int(min(100, ev_base * pop * dte_fit * 100))
        out.append({
            "strategy":  strat,
            "pop":       round(pop, 3),
            "score":     score,
            "dte_fit":   round(dte_fit, 2),
            "rationale": rationale,
            "iv_env":    "SELL VOL" if sell_vol else ("BUY VOL" if buy_vol else "FAIR"),
        })
    out.sort(key=lambda x: -x["score"])
    return out[:6]

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/expiries/{symbol}")
async def get_expiries(symbol: str):
    """Get available expiry dates for a symbol."""
    try:
        expiries = _get_expiries(symbol.upper())
        return {"symbol": symbol.upper(), "expiries": expiries}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/chain/{symbol}")
async def get_chain(symbol: str, expiry: str = Query(...), rfr: float = Query(RFR_DEFAULT)):
    """Full option chain with greeks, IV, OI analysis."""
    sym = symbol.upper()
    spot = _fetch_ltp(sym)
    if not spot:
        raise HTTPException(400, f"Could not fetch LTP for {sym}")
    raw = _fetch_chain(sym, expiry)
    result = _process_chain(raw, spot, expiry, rfr)
    return result

@router.get("/analysis/{symbol}")
async def get_analysis(symbol: str, expiry: str = Query(...), rfr: float = Query(RFR_DEFAULT)):
    """Full options intelligence: chain + directional bias + strategy recommendations."""
    sym = symbol.upper()
    spot = _fetch_ltp(sym)
    if not spot:
        raise HTTPException(400, f"Could not fetch LTP for {sym}")
    raw      = _fetch_chain(sym, expiry)
    chain_dt = _process_chain(raw, spot, expiry, rfr)
    bias     = _directional_bias(sym, spot, chain_dt)
    strats   = _strategy_recommendations(bias, chain_dt, chain_dt.get("dte", 7))
    lot      = LOT_SIZES.get(sym, 500)
    return {
        "symbol":      sym,
        "spot":        spot,
        "lot_size":    lot,
        "chain":       chain_dt,
        "bias":        bias,
        "strategies":  strats,
    }

@router.get("/greeks/{symbol}")
async def get_greeks(symbol: str, strike: float = Query(...),
                     opt_type: str = Query("call"), expiry: str = Query(...),
                     rfr: float = Query(RFR_DEFAULT)):
    """Compute BS greeks for a specific strike."""
    sym  = symbol.upper()
    spot = _fetch_ltp(sym) or 0
    if not spot:
        raise HTTPException(400, "LTP fetch failed")
    today  = date.today()
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
    T = max((exp_dt - today).days, 0) / 365.0
    sigma = 0.15  # default; caller can pass IV
    g = bs_greeks(spot, strike, T, rfr / 100, sigma, opt_type.lower())
    return {"symbol": sym, "strike": strike, "opt_type": opt_type, "spot": spot,
            "dte": (exp_dt - today).days, **g}
