"""
routers/fundamentals.py — MONARCH PRO Fundamental Research Engine
FastAPI router. Data via yfinance (Yahoo Finance).

Sections:
  1. Snapshot / Scorecard
  2. Valuation ratios + peer comparison
  3. Profitability (ROE, margins over time)
  4. Growth (revenue, EPS CAGR)
  5. Financial health (D/E, current ratio, Altman-Z)
  6. Cash flow (OCF, FCF, FCF yield)
  7. Dividends
  8. Earnings (quarterly EPS, beat/miss)
  9. Analyst ratings
"""

import time
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/fundamentals", tags=["fundamentals"])

_cache: dict = {}  # {symbol: {data, ts}}
_PEER_MAP = {
    "IT":        ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE"],
    "Bank":      ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "FEDERALBNK"],
    "PSUBank":   ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK"],
    "Pharma":    ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA"],
    "Auto":      ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "FMCG":      ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP"],
    "Metal":     ["TATASTEEL", "JSWSTEEL", "HINDALCO", "SAIL", "VEDL", "COALINDIA"],
    "Energy":    ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC", "GAIL"],
    "Infra":     ["LT", "ADANIPORTS", "ULTRACEMCO", "SIEMENS", "ABB", "BEL", "HAL"],
    "Chemicals": ["PIDILITIND", "SRF", "DEEPAKNTR", "AARTIIND", "NAVINFLUOR"],
    "Realty":    ["DLF", "LODHA", "OBEROIRLTY", "PHOENIXLTD", "GODREJPROP"],
}

STOCK_SECTOR_MAP = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT","LTIM":"IT",
    "MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank",
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank","UNIONBANK":"PSUBank",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "LUPIN":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal",
    "RELIANCE":"Energy","ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy",
    "BPCL":"Energy","IOC":"Energy","GAIL":"Energy",
    "LT":"Infra","ADANIPORTS":"Infra","ULTRACEMCO":"Infra","SIEMENS":"Infra",
    "ABB":"Infra","BEL":"Infra","HAL":"Infra",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG",
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","GODREJPROP":"Realty",
}


def _safe(info: dict, key: str, scale=1.0, decimals=2):
    v = info.get(key)
    if v is None or v == "N/A":
        return None
    try:
        return round(float(v) * scale, decimals)
    except Exception:
        return None


def _fetch_yf(sym: str) -> dict:
    """Fetch full fundamental data for a symbol from yfinance."""
    import yfinance as yf
    yf_sym = sym + ".NS"
    tk   = yf.Ticker(yf_sym)
    info = tk.info or {}

    # Quarterly financials
    try:
        qfin = tk.quarterly_financials
    except Exception:
        qfin = pd.DataFrame()
    try:
        qbs = tk.quarterly_balance_sheet
    except Exception:
        qbs = pd.DataFrame()
    try:
        qcf = tk.quarterly_cashflow
    except Exception:
        qcf = pd.DataFrame()
    try:
        ann_fin = tk.financials
    except Exception:
        ann_fin = pd.DataFrame()
    try:
        ann_bs = tk.balance_sheet
    except Exception:
        ann_bs = pd.DataFrame()
    try:
        ann_cf = tk.cashflow
    except Exception:
        ann_cf = pd.DataFrame()

    def _row(df, *keys):
        for k in keys:
            if k in df.index:
                return df.loc[k]
        return pd.Series(dtype=float)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    snapshot = {
        "symbol":        sym,
        "name":          info.get("longName") or info.get("shortName") or sym,
        "sector":        info.get("sector", "—"),
        "industry":      info.get("industry", "—"),
        "market_cap_cr": _safe(info, "marketCap", 1/1e7),
        "pe_ttm":        _safe(info, "trailingPE"),
        "forward_pe":    _safe(info, "forwardPE"),
        "pb":            _safe(info, "priceToBook"),
        "ps":            _safe(info, "priceToSalesTrailing12Months"),
        "ev_ebitda":     _safe(info, "enterpriseToEbitda"),
        "ev_revenue":    _safe(info, "enterpriseToRevenue"),
        "roe":           _safe(info, "returnOnEquity", 100),
        "roa":           _safe(info, "returnOnAssets", 100),
        "profit_margin": _safe(info, "profitMargins", 100),
        "gross_margin":  _safe(info, "grossMargins", 100),
        "ebitda_margin": _safe(info, "ebitdaMargins", 100),
        "op_margin":     _safe(info, "operatingMargins", 100),
        "revenue_growth":_safe(info, "revenueGrowth", 100),
        "earn_growth":   _safe(info, "earningsGrowth", 100),
        "debt_equity":   _safe(info, "debtToEquity"),
        "current_ratio": _safe(info, "currentRatio"),
        "quick_ratio":   _safe(info, "quickRatio"),
        "dividend_yield":_safe(info, "dividendYield", 100),
        "payout_ratio":  _safe(info, "payoutRatio", 100),
        "beta":          _safe(info, "beta"),
        "eps_ttm":       _safe(info, "trailingEps"),
        "book_value":    _safe(info, "bookValue"),
        "52w_high":      _safe(info, "fiftyTwoWeekHigh"),
        "52w_low":       _safe(info, "fiftyTwoWeekLow"),
        "avg_vol_30d":   _safe(info, "averageVolume"),
        "float_shares":  _safe(info, "floatShares"),
        "analyst_target":_safe(info, "targetMeanPrice"),
        "analyst_low":   _safe(info, "targetLowPrice"),
        "analyst_high":  _safe(info, "targetHighPrice"),
        "analyst_count": _safe(info, "numberOfAnalystOpinions"),
        "recommendation":info.get("recommendationKey", "—"),
        "description":   (info.get("longBusinessSummary") or "")[:500],
    }

    # ── Annual revenue + earnings trend ──────────────────────────────────────
    rev_series  = _row(ann_fin, "Total Revenue", "Revenue")
    ni_series   = _row(ann_fin, "Net Income", "Net Income Common Stockholders")
    ebitda_row  = _row(ann_fin, "EBITDA", "Normalized EBITDA")

    def _to_cr(s):
        if s.empty:
            return []
        s = s.dropna().sort_index()
        return [{"date": str(d)[:10], "value_cr": round(float(v) / 1e7, 1)}
                for d, v in s.items() if not math.isnan(float(v))]

    annual = {
        "revenue":  _to_cr(rev_series),
        "net_income": _to_cr(ni_series),
        "ebitda":   _to_cr(ebitda_row),
    }

    # CAGR helper
    def _cagr(series_list, years):
        if len(series_list) < years + 1:
            return None
        end = series_list[-1]["value_cr"]
        start = series_list[-(years + 1)]["value_cr"]
        if start <= 0 or end <= 0:
            return None
        return round((end / start) ** (1 / years) - 1, 4) * 100

    growth = {
        "rev_1y":  _cagr(annual["revenue"], 1),
        "rev_3y":  _cagr(annual["revenue"], 3),
        "ni_1y":   _cagr(annual["net_income"], 1),
        "ni_3y":   _cagr(annual["net_income"], 3),
    }

    # ── Quarterly EPS trend ───────────────────────────────────────────────────
    try:
        eq = tk.earnings_dates
        if eq is not None and not eq.empty:
            eq = eq.dropna(subset=["Reported EPS"]).tail(8)
            eq_list = []
            for dt, row2 in eq.iterrows():
                rep = row2.get("Reported EPS")
                est = row2.get("EPS Estimate")
                if rep is not None and not (isinstance(rep, float) and math.isnan(rep)):
                    surprise = None
                    if est and not (isinstance(est, float) and math.isnan(est)) and est != 0:
                        surprise = round((float(rep) - float(est)) / abs(float(est)) * 100, 1)
                    eq_list.append({
                        "date":     str(dt)[:10],
                        "reported": round(float(rep), 2),
                        "estimate": round(float(est), 2) if est and not math.isnan(float(est)) else None,
                        "surprise_pct": surprise,
                        "beat":     (surprise or 0) > 0,
                    })
            earnings_q = list(reversed(eq_list))
        else:
            earnings_q = []
    except Exception:
        earnings_q = []

    # ── Cash flow ─────────────────────────────────────────────────────────────
    ocf_s  = _row(ann_cf, "Operating Cash Flow", "Cash From Operations")
    capex_s= _row(ann_cf, "Capital Expenditure", "Purchase Of Ppe")
    def _cf_list(s):
        if s.empty: return []
        s = s.dropna().sort_index()
        return [{"date": str(d)[:10], "value_cr": round(float(v)/1e7, 1)} for d, v in s.items()]
    ocf_list   = _cf_list(ocf_s)
    capex_list = _cf_list(capex_s)
    fcf_list   = []
    for i in range(min(len(ocf_list), len(capex_list))):
        if ocf_list[i]["date"] == capex_list[i]["date"]:
            fcf_list.append({"date": ocf_list[i]["date"],
                              "value_cr": round(ocf_list[i]["value_cr"] - abs(capex_list[i]["value_cr"]), 1)})

    cashflow = {"ocf": ocf_list, "capex": capex_list, "fcf": fcf_list}

    # ── Altman-Z score (simplified) ───────────────────────────────────────────
    altman_z = None
    try:
        mktcap = info.get("marketCap", 0) or 0
        bv_eq  = info.get("bookValue", 0) or 0
        tot_as = info.get("totalAssets", 0) or 0
        tot_li = info.get("totalDebt", 0) or 0
        wc     = (info.get("currentAssets", 0) or 0) - (info.get("currentLiabilities", 0) or 0)
        ebit   = info.get("ebitda", 0) or 0
        rev    = info.get("totalRevenue", 0) or 0
        re     = info.get("retainedEarnings", 0) or 0
        if tot_as > 0:
            x1 = wc / tot_as
            x2 = re / tot_as
            x3 = ebit / tot_as
            x4 = mktcap / max(tot_li, 1)
            x5 = rev / tot_as
            altman_z = round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + x5, 2)
    except Exception:
        pass

    return {
        "snapshot":   snapshot,
        "annual":     annual,
        "growth":     growth,
        "cashflow":   cashflow,
        "earnings_q": earnings_q,
        "altman_z":   altman_z,
        "sector":     STOCK_SECTOR_MAP.get(sym, info.get("sector", "—")),
    }


def _fetch_peer_snapshots(sector: str, exclude: str) -> list:
    """Fetch lightweight peer data for comparison table."""
    peers = [p for p in _PEER_MAP.get(sector, []) if p != exclude][:6]
    rows = []
    for p in peers:
        cached = _cache.get(p)
        if cached and time.time() - cached["ts"] < 14400:
            d = cached["data"]["snapshot"]
        else:
            try:
                import yfinance as yf
                info = yf.Ticker(p + ".NS").info or {}
                d = {
                    "symbol":        p,
                    "pe_ttm":        _safe(info, "trailingPE"),
                    "pb":            _safe(info, "priceToBook"),
                    "roe":           _safe(info, "returnOnEquity", 100),
                    "profit_margin": _safe(info, "profitMargins", 100),
                    "debt_equity":   _safe(info, "debtToEquity"),
                    "revenue_growth":_safe(info, "revenueGrowth", 100),
                    "market_cap_cr": _safe(info, "marketCap", 1/1e7),
                }
            except Exception:
                d = {"symbol": p}
        rows.append(d)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/{symbol}")
async def get_fundamentals(symbol: str, refresh: bool = Query(False)):
    """Full fundamental analysis for a NSE-listed stock."""
    sym    = symbol.upper()
    cached = _cache.get(sym)
    if not refresh and cached and (time.time() - cached["ts"]) < 14400:
        return cached["data"]
    try:
        data = _fetch_yf(sym)
        _cache[sym] = {"data": data, "ts": time.time()}
        return data
    except Exception as e:
        raise HTTPException(500, f"yfinance error for {sym}: {e}")


@router.get("/{symbol}/peers")
async def get_peers(symbol: str):
    """Peer comparison table for the stock's sector."""
    sym    = symbol.upper()
    sector = STOCK_SECTOR_MAP.get(sym, "")
    if not sector:
        raise HTTPException(404, f"Sector mapping not found for {sym}")
    # Ensure primary is fetched
    cached = _cache.get(sym)
    primary_snap = {}
    if cached:
        primary_snap = cached["data"]["snapshot"]
    peer_rows = _fetch_peer_snapshots(sector, sym)
    return {
        "symbol":  sym,
        "sector":  sector,
        "primary": primary_snap,
        "peers":   peer_rows,
    }


@router.get("/{symbol}/scorecard")
async def get_scorecard(symbol: str):
    """Quick scorecard — valuation, quality, growth grades."""
    sym    = symbol.upper()
    cached = _cache.get(sym)
    if not cached:
        raise HTTPException(404, "Not cached — call /api/fundamentals/{symbol} first")
    snap = cached["data"]["snapshot"]
    grow = cached["data"]["growth"]

    def _grade(val, lo, hi, invert=False):
        if val is None: return "—"
        good = val < lo if invert else val > hi
        med  = (lo <= val <= hi) if not invert else (lo <= val <= hi)
        return "A" if good else ("B" if med else "C")

    return {
        "symbol": sym,
        "grades": {
            "valuation_pe":    _grade(snap.get("pe_ttm"), 15, 30, invert=True),
            "valuation_pb":    _grade(snap.get("pb"), 1, 4, invert=True),
            "profitability":   _grade(snap.get("roe"), 10, 20),
            "margin":          _grade(snap.get("profit_margin"), 10, 20),
            "financial_health":_grade(snap.get("debt_equity"), 0, 1, invert=True),
            "growth_rev":      _grade(grow.get("rev_1y"), 10, 20),
            "growth_ni":       _grade(grow.get("ni_1y"), 10, 20),
        },
        "altman_z": cached["data"].get("altman_z"),
        "recommendation": snap.get("recommendation", "—"),
    }
