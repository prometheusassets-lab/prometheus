"""
MONARCH PRO — FastAPI Backend  (Refactored v2)
═══════════════════════════════════════════════
Scoring pipeline:
  compute_indicators → compute_features → compute_signals
  → compute_penalties → aggregate_score

All thresholds are data-driven (percentile ranks, z-scores, tanh
normalisation).  No magic numbers inside logic.  All constants live
in ScoreConfig.

Routers preserved:
  • upstox_auth       – OAuth
  • routers/options   – Options intelligence
  • routers/fundamentals – Fundamental research
  • routers/ml        – ML predictions
  • (Polymarket page served via static/pages/polymarket.html)
"""

# ─────────────────────────────────────────────────────────────────
# STDLIB / THIRD-PARTY
# ─────────────────────────────────────────────────────────────────
import gzip, io, json, math, os, pathlib, sqlite3, threading, time, urllib.parse
import io as _io
import json as _json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import datetime as _dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from scipy.stats import rankdata


# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURATION — single source of truth, zero magic numbers
# ═════════════════════════════════════════════════════════════════

@dataclass
class ScoreConfig:
    # ── Market session (minutes since midnight IST) ──
    NSE_OPEN_MIN:           int   = 9 * 60 + 15        # 09:15
    NSE_CLOSE_MIN:          int   = 15 * 60 + 30       # 15:30

    # ── Liquidity / ADV ──
    ADV_THRESHOLD:          float = 2e7                 # ₹2 Cr minimum ADV
    LIQ_CENTRE_LOG:         float = math.log(5e7)       # sigmoid centre (log-ADV)
    LIQ_SCALE:              float = 1.0                 # sigmoid steepness

    # ── Rolling windows ──
    PERCENTILE_WINDOW_SHORT: int  = 60
    PERCENTILE_WINDOW_LONG:  int  = 250
    RS_WINDOW_5D:            int  = 5
    RS_WINDOW_20D:           int  = 20
    BB_WINDOW:               int  = 20
    ATR_FAST:                int  = 5
    ATR_SLOW:                int  = 20
    ATR_PERIOD:              int  = 14
    EMA_FAST:                int  = 9
    EMA_MID:                 int  = 20
    EMA_SLOW:                int  = 50
    EMA_ACCEL:               int  = 5                   # for momentum acceleration
    SMA_TREND:               int  = 200                 # long-term trend SMA
    VIX_HIST_WINDOW:         int  = 252

    # ── RS / alpha normalisation ──
    RS_TANH_SCALE:          float = 1.0                 # tanh squash divisor for RS
    RS_WEIGHT_5D:           float = 0.25                # weight in BULL/default
    RS_WEIGHT_20D:          float = 0.75
    RS_WEIGHT_5D_BEAR:      float = 0.15
    RS_WEIGHT_20D_BEAR:     float = 0.85

    # ── Score component weights (sum = 1.0 each group) ──
    W_RS:                   float = 0.18
    W_RS_SECT:              float = 0.07
    W_MOMENTUM:             float = 0.12
    W_VOLUME:               float = 0.14
    W_COIL:                 float = 0.12
    W_MA:                   float = 0.10
    W_PROXIMITY:            float = 0.10
    W_VCP:                  float = 0.08
    W_DARVAS:               float = 0.05
    W_MICROSTRUCTURE:       float = 0.04   # CLV + BB + VC + spread

    # ── Signal coverage floor (fraction of valid signals required) ──
    COVERAGE_FLOOR:         float = 0.40

    # ── Interaction term amplifier ────────────────────────────────
    # When RS + volume + proximity all fire together the score gets a
    # multiplicative lift.  Cap prevents runaway scores.
    INTERACTION_FLOOR:      float = 0.60   # each signal must exceed this to count
    INTERACTION_BOOST_MAX:  float = 0.12   # max fractional boost (12 %)

    # ── Calibration-to-weight adaptation ─────────────────────────
    # How aggressively learned win-rates nudge static weights.
    # 0 = disabled, 1 = full replacement.
    CALIB_ADAPT_ALPHA:      float = 0.25   # blend factor

    # ── Universe breakout saturation guard ───────────────────────
    # If this fraction of the universe is already tagged Breakout,
    # marginal breakout probability gets discounted.
    BO_SATURATION_FLOOR:    float = 0.50   # start discounting above 50 %
    BO_SATURATION_DISCOUNT: float = 0.30   # max discount applied at 100 %

    # ── Sector distribution penalty ───────────────────────────────
    # Breakout score is discounted if sector 5D return is in bottom quartile
    SECT_DIST_PENALTY_CAP:  float = 8.0

    # ── Penalty caps (score points) ──
    RSI_OB_CAP:             float = 15.0
    VOL_LOW_CAP:            float = 12.0
    GAP_CAP:                float = 15.0
    SMA_CAP:                float = 20.0
    LIQ_CAP:                float = 15.0
    STAB_CAP:               float = 20.0
    EXT_CAP:                float = 15.0
    ALREADY_BO_CAP:         float = 18.0
    VIX_PENALTY_FLOOR:      float = -8.0
    VIX_BONUS_CAP:          float = 2.0
    BREADTH_PENALTY_FLOOR:  float = -8.0
    BREADTH_BONUS_CAP:      float = 4.0

    # ── Bonus caps ──
    BONUS_CAP:              float = 8.0
    OI_CAP:                 float = 3.0
    UV_CAP:                 float = 3.0
    CPR_CAP:                float = 3.0
    SC_CAP:                 float = 3.0
    ATR_EXP_CAP:            float = 3.0
    VCVE_CAP:               float = 3.0
    SWEEP_CAP:              float = 4.0

    # ── Candle pattern ratios ──
    CANDLE_STRONG_BODY_RATIO:  float = 0.60
    CANDLE_STRONG_CLOSE_RATIO: float = 0.75
    CANDLE_HAMMER_BODY_RATIO:  float = 0.10
    CANDLE_HAMMER_LOWER_MULT:  float = 2.0
    CANDLE_UPPER_WICK_MAX:     float = 0.40

    # ── Fetch / extraction ──
    FETCH_WORKERS:          int   = 4
    FETCH_DELAY:            float = 0.15
    FETCH_RETRIES:          int   = 4
    FETCH_BACKOFF:          float = 0.5
    LIVE_QUOTE_CHUNK:       int   = 50
    LIVE_QUOTE_DELAY:       float = 0.12
    HISTORY_DAYS:           int   = 600
    CHART_BARS:             int   = 120
    MIN_BARS:               int   = 60

    # ── Cache TTLs (seconds) ──
    MASTER_TTL:             int   = 3_600
    NIFTY50_TTL:            int   = 14_400
    MKT_CONTEXT_TTL:        int   = 900
    LIVE_REFRESH_SEC:       int   = 60

    # ── Calibration / DB ──
    FORWARD_RETURN_DAYS:    int   = 5
    SNAPSHOT_KEEP_DAYS:     int   = 30

    # ── Registry history ──
    REG_HISTORY:            int   = 200


SCORE_CFG = ScoreConfig()

# ═════════════════════════════════════════════════════════════════
# 2. APP + MIDDLEWARE
# ═════════════════════════════════════════════════════════════════

app = FastAPI(title="Monarch Pro")

_CORS_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

from upstox_auth import router as auth_router, init_state as _auth_init_state
app.include_router(auth_router)

from routers.options      import router as options_router,      init_state as _options_init_state
from routers.fundamentals import router as fundamentals_router
from routers.ml           import router as ml_router,           init_state as _ml_init_state
app.include_router(options_router)
app.include_router(fundamentals_router)
app.include_router(ml_router)


# ═════════════════════════════════════════════════════════════════
# 3. SECTOR MAP
# ═════════════════════════════════════════════════════════════════

SECTOR_TICKERS: Dict[str, str] = {
    "IT":          "^CNXIT",   "Bank":        "^NSEBANK",   "Auto":     "^CNXAUTO",
    "Pharma":      "^CNXPHARMA","Metal":       "^CNXMETAL",  "Energy":   "^CNXENERGY",
    "Infra":       "^CNXINFRA", "FMCG":        "^CNXFMCG",  "Realty":   "^CNXREALTY",
    "PSUBank":     "^CNXPSUBANK","Chemicals":  "^CNXCHEMICALS",
    "ConsumerDur": "^CNXCONSUMER","Insurance": "^CNXFINSERVICE",
    "Telecom":     "^CNXTELECOM",
    "Retail":      "^CNXCONSUMER",   # nearest proxy (NSE has no separate Retail index)
    "Logistics":   "^CNXINFRA",      # nearest proxy
}

STOCK_SECTOR_MAP: Dict[str, str] = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","MASTEK":"IT","HEXAWARE":"IT",
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank","RBLBANK":"Bank","YESBANK":"Bank",
    "CSBBANK":"Bank","DCBBANK":"Bank","KARURVYSYA":"Bank",
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "UNIONBANK":"PSUBank","BANKINDIA":"PSUBank","MAHABANK":"PSUBank",
    "INDIANB":"PSUBank","UCOBANK":"PSUBank","CENTRALBK":"PSUBank",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "APOLLOTYRE":"Auto","MRF":"Auto","CEATLTD":"Auto","EXIDEIND":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "LUPIN":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","GLENMARK":"Pharma",
    "IPCALAB":"Pharma","NATCOPHARM":"Pharma","LAURUSLABS":"Pharma",
    "FORTIS":"Pharma","METROPOLIS":"Pharma","LALPATHLAB":"Pharma",
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal","JINDALSTEL":"Metal",
    "APLAPOLLO":"Metal","RATNAMANI":"Metal","NATIONALUM":"Metal","MOIL":"Metal",
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy","HPCL":"Energy",
    "PETRONET":"Energy","OIL":"Energy","HINDPETRO":"Energy","MGL":"Energy",
    "IGL":"Energy","TATAPOWER":"Energy","ADANIGREEN":"Energy","ADANIENT":"Energy",
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEMENT":"Infra","ACC":"Infra","SIEMENS":"Infra","ABB":"Infra",
    "BEL":"Infra","HAL":"Infra","BHEL":"Infra","CUMMINSIND":"Infra",
    "THERMAX":"Infra","KEC":"Infra","KALPATPOWR":"Infra","VOLTAS":"Infra",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "EMAMILTD":"FMCG","COLPAL":"FMCG","TATACONSUM":"FMCG","UBL":"FMCG",
    "RADICO":"FMCG","VBL":"FMCG",
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "GODREJPROP":"Realty","PRESTIGE":"Realty","BRIGADE":"Realty","SOBHA":"Realty",
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ALKYLAMINE":"Chemicals",
    "FINEORG":"Chemicals","VINATIORGA":"Chemicals","BALRAMCHIN":"Chemicals",
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIPRULI":"Insurance",
    "LICIHSGFIN":"Insurance","MUTHOOTFIN":"Insurance","CHOLAFIN":"Insurance",
    "ICICIGI":"Insurance","NIACL":"Insurance","GICRE":"Insurance",
    "HDFCAMC":"Insurance","NAM-INDIA":"Insurance","ABSLAMC":"Insurance",
    "BHARTIARTL":"Telecom","IDEA":"Telecom","TATACOMM":"Telecom","INDUSTOWER":"Telecom",
    "HAVELLS":"ConsumerDur","CROMPTON":"ConsumerDur","TITAN":"ConsumerDur",
    "TRENT":"Retail","DMART":"Retail",
    "CONCOR":"Logistics","BLUEDART":"Logistics",
}

# ═════════════════════════════════════════════════════════════════
# 4. SHARED STATE
# ═════════════════════════════════════════════════════════════════

STATE: Dict = {
    "token": "",
    "raw_data_cache": {},
    "live_quotes_cache": {},
    "targets": {},
    "score_cache": {},
    "cs_rs_5d": {}, "cs_rs_20d": {},
    "cs_bb_squeeze": {}, "cs_vol_dryup": {}, "cs_clv_accum": {}, "cs_vcp": {},
    "breadth_cache": None,
    "breadth_hist": [],
    "rs_div_hist": {},
    "param_registry": {
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [],
        "stab_adj_scale": [], "stab_adj_obs": [], "pos52w_max": [],
    },
    "per_stock_winrate": {},
    "_setup_winrate": {},
    "last_live_refresh": 0,
    "extraction_status": {
        "running": False, "done": 0, "total": 0,
        "errors": 0, "rate_limited": 0, "log": []
    },
    "mkt": {},
    "sector_returns": {},
    "sector_returns_10d": {},
    "top_sectors": set(),
    "rsi_period": 7,
    "min_avg_vol": 100_000,
    "sector_cap_enabled": False,
    # Progressive streaming: scored rows are pushed here as extraction completes
    # each stock. The SSE stream reads and clears this list, pushing "row" events
    # to the frontend so the table populates stock-by-stock without waiting.
    "_row_stream_queue": [],
}
STATE_LOCK = threading.Lock()

_auth_init_state(STATE)
_options_init_state(STATE)
_ml_init_state(STATE)


# ═════════════════════════════════════════════════════════════════
# 5. LOW-LEVEL UTILITIES
# ═════════════════════════════════════════════════════════════════

def normalize_key(k: str) -> str:
    return k.replace("%7C", "|").replace(":", "|")


def to_ascending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[::-1].reset_index(drop=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return df


def get_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {STATE['token']}", "Accept": "application/json"}


def get_sector(ticker: str) -> Optional[str]:
    return STOCK_SECTOR_MAP.get(ticker.upper())


# ═════════════════════════════════════════════════════════════════
# 6. PURE INDICATOR HELPERS  (stateless, vectorised)
# ═════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def percentile_last(series: pd.Series, window: int) -> float:
    """Percentile rank of the last value over the last `window` observations."""
    s = series.tail(window).dropna()
    if len(s) < 2:
        return np.nan
    ranks = rankdata(s.values, method="average")
    return float(ranks[-1] / len(s))


def _robust_z(value: float, history: pd.Series) -> float:
    """Median + MAD robust z-score (no lookahead — history excludes current bar)."""
    h = history.dropna()
    if len(h) < 5:
        return 0.0
    med = float(h.median())
    mad = float((h - med).abs().median())
    mad = max(mad, 1e-9)
    return (value - med) / (mad * 1.4826)   # 1.4826 makes MAD consistent with σ


def _tanh_squash(z: float, scale: float) -> float:
    """Map z-score → (0, 1) via tanh."""
    return float(0.5 * (1.0 + np.tanh(z / max(scale, 1e-9))))


def _iqr_scale(series: pd.Series, fallback: float = 1.5) -> float:
    """IQR-based dispersion normaliser (σ-equivalent)."""
    s = series.dropna()
    if len(s) < 10:
        return fallback
    q75, q25 = np.percentile(s, 75), np.percentile(s, 25)
    return float(max((q75 - q25) / 1.35, 0.3))


# ═════════════════════════════════════════════════════════════════
# 7. CANDLE PATTERNS
# ═════════════════════════════════════════════════════════════════

def detect_candle_patterns(
    o: float, h: float, l: float, c: float,
    po: float, ph: float, pl: float, pc: float
) -> Tuple[float, List[str]]:
    body  = abs(c - o);  rng  = h - l + 1e-9
    upper_w = h - max(o, c); lower_w = min(o, c) - l
    prev_body = abs(pc - po); prev_rng = ph - pl + 1e-9
    patterns: List[str] = []; pts = 0.0
    cfg = SCORE_CFG
    if pc < po and c > o and c > po and o < pc:
        patterns.append("Engulfing"); pts += 3
    if (lower_w >= cfg.CANDLE_HAMMER_LOWER_MULT * body
            and upper_w <= cfg.CANDLE_UPPER_WICK_MAX * rng and c > o):
        patterns.append("Hammer"); pts += 2.5
    if h <= ph and l >= pl:
        patterns.append("InsideBar"); pts += 1.5
    if h > ph and l < pl and c > o and c > (h + l) / 2:
        patterns.append("OutsideBar"); pts += 2
    if (body / rng > cfg.CANDLE_STRONG_BODY_RATIO and c > o
            and (c - l) / rng > cfg.CANDLE_STRONG_CLOSE_RATIO):
        patterns.append("StrongGreen"); pts += 2
    if body / rng < cfg.CANDLE_HAMMER_BODY_RATIO and lower_w > 1.5 * upper_w:
        patterns.append("BullDoji"); pts += 1
    if pc < po and prev_body / prev_rng > 0.5 and c > o and c > (po + pc) / 2:
        patterns.append("MorningStar"); pts += 2.5
    _gap_min = max(prev_body * 0.5, prev_rng * 0.01)
    if o > pc + _gap_min and c > o:
        patterns.append("GapContinue"); pts += 2
    return min(pts, 10.0), patterns


# ═════════════════════════════════════════════════════════════════
# 8. STRUCTURAL PATTERN DETECTORS  (Darvas, VCP, BB, CLV, Vol Dryup)
# ═════════════════════════════════════════════════════════════════

def darvas_box_score(df: pd.DataFrame, atr_v: float) -> dict:
    null = {"darvas_score": 0.0, "box_high": np.nan, "box_low": np.nan,
            "in_box": False, "bars_in_box": 0, "box_atr_ratio": np.nan}
    if len(df) < 20:
        return null
    hh, hl, hc = df["high"], df["low"], df["close"]
    tr = pd.concat([(hh - hl), (hh - hc.shift(1)).abs(), (hl - hc.shift(1)).abs()], axis=1).max(axis=1)
    _atr_v = float(tr.ewm(alpha=1 / SCORE_CFG.ATR_PERIOD, adjust=False).mean().iloc[-1]) \
             if len(tr) >= 5 else atr_v
    _daily_med = float(tr.tail(20).median()) if len(tr) >= 20 else (_atr_v * 0.5)
    _bars_per_atr = max(3, int(_atr_v / (_daily_med + 1e-9)))
    _confirm_n = int(np.clip(_bars_per_atr, 3, 10))
    _box_high = _box_low = _box_start = None
    for i in range(len(hh) - 1, _confirm_n * 2, -1):
        ws = max(0, i - _confirm_n)
        ch = float(hh.iloc[ws:i + 1].max())
        cs, ce = i + 1, min(len(hh), i + 1 + _confirm_n)
        if ce > len(hh):
            continue
        if float(hh.iloc[cs:ce].max()) > ch:
            continue
        _box_high = ch
        _box_low = float(hl.iloc[ws:].min())
        _box_start = ws
        break
    if _box_high is None:
        return null
    ltp = float(hc.iloc[-1])
    _bw = _box_high - _box_low + 1e-9
    bars_in_box = len(hh) - _box_start
    in_box = _box_low <= ltp <= _box_high
    bar = _bw / (atr_v + 1e-9)
    tightness  = 1.0 / (1.0 + bar)
    pos_in_box = (ltp - _box_low) / _bw
    pos_score  = float(np.clip(1.0 - 4.0 * (pos_in_box - 0.5) ** 2, 0.0, 1.0))
    _coiling_frac = float(
        (hh.rolling(_confirm_n).max() <= hh.rolling(_confirm_n).max().shift(_confirm_n))
        .fillna(False).tail(60).mean()
    ) if len(hh) >= 10 else 0.5
    _typical_dur = max(int(_coiling_frac * 20), 3)
    time_score = float(np.clip(bars_in_box / (_typical_dur * 2.0 + 1e-9), 0.0, 1.0))
    darvas_score = round((tightness * 0.40 + pos_score * 0.35 + time_score * 0.25) * 10.0, 1)
    return {"darvas_score": darvas_score, "box_high": round(_box_high, 2),
            "box_low": round(_box_low, 2), "in_box": in_box,
            "bars_in_box": bars_in_box, "box_atr_ratio": round(bar, 2)}


def bb_width_compression_score(c: pd.Series,
                                window: int = 20,
                                mult: float = 2.0) -> Tuple[float, float]:
    if len(c) < window + 10:
        return 0.5, 0.5
    sma = c.rolling(window).mean()
    std = c.rolling(window).std()
    bw  = (2.0 * mult * std) / sma.replace(0, np.nan)
    bw  = bw.dropna()
    if len(bw) < 10:
        return 0.5, 0.5
    cur = float(bw.iloc[-1])
    pct = float((bw.iloc[:-1] <= cur).mean())
    return round(pct, 4), round(1.0 - pct, 4)


def volume_dryup_score(v: pd.Series,
                        short: int = 5, long: int = 20) -> Tuple[float, float]:
    if len(v) < long + short:
        return 1.0, 0.5
    v = v.replace(0, np.nan).dropna()
    if len(v) < long:
        return 1.0, 0.5
    ratio  = float(v.tail(short).mean()) / (float(v.tail(long).mean()) + 1e-9)
    h_roll = (v.rolling(short).mean() / (v.rolling(long).mean() + 1e-9)).dropna().iloc[:-1]
    if len(h_roll) < 5:
        return round(ratio, 4), round(float(np.clip(1.0 - ratio, 0.0, 1.0)), 4)
    pct = float((h_roll <= ratio).mean())
    return round(ratio, 4), round(1.0 - pct, 4)


def clv_accumulation_score(c: pd.Series, h: pd.Series,
                            l: pd.Series, v: pd.Series,
                            window: int = 20) -> Tuple[float, float]:
    if len(c) < window + 5:
        return 0.0, 0.5
    hl  = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / hl
    mf  = clv.fillna(0) * v
    mfn = mf / v.rolling(window).mean().replace(0, np.nan)
    rmf = mfn.rolling(window).sum().dropna()
    if len(rmf) < 5:
        return float(clv.iloc[-1]) if pd.notna(clv.iloc[-1]) else 0.0, 0.5
    cur = float(rmf.iloc[-1])
    pct = float((rmf.iloc[:-1] <= cur).mean())
    return round(cur, 4), round(pct, 4)


def detect_vcp(c: pd.Series, h: pd.Series, l: pd.Series,
               v: pd.Series, atr: pd.Series) -> dict:
    _N = {"vcp_score": 0.0, "vcp_pullback_n": 0, "vcp_contraction": 0.5,
          "vcp_vol_comp": 0.5, "vcp_vol_dryup": 0.5, "vcp_tightness": 0.5,
          "vcp_position": 0.5, "vcp_detected": False}
    if len(c) < 60:
        return _N
    hc, hh, hl, hv, ha = c.iloc[:-1], h.iloc[:-1], l.iloc[:-1], v.iloc[:-1], atr.iloc[:-1]
    dr = (hh - hl).replace(0, np.nan).dropna()
    if len(dr) < 20:
        return _N
    med_r = float(dr.tail(60).median())
    cur_atr = float(ha.dropna().tail(1).iloc[0]) if len(ha.dropna()) >= 1 else med_r
    if med_r <= 0 or cur_atr <= 0:
        return _N
    sw = max(3, min(25, int(round(float(np.clip(cur_atr / med_r * 5.0, 3.0, 25.0))))))
    n  = len(hc)
    if n < sw * 4:
        return _N
    rmax = hh.rolling(2 * sw + 1, center=True).max()
    rmin = hl.rolling(2 * sw + 1, center=True).min()
    shi_idx = hh.index[hh == rmax].tolist()
    sli_idx = hl.index[hl == rmin].tolist()
    pullbacks = []
    for shi in shi_idx:
        sp = hh.index.get_loc(shi)
        sv = float(hh.loc[shi])
        subs = [s for s in sli_idx if hl.index.get_loc(s) > sp]
        if not subs:
            continue
        sli = subs[0]
        slv = float(hl.loc[sli])
        if sv <= 0:
            continue
        depth = (sv - slv) / (sv + 1e-9)
        if not (0 < depth < 0.99):
            continue
        pullbacks.append({"depth": depth, "sh_pos": sp,
                          "sl_pos": hl.index.get_loc(sli),
                          "sh_val": sv, "sl_val": slv})
    if len(pullbacks) < 2:
        return _N
    all_depths = [p["depth"] for p in pullbacks]
    recent = [p for p in pullbacks if p["sh_pos"] >= n - 60] or pullbacks[-min(4, len(pullbacks)):]
    rd = [p["depth"] for p in recent]
    if len(rd) < 2:
        return _N
    x = np.arange(len(rd), dtype=float)
    slope = float(np.polyfit(x, rd, 1)[0]) if len(rd) >= 2 else 0.0
    all_slopes = [float(np.polyfit(np.arange(i, dtype=float), all_depths[:i], 1)[0])
                  for i in range(2, len(all_depths))]
    if len(all_slopes) >= 3:
        contraction = 1.0 - float((np.array(all_slopes) >= slope).mean())
    else:
        std_d = float(np.std(all_depths)) if np.std(all_depths) > 0 else 1e-9
        contraction = float(np.clip(-slope / std_d, 0, 1))
    _, vc_sc   = bb_width_compression_score(hc, window=min(20, len(hc) // 3))
    _, vdu_sc  = volume_dryup_score(hv)
    last_sh, last_sl = recent[-1]["sh_val"], recent[-1]["sl_val"]
    cons_rng = last_sh - last_sl + 1e-9
    _atr_c = float(ha.dropna().iloc[-1]) if len(ha.dropna()) >= 1 else cons_rng * 0.5
    tight_r  = cons_rng / (_atr_c + 1e-9)
    _th = [((p["sh_val"] - p["sl_val"]) /
             (float(ha.iloc[min(p["sh_pos"], len(ha) - 1)]) + 1e-9))
           for p in recent]
    tight_sc = float(np.clip(1.0 - tight_r / (np.percentile(_th, 75) + 1e-9), 0.0, 1.0)) \
               if len(_th) >= 3 else float(np.clip(1.0 - tight_r / 3.0, 0.0, 1.0))
    pos_sc   = float(np.clip((float(c.iloc[-1]) - last_sl) / cons_rng, 0.0, 1.0))
    vcp_score = float(np.clip(np.mean([contraction, vc_sc, vdu_sc, tight_sc, pos_sc]), 0.0, 1.0))
    detected  = vcp_score >= 0.55 and len(recent) >= 2 and contraction >= 0.4
    return {
        "vcp_score": round(vcp_score, 3), "vcp_pullback_n": len(recent),
        "vcp_contraction": round(contraction, 3), "vcp_vol_comp": round(vc_sc, 3),
        "vcp_vol_dryup": round(vdu_sc, 3), "vcp_tightness": round(tight_sc, 3),
        "vcp_position": round(pos_sc, 3), "vcp_detected": detected,
    }


# ═════════════════════════════════════════════════════════════════
# 9. RELATIVE STRENGTH  (vol-normalised alpha + tanh squash)
# ═════════════════════════════════════════════════════════════════

def _vol_normalised_rs(c: pd.Series, bench_r5: Optional[float],
                        bench_r20: Optional[float],
                        regime: str = "BULL") -> float:
    cfg = SCORE_CFG
    if len(c) < 23:
        return 0.5
    base6  = float(c.iloc[-7])  if len(c) >= 7  else float(c.iloc[0])
    base21 = float(c.iloc[-22]) if len(c) >= 22 else float(c.iloc[0])
    end_p  = float(c.iloc[-2])
    sr5    = end_p / base6  - 1 if base6  != 0 else 0.0
    sr20   = end_p / base21 - 1 if base21 != 0 else 0.0
    dr     = c.pct_change().dropna()
    std5   = max(float(dr.tail(5).std())  if len(dr) >= 5  else 0.01, 0.001)
    std20  = max(float(dr.tail(20).std()) if len(dr) >= 20 else 0.01, 0.001)
    a5  = (sr5  - (bench_r5  or 0.0)) / (std5  * np.sqrt(cfg.RS_WINDOW_5D)  + 1e-9)
    a20 = (sr20 - (bench_r20 or 0.0)) / (std20 * np.sqrt(cfg.RS_WINDOW_20D) + 1e-9)
    # Adaptive IQR scale from historical alpha distribution
    h5  = (c.pct_change(5).dropna()  - (bench_r5  or 0.0)) / (std5  * np.sqrt(5)  + 1e-9)
    h20 = (c.pct_change(20).dropna() - (bench_r20 or 0.0)) / (std20 * np.sqrt(20) + 1e-9)
    w5  = _iqr_scale(h5)
    w20 = _iqr_scale(h20)
    rs5  = _tanh_squash(a5  / w5,  cfg.RS_TANH_SCALE)
    rs20 = _tanh_squash(a20 / w20, cfg.RS_TANH_SCALE)
    if regime == "BEAR":
        return rs5 * cfg.RS_WEIGHT_5D_BEAR + rs20 * cfg.RS_WEIGHT_20D_BEAR
    return rs5 * cfg.RS_WEIGHT_5D + rs20 * cfg.RS_WEIGHT_20D


# ═════════════════════════════════════════════════════════════════
# 10.  SCORING PIPELINE  (5 pure stages)
# ═════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, rsi_period: int) -> dict:
    """Stage 1 — raw indicators, no scoring logic."""
    cfg = SCORE_CFG
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    e9  = _ema(c, cfg.EMA_FAST)
    e20 = _ema(c, cfg.EMA_MID)
    e50 = _ema(c, cfg.EMA_SLOW)
    e5  = _ema(c, cfg.EMA_ACCEL)
    atr = _atr(df, cfg.ATR_PERIOD)
    rsi = _rsi_wilder(c, rsi_period)
    tr  = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr5  = tr.rolling(cfg.ATR_FAST).mean()
    atr20 = tr.rolling(cfg.ATR_SLOW).mean()
    sma200_n = min(cfg.SMA_TREND, len(c))
    sma200   = float(c.tail(sma200_n).mean())
    vol_ma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
    return {
        "c": c, "h": h, "l": l, "v": v,
        "e9": e9, "e20": e20, "e50": e50, "e5": e5,
        "atr": atr, "tr": tr, "atr5": atr5, "atr20": atr20,
        "rsi": rsi, "sma200": sma200, "vol_ma20": vol_ma20,
    }


def compute_features(ind: dict, ticker: str, ltp: float, day_vol: float,
                      day_hi: float, day_lo: float, day_o: float,
                      nifty_r5: Optional[float], nifty_r20: Optional[float],
                      market_ctx: dict, cs_state: dict,
                      param_registry: dict) -> dict:
    """Stage 2 — all normalized (0-1) feature scores. Pure, no global mutation."""
    cfg  = SCORE_CFG
    c, h, l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    e9, e20, e50, e5 = ind["e9"], ind["e20"], ind["e50"], ind["e5"]
    atr, tr = ind["atr"], ind["tr"]
    atr5, atr20 = ind["atr5"], ind["atr20"]
    rsi = ind["rsi"]
    sma200, vol_ma20 = ind["sma200"], ind["vol_ma20"]

    n = len(c)
    atr_v   = float(atr.iloc[-1]);  atr_v = max(atr_v, 1e-9)
    rsi_v   = float(rsi.iloc[-1]);  rsi_p = float(rsi.iloc[-2])
    e9_v    = float(e9.iloc[-1]);   e20_v = float(e20.iloc[-1]);  e50_v = float(e50.iloc[-1])
    e9_y    = float(e9.iloc[-2]);   e20_y = float(e20.iloc[-2])
    atr_pct = (atr_v / ltp) * 100 if ltp > 0 else 0.0

    regime  = market_ctx.get("regime", "BULL")
    _w5, _w20 = (cfg.RS_WEIGHT_5D_BEAR, cfg.RS_WEIGHT_20D_BEAR) if regime == "BEAR" \
                 else (cfg.RS_WEIGHT_5D,      cfg.RS_WEIGHT_20D)

    # ── RS features ──────────────────────────────────────────────
    cs5  = cs_state.get("cs_rs_5d",  {}).get(ticker)
    cs20 = cs_state.get("cs_rs_20d", {}).get(ticker)
    cs_rs = (cs5 * _w5 + cs20 * _w20) if (cs5 is not None and cs20 is not None) \
            else (cs5 if cs5 is not None else (cs20 if cs20 is not None else 0.5))
    rs_slope = float(np.clip(0.5 + ((cs5 or 0.5) - (cs20 or 0.5)) * 2.0, 0.0, 1.0))
    abs_rs   = _vol_normalised_rs(c, nifty_r5, nifty_r20, regime)
    rs_combined = cs_rs * 0.40 + rs_slope * 0.35 + abs_rs * 0.25

    # RS momentum acceleration
    vel  = e5 - e20
    acc  = vel.diff()
    ah   = acc.iloc[:-1]
    if len(ah) >= 10:
        az    = _robust_z(float(acc.iloc[-2]) if len(acc) >= 2 else 0.0, ah)
        acc_w = _iqr_scale(ah)
        acc_sc = _tanh_squash(az, acc_w)
    else:
        acc_sc = 0.5

    # RS divergence bonus (5d momentum vs 20d)
    rs_div = (cs5 - cs20) if (cs5 is not None and cs20 is not None) else 0.0
    prev_divs = cs_state.get("rs_div_hist", {}).get(ticker, [])
    rs_div_pct = float((np.array(prev_divs) <= rs_div).mean()) if len(prev_divs) >= 10 else 0.5

    # ── Sector RS ─────────────────────────────────────────────────
    sect = STOCK_SECTOR_MAP.get(ticker.upper())
    sect_ret  = market_ctx.get("sector_returns",     {}).get(sect)
    sect_r10  = market_ctx.get("sector_returns_10d", {}).get(sect)
    sect_name = sect or "?"
    if sect_ret is not None and n >= 7:
        base6 = float(c.iloc[-6])
        sr5   = float(c.iloc[-1] / base6 - 1) if base6 != 0 else 0.0
        dr    = c.pct_change().dropna()
        sv    = max(float(dr.tail(20).std() * np.sqrt(5)) if len(dr) >= 10 else 0.02, 0.005)
        sb_z  = (sr5 - sect_ret) / sv
        rs_sect = _tanh_squash(sb_z, 1.0)
        all_sv  = sorted(market_ctx.get("sector_returns", {}).values())
        if len(all_sv) > 1:
            sr_rank = sum(1 for x in all_sv if x <= sect_ret) / len(all_sv)
            rs_sect = min(1.0, rs_sect + 0.15 * sr_rank)
        if sect_r10 is not None and len(all_sv) > 2:
            accel_z  = (sect_ret - sect_r10) / max(float(pd.Series(all_sv).std()), 1e-4)
            rs_sect  = float(np.clip(rs_sect + 0.10 * np.tanh(accel_z), 0.0, 1.0))
    else:
        rs_sect = 0.0

    # ── MA structure (continuous, percentile-rank based) ──────────
    ratio_hist = (e9 / e50.replace(0, np.nan)).dropna()
    ma_ratio_pct = percentile_last(ratio_hist, min(cfg.PERCENTILE_WINDOW_LONG, len(ratio_hist)))
    if pd.isna(ma_ratio_pct):
        ma_ratio_pct = 0.5
    # EMA9 slope (normalised by ATR)
    e9_slope = 0.5
    if n >= 4:
        slope_raw = (float(e9.iloc[-1]) - float(e9.iloc[-4])) / (atr_v * 3.0 + 1e-9)
        e9_slope  = float(np.clip(0.5 + slope_raw * 2.0, 0.0, 1.0))
    # EMA9 vs EMA20 convergence
    gap_now  = (e9_v  - e20_v) / (atr_v + 1e-9)
    gap_prev = (e9_y  - e20_y) / (atr_v + 1e-9)
    cross_prox  = float(np.exp(-abs(gap_now) * 2.0))
    converge_sc = float(np.clip(cross_prox * (1.2 if gap_now > gap_prev else 0.8), 0.0, 1.0))
    above_e50   = float(np.clip((ltp - e50_v) / (atr_v + 1e-9) + 0.5, 0.0, 1.0))
    ma_feature  = e9_slope * 0.35 + converge_sc * 0.45 + above_e50 * 0.20

    # ── Momentum acceleration (percentile rank) ───────────────────
    acc_rank = percentile_last(acc.dropna(), min(cfg.PERCENTILE_WINDOW_SHORT, len(acc.dropna())))
    if pd.isna(acc_rank):
        acc_rank = 0.5

    # ── Volatility contraction (ATR5/ATR20 inverted) ──────────────
    vc_series = atr5 / (atr20.replace(0, np.nan))
    vc_pct    = percentile_last(vc_series, min(cfg.PERCENTILE_WINDOW_LONG, len(vc_series)))
    if pd.isna(vc_pct):
        vc_pct = 0.5
    vc_feature = 1.0 - vc_pct   # lower ATR5/ATR20 = tighter = better

    vc_ratio_now = float(atr5.iloc[-1]) / (float(atr20.iloc[-1]) + 1e-9) \
                   if pd.notna(atr5.iloc[-1]) and pd.notna(atr20.iloc[-1]) else 1.0

    # ── Range compression (inverted percentile) ───────────────────
    rng_ser = ((h.rolling(5).max() - l.rolling(5).min()) /
               (h.rolling(20).max() - l.rolling(20).min() + 1e-9))
    rci_pct = percentile_last(rng_ser, min(cfg.PERCENTILE_WINDOW_LONG, len(rng_ser)))
    if pd.isna(rci_pct):
        rci_pct = 0.5
    rci_feature = 1.0 - rci_pct
    rci_val     = float(rng_ser.iloc[-1]) if pd.notna(rng_ser.iloc[-1]) else 1.0

    # ── Volume features ───────────────────────────────────────────
    vol_mu    = float(v.tail(20).mean()) if len(v) >= 5 else vol_ma20
    vol_sigma = max(float(v.tail(20).std()) if len(v) > 1 else vol_mu * 0.3, vol_mu * 0.05)
    vol_z     = (float(v.iloc[-1]) - vol_mu) / (vol_sigma + 1e-9)
    vol_ratio = float(v.iloc[-1]) / (vol_ma20 + 1e-9)
    # Institutional ratio (5-day avg / 20-day avg)
    inst_ratio = float(v.tail(5).mean()) / (vol_ma20 + 1e-9)
    _inst_hist = (v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)).dropna()
    if len(_inst_hist) >= 20:
        ic = float(_inst_hist.tail(60).median())
        is_ = max(float(_inst_hist.tail(60).std()), 0.05)
    else:
        ic, is_ = 1.2, 0.33
    inst_feature = float(1.0 / (1.0 + np.exp(-((inst_ratio - ic) / is_))))

    # Volume trend (5-bar slope, normalised)
    if n >= 8:
        v5_vals = v.tail(5).values.astype(float)
        v5_slope = float(np.polyfit(np.arange(5, dtype=float), v5_vals, 1)[0]) / (vol_mu + 1e-9)
        v_trend_pct = float((v.rolling(5).mean().dropna() <= float(v5_vals.mean())).mean())
        vol_signal = float(np.clip(0.5 + v5_slope * 2.0, 0.0, 1.0)) * 0.6 + v_trend_pct * 0.4
    else:
        vol_signal = 0.5

    # Up-volume skew
    uv_feature = 0.5
    if n >= 20:
        up_mask  = c.diff() > 0;  dn_mask = c.diff() < 0
        up_vol   = float(v[up_mask].tail(20).sum())
        dn_vol   = float(v[dn_mask].tail(20).sum())
        uv_ratio = up_vol / (dn_vol + 1e-9)
        uv_hist  = pd.Series([
            v[up_mask].iloc[max(0, i - 20):i].sum() / (v[dn_mask].iloc[max(0, i - 20):i].sum() + 1e-9)
            for i in range(20, min(60, n))
        ], dtype=float)
        uv_feature = float((uv_hist <= uv_ratio).mean()) if len(uv_hist) >= 5 else (0.7 if uv_ratio > 1.0 else 0.4)

    # Close Position Ratio (close location in range)
    cpr_feature = 0.5
    if n >= 10:
        hl_r    = (h - l).replace(0, np.nan)
        cpr_raw = ((c - l) / hl_r).dropna()
        cpr10   = float(cpr_raw.tail(10).mean())
        cpr_h   = cpr_raw.rolling(10).mean().dropna()
        cpr_feature = float((cpr_h <= cpr10).mean()) if len(cpr_h) >= 10 else round(cpr10, 3)

    # Spread compression + rising close
    sc_feature = 0.5
    if n >= 15:
        r5d  = float(h.tail(5).max() - l.tail(5).min())
        r10d = float(h.tail(10).max() - l.tail(10).min())
        comp = 1.0 - (r5d / (r10d + 1e-9))
        try:
            cslope = float(np.polyfit(range(5), c.tail(5).values, 1)[0]) / (atr_v + 1e-9)
        except Exception:
            cslope = 0.0
        qa = max(0.0, comp) * max(0.0, cslope)
        if n >= 20:
            x_    = np.arange(5, dtype=float)
            sx, sx2 = x_.sum(), (x_ ** 2).sum()
            dn_   = 5 * sx2 - sx ** 2
            cv    = c.values.astype(float)
            sw    = np.lib.stride_tricks.sliding_window_view(cv, 5)
            slop_ = (5 * (sw * x_).sum(axis=1) - sx * sw.sum(axis=1)) / (dn_ + 1e-9)
            av_   = atr.iloc[:-1].values.astype(float)[4:]
            nw    = min(len(slop_), len(av_))
            slop_ = slop_[:nw]; av_ = av_[:nw]
            ra5  = np.array([h.values[i:i+5].max() - l.values[i:i+5].min()   for i in range(nw)])
            ra10 = np.array([h.values[max(0,i-4):i+6].max() - l.values[max(0,i-4):i+6].min() + 1e-9 for i in range(nw)])
            ch_  = np.clip(1.0 - ra5 / ra10, 0.0, 1.0)
            sn_  = np.clip(slop_ / (av_ + 1e-9), 0.0, None)
            sch  = pd.Series((ch_ * sn_)[-60:], dtype=float).dropna()
            sc_feature = float((sch <= qa).mean()) if len(sch) >= 5 else float(np.clip(qa * 3.0, 0.0, 1.0)) / 3.0

    # ATR expansion onset (post-compression)
    atr_exp_feature = 0.0
    if len(tr) >= 25:
        vc_r   = (atr5 / (atr20.replace(0, np.nan))).dropna()
        vc_p30 = float(vc_r.quantile(0.30)) if len(vc_r) >= 10 else 0.85
        va     = vc_r.values
        vd     = np.diff(va)
        was_compressed = False; bars_since = None
        for j in range(len(vd) - 1, -1, -1):
            if va[j] < vc_p30:
                was_compressed = True
            if was_compressed and vd[j] > 0:
                bars_since = len(vd) - j
                break
        if bars_since is not None and bars_since <= 5:
            atr_exp_feature = float(np.clip(1.0 / bars_since, 0.0, 1.0))

    # ── Position in 52-week range ─────────────────────────────────
    n250   = min(cfg.PERCENTILE_WINDOW_LONG, n)
    hi250  = float(h.tail(n250).max()); lo250 = float(l.tail(n250).min())
    pos52w = (ltp - lo250) / (hi250 - lo250 + 1e-9)
    pos_ser = (c - c.rolling(n250).min()) / (c.rolling(n250).max() - c.rolling(n250).min() + 1e-9)
    pos_pct = percentile_last(pos_ser, min(n250, len(pos_ser)))
    if pd.isna(pos_pct):
        pos_pct = pos52w

    # ── Stability (% of last-20 closes positive) ──────────────────
    if n >= 21:
        stability = float((c.iloc[-20:].pct_change().dropna() > 0).mean())
    elif n >= 11:
        stability = float((c.iloc[-10:].pct_change().dropna() > 0).mean())
    else:
        stability = 0.5
    if n >= 40:
        stab_ser = c.pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / max(len(x.dropna()), 1), raw=False
        ).dropna()
        stab_pct = percentile_last(stab_ser, min(60, len(stab_ser)))
    else:
        stab_pct = None

    # ── Liquidity ─────────────────────────────────────────────────
    price_med = float(c.tail(20).median()) if n >= 20 else ltp
    adv_turnover  = vol_ma20 * price_med
    liq_logadv    = float(np.log(adv_turnover + 1.0))
    liquidity_sc  = float(1.0 / (1.0 + np.exp(-cfg.LIQ_SCALE * (liq_logadv - cfg.LIQ_CENTRE_LOG))))

    # ── Cross-sectional signals (computed externally, read here) ──
    bb_cs  = cs_state.get("cs_bb_squeeze", {}).get(ticker)
    if bb_cs is None:
        _, bb_cs = bb_width_compression_score(c)
    vdu_cs = cs_state.get("cs_vol_dryup", {}).get(ticker)
    if vdu_cs is None:
        _, vdu_cs = volume_dryup_score(v)
    clv_cs = cs_state.get("cs_clv_accum", {}).get(ticker)
    if clv_cs is None:
        _, clv_cs = clv_accumulation_score(c, h, l, v)
    vcp_cs = cs_state.get("cs_vcp", {}).get(ticker)

    # ── ADR / base data ───────────────────────────────────────────
    base_hi  = float(h.tail(20).max());  base_lo = float(l.tail(20).min())
    base_rng = base_hi - base_lo + 1e-9
    breakout_ext = (ltp - base_hi) / (atr_v + 1e-9)

    # ── OI buildup (F&O stocks) ───────────────────────────────────
    oi_feature = 0.0
    # NOTE: df not available here; must be passed or handled in aggregate

    return dict(
        # Meta
        regime=regime, ltp=ltp, atr_v=atr_v, atr_pct=atr_pct,
        e9_v=e9_v, e20_v=e20_v, e50_v=e50_v,
        rsi_v=rsi_v, rsi_p=rsi_p,
        # RS
        rs_combined=rs_combined, acc_sc=acc_sc, rs_div_pct=rs_div_pct,
        rs_sect=rs_sect, sect_name=sect_name,
        # MA
        ma_feature=ma_feature, acc_rank=acc_rank,
        # Volatility
        vc_feature=vc_feature, rci_feature=rci_feature,
        vc_ratio_now=vc_ratio_now, rci_val=rci_val,
        # Volume
        vol_mu=vol_mu, vol_sigma=vol_sigma, vol_z=vol_z,
        vol_ratio=vol_ratio, vol_signal=vol_signal,
        inst_feature=inst_feature, inst_ratio=inst_ratio,
        uv_feature=uv_feature, cpr_feature=cpr_feature,
        sc_feature=sc_feature, atr_exp_feature=atr_exp_feature,
        # Structure
        pos52w=pos52w, pos_pct=pos_pct,
        stability=stability, stab_pct=stab_pct,
        liquidity_sc=liquidity_sc, adv_turnover=adv_turnover,
        base_hi=base_hi, base_lo=base_lo, base_rng=base_rng,
        breakout_ext=breakout_ext,
        # Cross-sectional
        bb_cs=bb_cs, vdu_cs=vdu_cs, clv_cs=clv_cs, vcp_cs=vcp_cs,
    )


def compute_signals(feat: dict, ind: dict, df: pd.DataFrame,
                     ticker: str, ltp: float, day_vol: float,
                     day_hi: float, day_lo: float, day_o: float,
                     nifty_r5: Optional[float], nifty_r20: Optional[float],
                     cs_state: dict, param_registry: dict,
                     rsi_period: int, elapsed_frac: float,
                     vol_ma20: float, session_mins: int,
                     elapsed_mins: int) -> dict:
    """Stage 3 — per-setup continuous signals (0-1 each)."""
    cfg  = SCORE_CFG
    c, h, l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    atr, tr    = ind["atr"], ind["tr"]
    atr5, atr20 = ind["atr5"], ind["atr20"]
    e9, e20, e50 = ind["e9"], ind["e20"], ind["e50"]
    rsi        = ind["rsi"]

    atr_v   = feat["atr_v"]
    e9_v    = feat["e9_v"];   e20_v = feat["e20_v"];   e50_v = feat["e50_v"]
    rsi_v   = feat["rsi_v"];  rsi_p = feat["rsi_p"]
    vol_mu  = feat["vol_mu"]; vol_z = feat["vol_z"]
    base_hi = feat["base_hi"]; base_lo = feat["base_lo"]; base_rng = feat["base_rng"]
    breakout_ext = feat["breakout_ext"]

    n = len(c)

    # ── Determine setup type ──────────────────────────────────────
    # Volume percentile thresholds (adaptive)
    vol_bo_thresh = float(v.tail(min(60, n)).quantile(0.85))

    # Extension percentiles
    ext_hist = ((c - h.rolling(20).max().shift(1)) / (atr.iloc[:-1] + 1e-9)).dropna()
    ext_p10  = float(ext_hist.quantile(0.10)) if len(ext_hist) >= 20 else -1.5
    ext_p90  = float(ext_hist.quantile(0.90)) if len(ext_hist) >= 20 else  0.3
    ext_p10  = min(ext_p10, -0.3);  ext_p90 = max(ext_p90, 0.5)

    t1_vol_ratio = float(v.tail(3).max()) / (vol_ma20 + 1e-9)
    hi10d        = float(h.tail(10).max())
    washout_depth = (hi10d - ltp) / (atr_v + 1e-9)
    t1_bar_rng   = float(h.iloc[-1]) - float(l.iloc[-1])
    t1_close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / (t1_bar_rng + 1e-9)

    is_reversal = (rsi_v < float(rsi.tail(60).quantile(0.35)) and
                   t1_vol_ratio >= 1.3 and washout_depth >= 1.5)

    near_e9  = abs(ltp - e9_v)  / (atr_v + 1e-9) < 1.0
    near_e20 = abs(ltp - e20_v) / (atr_v + 1e-9) < 1.0
    above_e50 = ltp > e50_v - atr_v
    hi10d_pb  = float(h.tail(10).max())
    real_pb   = ltp < (hi10d_pb - 0.3 * atr_v)

    if is_reversal:
        setup_type = "Reversal"
    elif breakout_ext >= ext_p10 and breakout_ext <= ext_p90 and day_vol >= vol_bo_thresh:
        # Price at/near base high AND volume confirming: genuine breakout signal
        setup_type = "Breakout"
    elif above_e50 and (near_e9 or near_e20):
        # Price pulled back to a key moving average while remaining above EMA50
        setup_type = "Pullback" if real_pb else "Breakout"
    elif breakout_ext > ext_p90:
        # Already extended well past the base — treat as post-breakout pullback
        # unless volume is surging (continuation breakout)
        setup_type = "Breakout" if day_vol >= vol_bo_thresh else "Pullback"
    elif breakout_ext >= ext_p10:
        # Within base range but volume not confirming: base still building → Pullback
        setup_type = "Pullback"
    else:
        # Below base range (deep correction)
        setup_type = "Pullback"

    # ── Coil score (adaptive: Breakout vs Pullback) ───────────────
    _sw = 20
    if n >= 40:
        peaks    = (h.rolling(3, center=True).max() == h).astype(int)
        pidx     = h.index[peaks == 1].tolist()
        if len(pidx) >= 3:
            pgaps = [pidx[i+1] - pidx[i] for i in range(len(pidx)-1)
                     if isinstance(pidx[i+1] - pidx[i], (int, float))]
            if pgaps:
                _sw = int(np.clip(np.median(pgaps), 10, 40))

    if setup_type == "Breakout":
        r5b   = float(h.tail(5).max()) - float(l.tail(5).min())
        tight = 1.0 - min(1.0, r5b / (base_rng + 1e-9))
        rhi   = h.tail(8)
        hsp   = (rhi.max() - rhi.min()) / (atr_v + 1e-9)
        flat  = max(0.0, 1.0 - min(hsp / 1.0, 1.0))
        th    = (1.0 - (h.rolling(5).max() - l.rolling(5).min()).tail(60) / (base_rng + 1e-9)).dropna()
        fh    = pd.Series([
            max(0.0, 1.0 - (h.iloc[max(0,i-7):i].max() - h.iloc[max(0,i-7):i].min()) / (atr_v + 1e-9))
            for i in range(max(8, n - 60), n)
        ], dtype=float).dropna()
        tv = float(th.var()) if len(th) >= 5 else 0.5
        fv = float(fh.var()) if len(fh) >= 5 else 0.5
        tot = tv + fv + 1e-9
        coil_sc = tight * (tv / tot) + flat * (fv / tot)
        base_pos = (ltp - base_lo) / (base_rng + 1e-9)
        bph = ((c - l.rolling(_sw).min()) /
               (h.rolling(_sw).max() - l.rolling(_sw).min() + 1e-9)).dropna().tail(60)
        bc  = float(bph.quantile(0.75)) if len(bph) >= 10 else 0.80
        bs  = max(float((bph.quantile(0.90) - bph.quantile(0.60)) / 1.35), 0.05) if len(bph) >= 10 else 0.10
        bp_bonus = 0.2 / (1.0 + np.exp(-(1.0 / bs) * (base_pos - bc)))
        coil_sc  = min(1.0, coil_sc + float(bp_bonus))
    else:
        psw_hi = float(h.tail(_sw).max()); psw_lo = float(l.tail(_sw).min())
        pm     = psw_hi - psw_lo + 1e-9
        pb_dep = (psw_hi - float(c.iloc[-1])) / pm
        if n >= 40:
            pbs = [((h.iloc[max(0,i-_sw):i].max() - c.iloc[i]) /
                    (h.iloc[max(0,i-_sw):i].max() - l.iloc[max(0,i-_sw):i].min() + 1e-9))
                   for i in range(_sw, min(n, _sw + 120))]
            pb_arr = np.array([x for x in pbs if 0.0 <= x <= 1.0])
            if len(pb_arr) >= 10:
                pb_c = float(np.median(pb_arr))
                q75p, q25p = np.percentile(pb_arr, 75), np.percentile(pb_arr, 25)
                pb_s = max((q75p - q25p) / 1.35, 0.05)
            else:
                pb_c, pb_s = 0.382, 0.125
        else:
            pb_c, pb_s = 0.382, 0.125
        coil_sc  = float(np.clip(np.exp(-0.5 * ((pb_dep - pb_c) / pb_s) ** 2), 0.0, 1.0))
        base_pos = (ltp - base_lo) / (base_rng + 1e-9)

    # ── Proximity score ───────────────────────────────────────────
    reg_lam = param_registry.get("prox_lambda", [])
    if n >= 30:
        dh = ((h.rolling(20).max().shift(1) - c) / (atr.iloc[:-1] + 1e-9)).dropna()
        dh = dh[dh > 0].tail(60)
        med_d = float(dh.median()) if len(dh) >= 10 else 0.7
        prox_lam = float(np.log(2) / max(med_d, 0.01))
        reg_lam.append(prox_lam)
        if len(reg_lam) >= 10:
            lam_lo, lam_hi = float(np.percentile(reg_lam, 5)), float(np.percentile(reg_lam, 95))
        else:
            lam_lo, lam_hi = 0.5, 3.0
        prox_lam = float(np.clip(prox_lam, lam_lo, lam_hi))
    else:
        prox_lam = 1.0

    if setup_type == "Breakout":
        d_trig = (base_hi - ltp) / (atr_v + 1e-9)
        r_ser  = h.rolling(20).max().shift(1)
        n_bo   = min(n, len(r_ser.dropna()), len(atr) - 1)
        hc_bo  = c.values[-n_bo:].astype(float)
        rs_bo  = r_ser.values[-n_bo:].astype(float)
        at_bo  = atr.values[-(n_bo + 1):-1].astype(float)
        valid  = ~np.isnan(rs_bo)
        below_ = valid & (hc_bo < rs_bo)
        above_ = np.zeros(len(hc_bo), dtype=bool)
        above_[:-1] = valid[:-1] & (hc_bo[1:] > rs_bo[:-1])
        bo_en  = below_ & above_
        dist_  = np.where(valid & (at_bo > 0), (rs_bo - hc_bo) / at_bo, np.nan)
        if bo_en.sum() >= 3:
            d_ = dist_[bo_en]; d_ = d_[(d_ > 0) & ~np.isnan(d_)]
            ideal_d = float(np.median(d_)) if len(d_) >= 2 else 1.0 / prox_lam
        else:
            d_ = dist_[(dist_ > 0) & ~np.isnan(dist_)]
            ideal_d = float(np.median(d_)) if len(d_) >= 5 else 1.0 / prox_lam
        d_adj   = abs(d_trig - ideal_d)
        if d_trig < 0:
            d_adj += abs(d_trig)
        prox_sc = max(0.0, min(1.0, float(np.exp(-prox_lam * d_adj))))
    else:
        de9  = ltp - e9_v;  de20 = ltp - e20_v
        cd   = de9 if abs(de9) <= abs(de20) else de20
        nc   = min(n, len(e9), len(e20), len(atr) - 1)
        hca  = c.values[-nc:].astype(float)
        e9a  = e9.values[-nc:].astype(float)
        e20a = e20.values[-nc:].astype(float)
        atra = atr.values[-(nc + 1):-1].astype(float)
        crm  = np.zeros(len(e9a), dtype=bool)
        if len(e9a) >= 2:
            crm[1:] = (e9a[1:] > e20a[1:]) & (e9a[:-1] <= e20a[:-1])
        pcm  = np.zeros(len(e9a), dtype=bool)
        if crm.sum() >= 3:
            ci   = np.where(crm)[0]; pi = ci[ci > 0] - 1; pcm[pi] = True
        dist_a = (hca - e20a) / (atra + 1e-9)
        if pcm.sum() >= 2:
            pbd  = dist_a[pcm]; pbd = pbd[pbd > 0]
            id_pb = float(np.median(pbd)) if len(pbd) >= 2 else 1.0 / prox_lam
        else:
            pd_  = dist_a[dist_a > 0]
            id_pb = float(np.median(pd_)) if len(pd_) >= 5 else 1.0 / prox_lam
        d_fi  = cd / (atr_v + 1e-9) - id_pb
        pd_sc = abs(d_fi) if d_fi >= 0 else abs(d_fi) * 1.5
        prox_sc = max(0.0, min(1.0, float(np.exp(-prox_lam * pd_sc))))

    # ── VCP / Darvas ──────────────────────────────────────────────
    vcp_r  = detect_vcp(c, h, l, v, atr)
    _vcp_sc = feat["vcp_cs"] if feat.get("vcp_cs") is not None else vcp_r["vcp_score"]
    darvas_r = darvas_box_score(df.iloc[:-1] if len(df) > 1 else df, atr_v)
    d_bh     = darvas_r.get("box_high", np.nan)
    d_bl     = darvas_r.get("box_low",  np.nan)
    if not (math.isnan(float(d_bh)) if isinstance(d_bh, float) else np.isnan(d_bh)):
        dw  = float(d_bh) - float(d_bl) + 1e-9
        dp  = (ltp - float(d_bl)) / dw
        dp_sc = float(np.clip(dp, 0.0, 1.0)) if setup_type == "Breakout" \
                else float(np.clip(1.0 - dp, 0.0, 1.0))
        d_ar  = float(darvas_r.get("box_atr_ratio", 1.0) or 1.0)
        if math.isnan(d_ar):
            d_ar = 1.0
        d_tight = 1.0 / (1.0 + d_ar)
        d_time  = float(np.clip(darvas_r.get("bars_in_box", 0) / 20.0, 0.0, 1.0))
        darvas_sc = d_tight * 0.40 + dp_sc * 0.35 + d_time * 0.25
    else:
        darvas_sc = darvas_r["darvas_score"] / 10.0

    # ── Sweep bonus (false-break reversal) ───────────────────────
    sweep_sc = 0.0
    if n >= 5:
        prior_sup   = float(l.tail(5).min())
        lower_wick  = min(day_o, ltp) - day_lo
        vz_hist     = ((v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)).tail(60)
        vz_p60      = float(vz_hist.quantile(0.60)) if len(vz_hist) >= 20 else 1.0
        if (day_lo < prior_sup and ltp > float(c.iloc[-1]) and
                lower_wick >= 0.5 * atr_v and vol_z >= vz_p60):
            wick_r   = lower_wick / (atr_v + 1e-9)
            sweep_sc = float(np.clip(np.tanh(wick_r), 0.0, 1.0))

    # ── VWAP position ─────────────────────────────────────────────
    vwap_sc = 0.0
    if n >= 20:
        tp       = (h + l + c) / 3
        cum_tv   = (tp * v).rolling(20).sum()
        cum_v20  = v.rolling(20).sum()
        vwma20   = float((cum_tv / cum_v20.replace(0, np.nan)).iloc[-1])
        if not math.isnan(vwma20):
            vwap_sc = 0.5 if ltp > vwma20 else 0.0
            if n >= 21:
                vm_prev = float((cum_tv / cum_v20.replace(0, np.nan)).iloc[-2])
                if not math.isnan(vm_prev) and vwma20 > vm_prev:
                    vwap_sc = min(1.0, vwap_sc + 0.25)

    # ── OI buildup ────────────────────────────────────────────────
    oi_sc = 0.0
    if "oi" in df.columns and n >= 10:
        oi = df["oi"].dropna(); oi_nz = oi[oi > 0]
        if len(oi_nz) >= 5:
            oi_now  = float(oi.iloc[-1])
            oi_5d   = float(oi.iloc[-6]) if n >= 6 else float(oi.iloc[0])
            oi_avg  = float(oi_nz.tail(20).mean())
            oi_std  = max(float(oi_nz.tail(20).std()), oi_avg * 0.05)
            oi_rising = oi_now > oi_5d
            oi_z = (oi_now - oi_avg) / (oi_std + 1e-9)
            vc_p30 = float((atr5 / atr20.replace(0, np.nan)).dropna().quantile(0.30)) if n >= 20 else 0.85
            if oi_rising and feat["vc_ratio_now"] < vc_p30 and oi_z > 0:
                cs  = 1.0 - min(feat["vc_ratio_now"], 1.0)
                oi_sc = float(np.clip(np.tanh(oi_z * cs), 0.0, 1.0))

    # ── Volume intraday velocity ──────────────────────────────────
    vol_velocity = 0.0
    if elapsed_mins >= 30 and elapsed_mins < session_mins:
        vr = day_vol / (feat["vol_mu"] + 1e-9)
        vol_velocity = float(np.clip(np.tanh(vr - 1.0), 0.0, 1.0))

    # ── ATR potential (inverted) ──────────────────────────────────
    atr_hist_pct = (_atr(df, 14).iloc[:-1] / c.iloc[:-1] * 100).tail(60).dropna()
    atr_pct_rank = float((atr_hist_pct <= feat["atr_pct"]).mean()) if len(atr_hist_pct) >= 10 else 0.5
    atp_sc       = 1.0 - atr_pct_rank

    # ── Candle patterns ───────────────────────────────────────────
    raw_cdl, cdl_names = detect_candle_patterns(
        day_o, day_hi, day_lo, ltp,
        float(df["open"].iloc[-2]), float(df["high"].iloc[-2]),
        float(df["low"].iloc[-2]),  float(df["close"].iloc[-2]),
    )
    cdl_sc = min(raw_cdl * 0.1, 0.5)   # normalise to 0-0.5

    # ── Stability adaptive adjustment ─────────────────────────────
    stab_adj = 0.0
    sp = feat["stab_pct"]
    if sp is not None and pd.notna(sp):
        reg_ss  = param_registry.get("stab_adj_scale", [])
        reg_so  = param_registry.get("stab_adj_obs",   [])
        stab_w  = max(float(np.std(list(reg_ss))) if len(reg_ss) >= 10 else 0.15, 0.05)
        stab_z  = (sp - 0.50) / stab_w
        sc_med  = float(np.median(reg_ss)) if len(reg_ss) >= 10 else 5.0
        cl_lo   = float(np.percentile(reg_so, 5))  if len(reg_so) >= 10 else -8.0
        cl_hi   = float(np.percentile(reg_so, 95)) if len(reg_so) >= 10 else  2.0
        stab_adj = float(np.clip(np.tanh(stab_z) * sc_med, cl_lo, cl_hi))
        reg_so.append(stab_adj)
        reg_ss.append(float(np.clip(abs(stab_z) * feat["atr_pct"] / 100.0, 1.0, 15.0)))
        param_registry["stab_adj_scale"] = reg_ss[-cfg.REG_HISTORY:]
        param_registry["stab_adj_obs"]   = reg_so[-cfg.REG_HISTORY:]
    else:
        stab_adj = 0.5 if feat["stability"] >= 0.60 else 0.0

    # ── VCVE (volume × volatility-contraction interaction) ────────
    ii = feat["inst_ratio"] * (1.0 - min(feat["vc_ratio_now"], 1.0))
    vcve_sc = float(np.clip(np.tanh(ii / max(feat["inst_ratio"] * 0.4, 0.1)), 0.0, 1.0))

    # ── Volume quietness (Pullback) ───────────────────────────────
    vr_hist = (v.iloc[:-1] / (v.iloc[:-1].rolling(20).mean() + 1e-9)).dropna()
    vr_now  = float(v.iloc[-1]) / (feat["vol_mu"] + 1e-9)
    vol_quiet_pct = float((vr_hist >= vr_now).mean()) if len(vr_hist) >= 10 else float(np.clip(1.0 - vr_now, 0.0, 1.0))

    return dict(
        setup_type=setup_type,
        # Primary signals (0-1)
        coil_sc=coil_sc, prox_sc=prox_sc,
        vcp_sc=_vcp_sc, darvas_sc=darvas_sc,
        sweep_sc=sweep_sc, vwap_sc=vwap_sc,
        oi_sc=oi_sc, vol_velocity=vol_velocity, atp_sc=atp_sc,
        cdl_sc=cdl_sc, stab_adj=stab_adj, vcve_sc=vcve_sc,
        vol_quiet_pct=vol_quiet_pct,
        # VCP detail
        vcp_detail=vcp_r,
        # Darvas detail
        darvas_detail=darvas_r,
        # Candle names
        cdl_names=cdl_names, raw_cdl=raw_cdl,
        # Structural
        base_pos=base_pos, breakout_ext=breakout_ext,
        t1_vol_ratio=t1_vol_ratio, t1_close_pos=t1_close_pos,
        washout_depth=washout_depth,
        vol_bo_thresh=vol_bo_thresh,
        ext_p10=ext_p10, ext_p90=ext_p90,
    )


def compute_penalties(feat: dict, sig: dict, market_ctx: dict,
                       ind: dict, df: pd.DataFrame,
                       ltp: float, rsi_period: int,
                       adv_threshold: float) -> dict:
    """Stage 4 — continuous penalties mapped to score-point deductions."""
    cfg = SCORE_CFG
    c, h, l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    atr = ind["atr"]
    atr_v    = feat["atr_v"];   atr_pct  = feat["atr_pct"]
    rsi_v    = feat["rsi_v"];   vol_mu   = feat["vol_mu"]
    sma200   = ind["sma200"]
    vol_ma20 = ind["vol_ma20"]
    n        = len(c)
    regime   = feat["regime"]

    penalties: Dict[str, float] = {}

    # ── RSI overbought (adaptive P90) ────────────────────────────
    rsi_all = _rsi_wilder(c, rsi_period)
    rsi_p90 = float(rsi_all.tail(60).quantile(0.90)) if len(rsi_all) >= 20 else 80.0
    if rsi_v > rsi_p90:
        z = (rsi_v - rsi_p90) / max(float(rsi_all.tail(20).std()), 1.0)
        penalties["rsi_ob"] = float(np.clip(8.0 * np.tanh(z), 0.0, cfg.RSI_OB_CAP))

    # ── Abnormally low volume ─────────────────────────────────────
    vol_p05 = float(v.tail(60).quantile(0.05)) if len(v) >= 20 else vol_mu * 0.10
    prev_v  = float(v.iloc[-1])
    if prev_v < vol_p05:
        vz = (vol_p05 - prev_v) / max(float(v.tail(20).std()), 1.0)
        penalties["vol_low"] = float(np.clip(6.0 * np.tanh(vz), 0.0, cfg.VOL_LOW_CAP))

    # ── Very low ATR (dead stock) ─────────────────────────────────
    atr_hist = (atr / c).tail(60).dropna()
    if len(atr_hist) >= 10:
        atr_p10 = float(atr_hist.quantile(0.10))
        if atr_pct / 100.0 < atr_p10:
            excess = atr_p10 - atr_pct / 100.0
            penalties["atr_low"] = float(np.clip(excess / max(atr_p10, 1e-9) * 5.0, 0.0, 5.0))

    # ── Below SMA200 ──────────────────────────────────────────────
    sma200_gap_atr = (ltp - sma200) / (atr_v + 1e-9)
    if sma200_gap_atr < -0.5:
        excess = abs(sma200_gap_atr + 0.5)
        penalties["sma200"] = float(np.clip(10.0 * np.tanh(excess / 2.0), 0.0, cfg.SMA_CAP))

    # ── Liquidity penalty ─────────────────────────────────────────
    liq = feat["liquidity_sc"]
    if liq < 0.5:
        penalties["liquidity"] = float(np.clip(cfg.LIQ_CAP * (1.0 - liq * 2.0), 0.0, cfg.LIQ_CAP))

    # ── Gap penalty ───────────────────────────────────────────────
    prev_c = float(c.iloc[-1]); prev_o = float(df["open"].iloc[-1]) if "open" in df.columns else prev_c
    if prev_c > 0 and atr_v > 0 and len(c) >= 2:
        gap = prev_o - float(c.iloc[-2])
        g_hist = ((h.shift(1) - c.shift(1)).abs() / (atr + 1e-9)).dropna().tail(60)
        g_p90  = float(g_hist.quantile(0.90)) if len(g_hist) >= 20 else 2.0
        g_atr  = abs(gap) / (atr_v + 1e-9)
        if g_atr > g_p90:
            penalties["gap"] = float(np.clip(8.0 * np.tanh((g_atr - g_p90) / (g_p90 + 1e-9)),
                                              0.0, cfg.GAP_CAP))

    # ── Already broken out (vol rank ≥ P85 at P90+ extension) ────
    setup = sig["setup_type"]
    if setup == "Breakout" and ltp >= feat["base_hi"] - 0.2 * atr_v:
        t1v     = float(v.iloc[-1])
        vrank   = float((v.iloc[:-1] <= t1v).mean())
        if vrank >= 0.85:
            z = (vrank - 0.85) / 0.15
            penalties["already_bo"] = float(np.clip(12.0 * np.tanh(z * 2.0), 0.0, cfg.ALREADY_BO_CAP))

    # ── Overextended breakout (above ext_p90) ─────────────────────
    if setup == "Breakout" and sig["breakout_ext"] > sig["ext_p90"]:
        excess = (sig["breakout_ext"] - sig["ext_p90"]) / max(sig["ext_p90"] - sig["ext_p10"], 0.5)
        penalties["overextended"] = float(np.clip(10.0 * np.tanh(excess), 0.0, cfg.EXT_CAP))

    # ── Stability kill (very choppy) ──────────────────────────────
    stab = feat["stability"]
    if stab < 0.20 and (feat["rsi_v"] < 40 or feat["rsi_v"] <= feat["rsi_p"]):
        penalties["stability"] = float(np.clip(15.0 * (0.20 - stab) / 0.20, 0.0, cfg.STAB_CAP))

    # ── VIX penalty (continuous z-score map) ──────────────────────
    vix_v   = market_ctx.get("vix_level")
    vix_med = market_ctx.get("vix_median", 14.5)
    vix_sig = market_ctx.get("vix_sigma",  4.5)
    vix_fall = market_ctx.get("vix_falling", True)
    if vix_v is not None:
        vix_z   = (vix_v - vix_med) / (vix_sig + 1e-9)
        vix_adj = float(np.clip(-6.0 * np.tanh(vix_z), cfg.VIX_PENALTY_FLOOR, cfg.VIX_BONUS_CAP))
        if not vix_fall:
            vix_adj = float(np.clip(vix_adj - 2.0 * abs(np.tanh(vix_z)),
                                     cfg.VIX_PENALTY_FLOOR, 0.0))
        penalties["vix"] = -vix_adj   # stored as positive penalty value
    elif not vix_fall:
        penalties["vix"] = 5.0

    # ── Nifty breadth penalty ─────────────────────────────────────
    breadth      = market_ctx.get("breadth_cache")
    breadth_hist = market_ctx.get("breadth_hist", [])
    cs_rs        = feat.get("rs_combined", 0.5)
    if breadth is not None and len(breadth_hist) >= 5:
        bmu  = float(np.mean(breadth_hist))
        bsig = max(float(np.std(breadth_hist)), 0.03)
        bz   = (breadth - bmu) / bsig
        raw_ba = float(np.clip(6.0 * np.tanh(bz), cfg.BREADTH_PENALTY_FLOOR, cfg.BREADTH_BONUS_CAP))
        if raw_ba < 0:
            ba = raw_ba * max(0.0, 1.0 - cs_rs)
        else:
            ba = raw_ba
        penalties["breadth"] = -ba
    elif market_ctx.get("nifty_r5") is not None:
        n5  = market_ctx.get("nifty_r5", 0.0) or 0.0
        n20 = market_ctx.get("nifty_r20", 0.0) or 0.0
        ba  = float(np.clip((n5 + n20 * 0.5) * 100, cfg.BREADTH_PENALTY_FLOOR, cfg.BREADTH_BONUS_CAP))
        penalties["breadth"] = -ba
    elif not market_ctx.get("nifty_above_20dma", True):
        penalties["breadth"] = 8.0

    # ── Regime penalty for Breakouts ──────────────────────────────
    if setup == "Breakout":
        bb_cs  = feat["bb_cs"] if feat["bb_cs"] is not None else 0.5
        vdu_cs = feat["vdu_cs"] if feat["vdu_cs"] is not None else 0.5
        nf     = (bb_cs + vdu_cs) / 2.0
        if regime == "BEAR":
            penalties["regime_bo"] = nf * 8.0
        elif regime == "CHOP":
            penalties["regime_bo"] = nf * 4.0

    # ── Sector in distribution (bottom quartile 5D return) ──────────
    # A breakout in a sector that is actively being distributed is
    # much lower probability.  We discount using sector-relative rank.
    sect = feat.get("sect_name")
    sect_rets = list(market_ctx.get("sector_returns", {}).values())
    if sect and len(sect_rets) >= 4:
        sr = market_ctx.get("sector_returns", {}).get(sect)
        if sr is not None:
            q25_sect = float(np.percentile(sect_rets, 25))
            if sr < q25_sect:
                depth = (q25_sect - sr) / (max(abs(q25_sect), 0.001) + 1e-9)
                penalties["sect_dist"] = float(np.clip(
                    cfg.SECT_DIST_PENALTY_CAP * np.tanh(depth * 3.0),
                    0.0, cfg.SECT_DIST_PENALTY_CAP
                ))

    # ── Universe breakout saturation guard ───────────────────────────
    # When >50% of the universe is already tagged Breakout, a new one
    # is less differentiated — the market is in a broad momentum phase
    # rather than stock-specific coiling.  Discount accordingly.
    if setup == "Breakout":
        bo_frac = STATE.get("_bo_saturation_frac", 0.0)
        if bo_frac > cfg.BO_SATURATION_FLOOR:
            excess = (bo_frac - cfg.BO_SATURATION_FLOOR) / (
                1.0 - cfg.BO_SATURATION_FLOOR + 1e-9
            )
            penalties["bo_saturation"] = float(np.clip(
                cfg.BO_SATURATION_DISCOUNT * 100.0 * excess,
                0.0,
                cfg.BO_SATURATION_DISCOUNT * 100.0
            ))

    return {"penalties": penalties,
            "total_penalty": sum(max(0.0, p) if k != "vix" and k != "breadth"
                                 else p for k, p in penalties.items())}


def aggregate_score(feat: dict, sig: dict, pen: dict,
                     market_ctx: dict, ind: dict,
                     cs_state: dict, df: pd.DataFrame,
                     ticker: str, ltp: float,
                     day_vol: float, day_hi: float, day_lo: float, day_o: float,
                     nifty_r5: Optional[float], nifty_r20: Optional[float],
                     rsi_period: int, elapsed_mins: int,
                     session_mins: int, vol_ma20: float,
                     param_registry: dict) -> dict:
    """Stage 5 — combine signals with coverage weighting, apply penalties,
    classify horizon, compute entry / target / stop."""
    cfg = SCORE_CFG
    setup  = sig["setup_type"]
    regime = feat["regime"]

    # ────────────────────────────────────────────────────────────
    # Signal strength × coverage  (prevents sparse-signal stocks)
    # ────────────────────────────────────────────────────────────
    c, h, l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    atr = ind["atr"]
    atr_v  = feat["atr_v"];   atr_pct = feat["atr_pct"]
    e9_v   = feat["e9_v"];    e20_v   = feat["e20_v"];   e50_v = feat["e50_v"]
    rsi_v  = feat["rsi_v"];   rsi_p   = feat["rsi_p"]
    vol_mu = feat["vol_mu"];  stability = feat["stability"]
    liquidity_sc = feat["liquidity_sc"]
    n = len(c)

    if setup == "Reversal":
        # Reversal uses its own sub-scoring
        rsi_all  = _rsi_wilder(c, rsi_period)
        rp90     = float(rsi_all.tail(60).quantile(0.90)) if len(rsi_all) >= 20 else 70.0
        rp10     = float(rsi_all.tail(60).quantile(0.10)) if len(rsi_all) >= 20 else 25.0
        r_range  = max(rp90 - rp10, 10.0)
        rev_rsi  = float(np.clip((rp90 - rsi_v) / r_range, 0.0, 1.0))
        rev_coil = float(np.clip(sig["coil_sc"], 0.0, 1.0))
        rev_prox = sig["prox_sc"]
        rev_spr  = sig["vol_quiet_pct"]
        rev_vol  = float((v.iloc[:-1] <= float(v.iloc[-1])).mean()) if n > 1 else 0.5
        rev_wash = float(np.clip((sig["washout_depth"] - 1.5) / 4.0, 0.0, 1.0))
        rev_tail = float(np.clip((sig["t1_close_pos"] - 0.30) / 0.70, 0.0, 1.0))
        sub_scores = [rev_rsi * 0.40, rev_coil * 0.30, rev_prox * 0.20,
                      rev_spr * 0.10, rev_vol * 0.05, rev_wash * 0.10, rev_tail * 0.05]
        coverage   = sum(1 for s in sub_scores if s > 0) / max(len(sub_scores), 1)
        signal_str = sum(sub_scores)
    else:
        # Breakout + Pullback unified pipeline
        vol_sig    = feat["vol_signal"] if setup == "Breakout" else (1.0 - feat["vol_signal"])
        raw_sigs   = {
            "rs":        feat["rs_combined"],
            "rs_sect":   feat["rs_sect"],
            "momentum":  feat["acc_rank"],
            "volume":    vol_sig,
            "coil":      sig["coil_sc"],
            "ma":        feat["ma_feature"],
            "proximity": sig["prox_sc"],
            "vcp":       sig["vcp_sc"],
            "darvas":    sig["darvas_sc"],
            "micro":     float(np.mean([
                feat["bb_cs"]  if feat["bb_cs"]  is not None else 0.5,
                feat["vdu_cs"] if feat["vdu_cs"] is not None else 0.5,
                feat["clv_cs"] if feat["clv_cs"] is not None else 0.5,
                feat["sc_feature"],
            ])),
        }
        weights = {
            "rs":        cfg.W_RS,     "rs_sect":   cfg.W_RS_SECT,
            "momentum":  cfg.W_MOMENTUM, "volume": cfg.W_VOLUME,
            "coil":      cfg.W_COIL,   "ma":       cfg.W_MA,
            "proximity": cfg.W_PROXIMITY, "vcp":   cfg.W_VCP,
            "darvas":    cfg.W_DARVAS, "micro":     cfg.W_MICROSTRUCTURE,
        }
        valid_sigs   = {k: v for k, v in raw_sigs.items() if pd.notna(v)}
        coverage     = len(valid_sigs) / max(len(raw_sigs), 1)
        wt_sum       = sum(weights[k] for k in valid_sigs)
        signal_str   = sum(raw_sigs[k] * weights[k] for k in valid_sigs) / (wt_sum + 1e-9)

        # ── Interaction term: RS × Volume × Proximity compounding ──────
        # A simple weighted sum treats these as independent.  When all three
        # fire strongly together the actual edge is multiplicative, not additive.
        # Only activates when all three exceed INTERACTION_FLOOR (default 0.60)
        # to avoid rewarding marginal co-occurrence.
        _fl  = cfg.INTERACTION_FLOOR
        _rs  = raw_sigs.get("rs", 0.0)
        _vol = raw_sigs.get("volume", 0.0)
        _prx = raw_sigs.get("proximity", 0.0)
        if _rs > _fl and _vol > _fl and _prx > _fl:
            _norm = (1.0 - _fl) ** 3
            _interaction = ((_rs - _fl) * (_vol - _fl) * (_prx - _fl)) / (_norm + 1e-9)
            signal_str *= (1.0 + float(np.clip(_interaction, 0.0, cfg.INTERACTION_BOOST_MAX)))

    # Apply coverage floor
    coverage = max(coverage, cfg.COVERAGE_FLOOR)
    base_score = signal_str * coverage * 100.0

    # ── Bonuses ───────────────────────────────────────────────────
    bonus_raw = (
        sig["sweep_sc"]    * cfg.SWEEP_CAP    +
        sig["oi_sc"]       * cfg.OI_CAP       +
        feat["uv_feature"] * cfg.UV_CAP       +
        feat["cpr_feature"]* cfg.CPR_CAP      +
        feat["sc_feature"] * cfg.SC_CAP       +
        feat["atr_exp_feature"] * cfg.ATR_EXP_CAP +
        sig["vcve_sc"]     * cfg.VCVE_CAP
    )
    # Persistence multiplier (compression in last 3 bars)
    persist = 1.0
    vc_ser  = (ind["atr5"] / ind["atr20"].replace(0, np.nan)).dropna()
    if n >= 6 and len(vc_ser) >= 10:
        vc3    = vc_ser.iloc[-4:-1].dropna()
        vc_p40 = float(vc_ser.quantile(0.40))
        persist = float(np.clip(0.5 + int((vc3 < vc_p40).sum()) * 0.25, 0.5, 1.0))
    bonus_raw *= persist
    bonuses   = float(np.clip(bonus_raw, 0.0, cfg.BONUS_CAP))

    total = base_score + bonuses

    # ── Apply all penalties ───────────────────────────────────────
    for k, p_val in pen["penalties"].items():
        total -= p_val    # penalties dict stores raw signed contributions
    total = float(np.clip(total, 0.0, 100.0))

    emi = round(total * atr_pct / 100, 3)
    bb_n  = feat["bb_cs"]  if feat["bb_cs"]  is not None else 0.5
    vdu_n = feat["vdu_cs"] if feat["vdu_cs"] is not None else 0.5
    clv_n = feat["clv_cs"] if feat["clv_cs"] is not None else 0.5
    vcp_n = feat["vcp_cs"] if feat["vcp_cs"] is not None else sig["vcp_detail"]["vcp_score"]
    vc_n  = feat["vc_feature"]
    breakout_prob = float(np.mean([bb_n, vdu_n, clv_n, vcp_n, vc_n]))
    composite_rank = round(emi * 0.70 + liquidity_sc * 0.20 + min(stability, 1.0) * 0.10, 4)

    # ── Horizon classification ────────────────────────────────────
    _atr_cv   = float(atr.iloc[-20:].std() / (atr.iloc[-20:].mean() + 1e-9)) if n >= 20 else 0.3
    _cv_scale = float(np.clip(1.0 + _atr_cv, 0.7, 1.5))
    _tgt_mult = {
        "Imminent BO": round(0.6 * _cv_scale, 2),
        "Intraday":    round(0.6 * _cv_scale, 2),
        "Swing 2-5D":  round(1.8 * _cv_scale, 2),
        "Mid 5-14D":   round(3.2 * _cv_scale, 2),
        "Long 14-30D": round(4.5 * _cv_scale, 2),
    }

    rsi_all = _rsi_wilder(c, rsi_period)
    if n >= 40:
        bo_dh = ((h.rolling(20).max() - c) / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
        p20_bo = float(np.percentile(bo_dh, 20)) if len(bo_dh) >= 10 else 0.25
        p50_bo = float(np.percentile(bo_dh, 50)) if len(bo_dh) >= 10 else 1.0
        p80_bo = float(np.percentile(bo_dh, 80)) if len(bo_dh) >= 10 else 3.0
        pb_dh  = ((ind["e20"].iloc[:-1] - c) / (atr.iloc[:-1] + 1e-9)).clip(0).dropna().tail(60)
        p20_pb = float(np.percentile(pb_dh, 20)) if len(pb_dh) >= 10 else 0.3
        p50_pb = float(np.percentile(pb_dh, 50)) if len(pb_dh) >= 10 else 1.0
        p80_pb = float(np.percentile(pb_dh, 80)) if len(pb_dh) >= 10 else 2.5
    else:
        p20_bo, p50_bo, p80_bo = 0.25, 1.0, 3.0
        p20_pb, p50_pb, p80_pb = 0.30, 1.0, 2.5

    base_hi  = feat["base_hi"];   base_lo = feat["base_lo"]
    vol_ratio = feat["vol_ratio"]; t1_vr  = sig["t1_vol_ratio"]
    t1_cp    = sig["t1_close_pos"]; cdl_n  = sig["cdl_names"]
    washout  = sig["washout_depth"]

    vol_bo_t = sig["vol_bo_thresh"]
    rsi_p60  = float(rsi_all.tail(60).quantile(0.60)) if len(rsi_all) >= 20 else 55.0

    if setup == "Breakout":
        d_bo = (base_hi - ltp) / (atr_v + 1e-9)
        if d_bo <= p20_bo and day_vol >= vol_bo_t:
            horizon = "Imminent BO"
            hz_note = f"AT TRIGGER — vol {vol_ratio:.1f}× avg. Enter now or market open."
        elif d_bo <= 0.0 and vol_ratio >= 1.5 and rsi_v < rsi_p60:
            horizon = "Intraday"
            hz_note = f"Breaking today — RSI {rsi_v:.0f}, vol {vol_ratio:.1f}×. Trail stop above base low."
        elif d_bo <= p20_bo:
            horizon = "Swing 2-5D"
            hz_note = f"{d_bo:.2f} ATR from trigger. Place limit above {base_hi:.1f}."
        elif d_bo <= p50_bo:
            horizon = "Mid 5-14D"
            hz_note = f"{d_bo:.2f} ATR from trigger. Coiling — alert for vol expansion."
        elif d_bo <= p80_bo:
            horizon = "Long 14-30D"
            hz_note = f"{d_bo:.2f} ATR from trigger. Base building — add to watchlist."
        else:
            horizon = "Long 14-30D"
            hz_note = f"{d_bo:.2f} ATR from trigger. Base still forming."
    elif setup == "Reversal":
        rsi_turn = rsi_v > rsi_p
        if rsi_turn and t1_cp >= 0.60 and sig["raw_cdl"] >= 1:
            horizon = "Intraday"
            hz_note = (f"Capitulation bottom confirmed — RSI {rsi_v:.0f} turning, "
                       f"vol {t1_vr:.1f}× avg. "
                       f"Pattern: {', '.join(cdl_n) if cdl_n else 'hammer/wick'}.")
        elif rsi_turn:
            horizon = "Swing 2-5D"
            hz_note = f"Washout in progress — RSI {rsi_v:.0f}. Enter on next green candle."
        else:
            horizon = "Mid 5-14D"
            hz_note = f"Panic selling — RSI {rsi_v:.0f}. Wait for RSI tick-up + confirmation."
    else:
        rsi_turn  = rsi_v > rsi_p
        pb_d_atr  = (e20_v - ltp) / (atr_v + 1e-9)
        if pb_d_atr <= p20_pb and rsi_turn and vol_ratio <= 0.8:
            horizon = "Intraday"
            hz_note = f"EMA20 support + RSI turning ({rsi_v:.0f}↑). Vol dry. Buy near {e20_v:.1f}."
        elif pb_d_atr <= p20_pb and rsi_turn and sig["raw_cdl"] >= 2:
            horizon = "Imminent BO"
            hz_note = f"Reversal candle at EMA. RSI {rsi_v:.0f}↑, pattern: {', '.join(cdl_n) or 'none'}."
        elif pb_d_atr <= p50_pb and rsi_v >= float(rsi_all.tail(60).quantile(0.35)):
            horizon = "Swing 2-5D"
            hz_note = f"Approaching EMA20. RSI {rsi_v:.0f}. Wait for reversal candle + vol."
        elif pb_d_atr <= p80_pb:
            horizon = "Mid 5-14D"
            hz_note = f"Pullback deepening ({pb_d_atr:.1f} ATR below EMA20). Do not enter yet."
        else:
            horizon = "Long 14-30D"
            hz_note = f"Extended correction ({pb_d_atr:.1f} ATR below EMA20). Watch for base."

    tgt_mult = _tgt_mult.get(horizon, round(1.8 * _cv_scale, 2))

    # ── Entry / Target / Stop ─────────────────────────────────────
    vc_ratio = feat["vc_ratio_now"]
    if setup == "Breakout":
        _buf   = atr_v * 0.1 * max(0.5, vc_ratio)
        entry  = round(base_hi + _buf, 2) if ltp < base_hi else round(ltp, 2)
        en_note = (f"Buy above {entry:.2f}" if ltp < base_hi
                   else f"Breaking now — buy on close above {base_hi:.2f}")
        tgt   = round(entry + tgt_mult * atr_v, 2)
        stp   = round(base_lo - atr_v * max(0.3, min(0.7, vc_ratio)), 2)
    elif setup == "Reversal":
        entry  = round(ltp, 2)
        en_note = f"Buy at open — reversal. RSI {rsi_v:.0f}. Stop below {float(l.iloc[-1]):.2f}"
        stp   = round(float(l.iloc[-1]) - 0.25 * atr_v, 2)
        tgt   = max(round(e20_v, 2), round(entry + 1.5 * atr_v, 2))
    else:
        entry  = round(ltp, 2)
        en_note = f"Buy near EMA20 ({e20_v:.2f}) on reversal candle"
        tgt_s  = round(float(h.tail(20).max()) * 0.997, 2)
        tgt    = max(tgt_s, round(entry + tgt_mult * atr_v, 2))
        stp    = round(e50_v - atr_v, 2)

    risk_raw   = max(entry - stp,  0.01)
    reward_raw = max(tgt   - entry, 0.01)
    rr         = round(reward_raw / risk_raw, 2)
    move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0

    # ── Win-rate prior: observed DB data takes precedence over formula ──
    # 1. Per-stock observed win rate (≥3 outcomes in calibration DB)
    # 2. Per-setup observed win rate (≥5 outcomes)
    # 3. Formula fallback based on RS + stability
    _wr_stock = STATE["per_stock_winrate"].get(ticker)
    _wr_setup = STATE.get("_setup_winrate", {}).get(setup)
    if _wr_stock is not None:
        wr_prior = float(_wr_stock)
    elif _wr_setup is not None:
        # Blend setup prior with formula — the formula is a decent prior
        # when we have setup-level data but not yet stock-level data.
        _formula = float(np.clip(0.40 + 0.20 * feat["rs_combined"] + 0.10 * stability, 0.30, 0.70))
        wr_prior = float(cfg.CALIB_ADAPT_ALPHA * _wr_setup + (1.0 - cfg.CALIB_ADAPT_ALPHA) * _formula)
    else:
        wr_prior = float(np.clip(0.40 + 0.20 * feat["rs_combined"] + 0.10 * stability, 0.30, 0.70))
    kelly    = round(float(np.clip(
        0.5 * (wr_prior * max(rr, 0.5) - (1.0 - wr_prior)) / (max(rr, 0.5) + 1e-9), 0.0, 0.25)), 3)

    # ── Reconstruct per-component scores (pts for explain endpoint) ─
    rs_pts      = round(feat["rs_combined"] * 15, 1)
    rs_sect_pts = round(feat["rs_sect"] * 10, 1)
    vol_pts     = round(feat["vol_signal"] * 15, 1) if setup == "Breakout" \
                  else round((1.0 - feat["vol_signal"]) * 15, 1)
    coil_pts    = round(sig["coil_sc"] * 10, 1)
    ma_pts      = round(feat["ma_feature"] * 10, 1)
    prox_pts    = round(sig["prox_sc"] * 10, 1)
    vcp_pts     = round(sig["vcp_sc"] * 10, 1)
    bb_pts      = round(float(feat["bb_cs"]) * 8, 1) if feat["bb_cs"] is not None else 4.0
    vdu_pts     = round(float(feat["vdu_cs"]) * 8, 1) if feat["vdu_cs"] is not None else 4.0
    clv_pts     = round(float(feat["clv_cs"]) * 8, 1) if feat["clv_cs"] is not None else 4.0
    darvas_pts  = round(sig["darvas_sc"] * 10, 1)
    spread_pts  = round(feat["sc_feature"] * 11, 1)
    vc_pts      = round(feat["vc_feature"] * 5 + feat["rci_feature"] * 5, 1)
    vc_quiet    = round(sig["vol_quiet_pct"] * 14, 1)
    inst_pts    = round(feat["inst_feature"] * 10, 1)
    atp_pts     = round(sig["atp_sc"] * 5, 1)
    cdl_pts     = round(sig["cdl_sc"] * 10, 1)

    total_pen = sum(max(0.0, p) for p in pen["penalties"].values())

    return {
        # ── Core ──────────────────────────────────────────────────
        "SetupType":   setup, "Score": round(total, 1),
        "EMI": emi, "CompositeRank": composite_rank,
        "Horizon": horizon, "HorizonNote": hz_note,
        "Entry": entry, "Target": tgt, "Stop": stp,
        "Risk": round(risk_raw, 1), "Reward": round(reward_raw, 1),
        "RR": rr, "KellyFrac": kelly, "Move%": move_pct, "EntryNote": en_note,
        # ── Component scores ──────────────────────────────────────
        "RS": rs_pts, "RS_Sector": rs_sect_pts,
        "Volume": vol_pts, "InstVol": inst_pts,
        "VolCont": vc_pts, "RCI": round(feat["rci_val"], 3),
        "VolQuiet": vc_quiet, "SpreadPts": spread_pts,
        "VolDryUp": vdu_pts, "CLVAccum": clv_pts,
        "BBSqueeze": bb_pts, "VCP": vcp_pts,
        "Coil": coil_pts, "MA_Struct": ma_pts, "Proximity": prox_pts,
        "ATR_Pot": atp_pts, "Candle": cdl_pts,
        "Darvas": darvas_pts,
        # ── Signal Persistence ────────────────────────────────────
        "BreakoutProb": round(breakout_prob, 3),
        "SignalPersist": round(coverage, 2),
        # ── VCP detail ────────────────────────────────────────────
        "VCP_Detected":    sig["vcp_detail"]["vcp_detected"],
        "VCP_Pullbacks":   sig["vcp_detail"]["vcp_pullback_n"],
        "VCP_Contraction": round(sig["vcp_detail"]["vcp_contraction"], 3),
        "VCP_VolComp":     round(sig["vcp_detail"]["vcp_vol_comp"],    3),
        "VCP_VolDryup":    round(sig["vcp_detail"]["vcp_vol_dryup"],   3),
        "VCP_Tightness":   round(sig["vcp_detail"]["vcp_tightness"],   3),
        "VCP_Position":    round(sig["vcp_detail"]["vcp_position"],    3),
        # ── Darvas detail ─────────────────────────────────────────
        "DarvasBox":    sig["darvas_detail"].get("box_high", None),
        "DarvasLow":    sig["darvas_detail"].get("box_low",  None),
        "DarvasInBox":  sig["darvas_detail"].get("in_box",   False),
        # ── Extras ────────────────────────────────────────────────
        "Patterns":      ", ".join(sig["cdl_names"]) if sig["cdl_names"] else "—",
        "RS_Accel":      round(feat["acc_sc"] * 4, 4),
        "AccelScore":    round(feat["acc_sc"] * 100, 1),
        "VCVE":          round(sig["vcve_sc"], 3),
        "BasePos":       round(base_hi and (ltp - feat["base_lo"]) / (feat["base_rng"] + 1e-9) or 0.0, 3),
        "Pos52W":        round(feat["pos52w"], 3),
        "Stability":     round(stability, 2),
        "Sweep":         sig["sweep_sc"] > 0,
        "VWMA20_OK":     sig.get("vwap_sc", 0) > 0,
        "DarvasBO":      0.0,
        "UpVolSkew":     round(feat["uv_feature"] * cfg.UV_CAP, 1),
        "CPR":           round(feat["cpr_feature"] * cfg.CPR_CAP, 1),
        "SpreadComp":    round(feat["sc_feature"] * cfg.SC_CAP, 1),
        "ATRExpOnset":   round(feat["atr_exp_feature"] * cfg.ATR_EXP_CAP, 1),
        "OI_Buildup":    round(sig.get("oi_sc", 0) * cfg.OI_CAP, 1),
        "VolVelocity":   round(sig.get("vol_velocity", 0) * 3, 1),
        "RSDivergence":  round(feat["rs_div_pct"] * 3, 1),
        "CSRank5d":      round(feat["rs_combined"], 3),
        "AbsRS":         round(feat["rs_combined"], 3),
        "RSI7":          round(feat["rsi_v"], 1),
        "VolRatio":      round(feat["vol_ratio"], 2),
        "VolZ":          round(feat["vol_z"], 2),
        "VolBOThr":      round(sig["vol_bo_thresh"] / max(feat["vol_mu"], 1.0), 2),
        "InstRatio":     round(feat["inst_ratio"], 2),
        "VC_Ratio":      round(feat["vc_ratio_now"], 2),
        "ATR%":          round(feat["atr_pct"], 2),
        "RS_vs_Nifty":   round(feat["rs_combined"] * 100, 1),
        "BO_Ext_ATR":    round(feat["breakout_ext"], 2),
        "Sector":        feat["sect_name"],
        "EMA9": round(feat["e9_v"], 2), "EMA20": round(feat["e20_v"], 2),
        "EMA50": round(feat["e50_v"], 2),
        "LiquidityScore": round(liquidity_sc, 3),
        "SoftPenalty":    round(total_pen, 1),
        "ADVTurnover":    round(feat["adv_turnover"] / 1e7, 2),
        "AboveSMA200":    ltp > ind["sma200"] - 0.5 * feat["atr_v"],
        # Reversal sub-scores (set to 0 for non-reversals)
        "Rev_RSI_Pts":    0.0, "Rev_Vol_Pts":     0.0,
        "Rev_Wash_Pts":   0.0, "Rev_Tail_Pts":    0.0,
        "Rev_Support_Pts":0.0,
        "WashoutDepth":   round(sig["washout_depth"], 2),
        "CandleTailPos":  round(sig["t1_close_pos"], 3),
        # Raw component values (for explain)
        "RS_raw":    rs_pts,  "Sect_raw":  rs_sect_pts,
        "Vol_raw":   vol_pts, "Inst_raw":  inst_pts,
        "VC_raw":    vc_pts,  "Coil_raw":  coil_pts,
        "MA_raw":    ma_pts,  "Prox_raw":  prox_pts,
        # Debug / decomposition
        "component_scores": {
            "rs": rs_pts, "rs_sect": rs_sect_pts, "momentum": round(feat["acc_rank"] * 10, 1),
            "volume": vol_pts, "coil": coil_pts, "ma": ma_pts,
            "proximity": prox_pts, "vcp": vcp_pts, "darvas": darvas_pts,
        },
        "signals": {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in sig.items()
                    if isinstance(v, (int, float, bool))},
        "penalties": {k: round(v, 2) for k, v in pen["penalties"].items()},
    }


# ═════════════════════════════════════════════════════════════════
# 11.  MAIN SCORING ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def score_stock_dual(ticker: str, df: pd.DataFrame,
                      live: dict,
                      nifty_r5: Optional[float],
                      nifty_r20: Optional[float]) -> Optional[dict]:
    """Top-level scorer: patches live bar, runs 5-stage pipeline."""
    cfg = SCORE_CFG
    if len(df) < cfg.MIN_BARS:
        return None

    df = df.copy()
    _ltp  = live.get("ltp");    ltp    = float(_ltp  if _ltp  is not None else df["close"].iloc[-1])
    _vol  = live.get("volume"); day_vol= float(_vol  if _vol  is not None else df["volume"].iloc[-1])
    _hi   = live.get("high");   day_hi = float(_hi   if _hi   is not None else df["high"].iloc[-1])
    _lo   = live.get("low");    day_lo = float(_lo   if _lo   is not None else df["low"].iloc[-1])
    _o    = live.get("open");   day_o  = float(_o    if _o    is not None else df["open"].iloc[-1])

    df.at[df.index[-1], "close"] = ltp
    df.at[df.index[-1], "high"]  = max(float(df["high"].iloc[-1]),  day_hi)
    df.at[df.index[-1], "low"]   = min(float(df["low"].iloc[-1]),   day_lo)

    # Intraday volume scaling
    _NSE_OPEN  = cfg.NSE_OPEN_MIN
    _NSE_CLOSE = cfg.NSE_CLOSE_MIN
    _SESS_MINS = _NSE_CLOSE - _NSE_OPEN
    _now_ist   = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    _elapsed   = _now_ist.hour * 60 + _now_ist.minute - _NSE_OPEN
    _el_frac   = float(np.clip(_elapsed / _SESS_MINS, 0.10, 1.0))
    if 6 <= _elapsed < _SESS_MINS:
        day_vol_sc = day_vol / _el_frac
    else:
        day_vol_sc = day_vol
    df.at[df.index[-1], "volume"] = day_vol_sc

    rsi_period  = int(STATE.get("rsi_period", 7))
    mkt         = STATE.get("mkt", {})
    cs_state    = {
        "cs_rs_5d":     STATE.get("cs_rs_5d",     {}),
        "cs_rs_20d":    STATE.get("cs_rs_20d",     {}),
        "cs_bb_squeeze":STATE.get("cs_bb_squeeze", {}),
        "cs_vol_dryup": STATE.get("cs_vol_dryup",  {}),
        "cs_clv_accum": STATE.get("cs_clv_accum",  {}),
        "cs_vcp":       STATE.get("cs_vcp",        {}),
        "rs_div_hist":  STATE.get("rs_div_hist",   {}),
        "breadth_cache":STATE.get("breadth_cache"),
        "breadth_hist": STATE.get("breadth_hist",  []),
    }
    # Add market context into cs_state for convenience
    cs_state.update({
        "sector_returns":      STATE.get("sector_returns", {}),
        "sector_returns_10d":  STATE.get("sector_returns_10d", {}),
    })
    param_reg   = STATE.setdefault("param_registry", {
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [],
        "stab_adj_scale": [], "stab_adj_obs": [], "pos52w_max": [],
    })

    # ── Build market_ctx to pass through pipeline ──────────────────
    market_ctx = dict(mkt)
    market_ctx["sector_returns"]     = STATE.get("sector_returns", {})
    market_ctx["sector_returns_10d"] = STATE.get("sector_returns_10d", {})
    market_ctx["breadth_cache"]      = STATE.get("breadth_cache")
    market_ctx["breadth_hist"]       = STATE.get("breadth_hist", [])

    vol_ma20 = float(df["volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df["volume"].mean())

    try:
        # Stage 1
        ind = compute_indicators(df, rsi_period)
        # Stage 2
        feat = compute_features(
            ind, ticker, ltp, day_vol_sc,
            day_hi, day_lo, day_o,
            nifty_r5, nifty_r20,
            market_ctx, cs_state, param_reg
        )
        # Validate
        if feat["atr_v"] <= 0 or vol_ma20 <= 0 or ltp <= 0:
            return None
        # Stage 3
        sig  = compute_signals(
            feat, ind, df, ticker, ltp,
            day_vol_sc, day_hi, day_lo, day_o,
            nifty_r5, nifty_r20,
            cs_state, param_reg,
            rsi_period, _el_frac, vol_ma20,
            _SESS_MINS, _elapsed
        )
        # Stage 4
        pen  = compute_penalties(
            feat, sig, market_ctx, ind, df,
            ltp, rsi_period, cfg.ADV_THRESHOLD
        )
        # Stage 5
        result = aggregate_score(
            feat, sig, pen, market_ctx, ind,
            cs_state, df, ticker, ltp,
            day_vol_sc, day_hi, day_lo, day_o,
            nifty_r5, nifty_r20,
            rsi_period, _elapsed, _SESS_MINS,
            vol_ma20, param_reg
        )

        # Update rs_div_hist — isolated side-effect, always guarded by the lock.
        cs5  = cs_state["cs_rs_5d"].get(ticker)
        cs20 = cs_state["cs_rs_20d"].get(ticker)
        if cs5 is not None and cs20 is not None:
            with STATE_LOCK:
                _prev = STATE.setdefault("rs_div_hist", {}).get(ticker, [])
                STATE["rs_div_hist"][ticker] = (_prev + [cs5 - cs20])[-60:]

        # Clean NaN/inf before returning
        clean: dict = {}
        for k, v in result.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        return clean

    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════
# 12.  LIVE QUOTE FETCH + BAR PATCHING
# ═════════════════════════════════════════════════════════════════

def fetch_live_quotes(all_keys: List[str]) -> dict:
    url  = "https://api.upstox.com/v2/market-quote/quotes"
    out  = {}
    chunk = SCORE_CFG.LIVE_QUOTE_CHUNK
    hdr   = get_headers()
    for i in range(0, len(all_keys), chunk):
        batch  = all_keys[i:i + chunk]
        params = {"instrument_key": ",".join(batch)}
        try:
            r = requests.get(url, headers=hdr, params=params, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json().get("data", {})
            for _, v in data.items():
                ikey = v.get("instrument_token", "")
                if not ikey:
                    continue
                nk  = normalize_key(ikey)
                ltp = v.get("last_price")
                if ltp is None:
                    continue
                ohlc = v.get("ohlc", {})
                out[nk] = {
                    "ltp":    float(ltp),
                    "open":   float(ohlc.get("open", ltp)),
                    "high":   float(ohlc.get("high", ltp)),
                    "low":    float(ohlc.get("low",  ltp)),
                    "volume": float(v["volume"]) if v.get("volume") else None,
                    "oi":     float(v["oi"])     if v.get("oi")     else None,
                    "last_trade_time": v.get("last_trade_time"),
                }
        except Exception:
            pass
        time.sleep(SCORE_CFG.LIVE_QUOTE_DELAY)
    return out


def patch_live_bar(df: pd.DataFrame, live: dict) -> pd.DataFrame:
    if not live:
        return df
    df = df.copy(); idx = df.index[-1]
    ph = float(df.at[idx, "high"]); pl = float(df.at[idx, "low"])
    ltp = live.get("ltp"); hi = live.get("high"); lo = live.get("low")
    vol = live.get("volume"); oi = live.get("oi")
    if ltp is not None: df.at[idx, "close"] = ltp
    if hi  is not None: df.at[idx, "high"]  = max(ph, hi)
    if lo  is not None: df.at[idx, "low"]   = min(pl, lo)
    if vol is not None: df.at[idx, "volume"] = vol
    if oi  is not None and "oi" in df.columns: df.at[idx, "oi"] = oi
    ph2 = float(df.at[idx, "high"]); pl2 = float(df.at[idx, "low"]); pc2 = float(df.at[idx, "close"])
    if ph2 < pl2: df.at[idx, "high"] = ph; df.at[idx, "low"] = pl
    if pc2 > float(df.at[idx, "high"]): df.at[idx, "high"] = pc2
    if pc2 < float(df.at[idx, "low"]):  df.at[idx, "low"]  = pc2
    return df


# ═════════════════════════════════════════════════════════════════
# 13.  INSTRUMENT MASTER + NIFTY 50
# ═════════════════════════════════════════════════════════════════

_master_cache = {"df": None, "ts": 0.0}
_nifty50_cache = {"syms": None, "ts": 0.0}


def get_live_master() -> pd.DataFrame:
    if _master_cache["df"] is not None and time.time() - _master_cache["ts"] < SCORE_CFG.MASTER_TTL:
        return _master_cache["df"]
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r   = requests.get(url, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                data = json.load(gz)
            df = pd.DataFrame(data)
            _master_cache["df"] = df; _master_cache["ts"] = time.time()
            return df
    except Exception:
        pass
    return _master_cache["df"] or pd.DataFrame()


def get_nifty50_live() -> set:
    if _nifty50_cache["syms"] is not None and time.time() - _nifty50_cache["ts"] < SCORE_CFG.NIFTY50_TTL:
        return _nifty50_cache["syms"]
    try:
        hdr = {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
               "Referer": "https://www.nseindia.com/"}
        r = requests.get("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
                         headers=hdr, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            syms = {d["symbol"] for d in data if d.get("symbol") and d["symbol"] != "NIFTY 50"}
            if len(syms) >= 40:
                _nifty50_cache["syms"] = syms; _nifty50_cache["ts"] = time.time()
                return syms
    except Exception:
        pass
    fallback = {
        "RELIANCE","HDFCBANK","ICICIBANK","INFY","ITC","TCS","LT","SBIN","AXISBANK",
        "KOTAKBANK","BHARTIARTL","ASIANPAINT","HCLTECH","MARUTI","SUNPHARMA","ULTRACEMCO",
        "TITAN","WIPRO","NESTLEIND","POWERGRID","NTPC","BAJFINANCE","BAJAJFINSV",
        "INDUSINDBK","TECHM","M&M","TATAMOTORS","ADANIENT","ADANIPORTS","ONGC",
        "COALINDIA","JSWSTEEL","HINDALCO","TATASTEEL","BPCL","GRASIM","CIPLA",
        "DRREDDY","EICHERMOT","HEROMOTOCO","BRITANNIA","DIVISLAB","SBILIFE",
        "HDFCLIFE","APOLLOHOSP","BAJAJ-AUTO","UPL","SHREECEM","HINDUNILVR","TATACONSUM",
    }
    _nifty50_cache["syms"] = fallback; _nifty50_cache["ts"] = time.time()
    return fallback


# ═════════════════════════════════════════════════════════════════
# 14.  MARKET CONTEXT
# ═════════════════════════════════════════════════════════════════

_mkt_cache: Dict = {"data": {}, "ts": 0.0}


def get_market_context() -> dict:
    if _mkt_cache["data"] and time.time() - _mkt_cache["ts"] < SCORE_CFG.MKT_CONTEXT_TTL:
        return _mkt_cache["data"]
    out = dict(nifty_r5=None, nifty_r20=None, nifty_above_20dma=True, nifty_above_50dma=True,
               regime="BULL", vix_level=None, vix_falling=True, vix_median=14.5, vix_sigma=4.5,
               sector_returns={}, sector_returns_10d={}, top_sectors=set(),
               market_ok=True, market_notes=[])
    try:
        n = yf.download("^NSEI", period="365d", interval="1d", progress=False)
        if not n.empty:
            c = n["Close"].squeeze()
            out["nifty_r5"]  = float(c.iloc[-1] / c.iloc[-6]  - 1) if len(c) >= 6  else None
            out["nifty_r20"] = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) >= 21 else None
            dma20 = float(c.tail(20).mean())
            dma50 = float(c.tail(50).mean()) if len(c) >= 50 else dma20
            out["nifty_above_20dma"] = float(c.iloc[-1]) > dma20
            out["nifty_above_50dma"] = float(c.iloc[-1]) > dma50
            _natr  = float(c.diff().abs().tail(14).mean())
            _gap   = float(c.iloc[-1]) - dma50
            _slope = float(c.tail(20).mean()) - float(c.iloc[-11:-1].mean()) if len(c) >= 11 else 0.0
            if   _gap > 0 and _slope > 0: out["regime"] = "BULL"
            elif _gap < -_natr:           out["regime"] = "BEAR"
            else:                         out["regime"] = "CHOP"
    except Exception:
        pass
    try:
        v = yf.download("^INDIAVIX", period="365d", interval="1d", progress=False)
        if not v.empty:
            vc = v["Close"].squeeze()
            out["vix_level"]  = round(float(vc.iloc[-1]), 2)
            if len(vc) >= 5:
                out["vix_falling"] = float(np.polyfit(range(5), vc.tail(5).values, 1)[0]) < 0
            out["vix_median"] = round(float(vc.median()), 2) if len(vc) >= 20 else 14.5
            out["vix_sigma"]  = round(float(vc.std()),    2) if len(vc) >= 20 else 4.5
    except Exception:
        pass
    sr5: dict = {}; sr10: dict = {}

    def _fetch_sec(nt):
        nm, tk = nt
        try:
            s = yf.download(tk, period="60d", interval="1d", progress=False)
            if not s.empty:
                sc = s["Close"].squeeze()
                r5  = float(sc.iloc[-1] / sc.iloc[-6]  - 1) if len(sc) >= 6  else None
                r10 = float(sc.iloc[-1] / sc.iloc[-11] - 1) if len(sc) >= 11 else None
                return nm, r5, r10
        except Exception:
            pass
        return nm, None, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for nm, r5, r10 in ex.map(_fetch_sec, SECTOR_TICKERS.items()):
            if r5  is not None: sr5[nm]  = r5
            if r10 is not None: sr10[nm] = r10
    out["sector_returns"]    = sr5
    out["sector_returns_10d"]= sr10
    if sr5:
        out["top_sectors"] = {k for k, _ in sorted(sr5.items(), key=lambda x: x[1], reverse=True)[:3]}
    out["market_ok"] = out["nifty_above_20dma"]
    _mkt_cache["data"] = out; _mkt_cache["ts"] = time.time()
    return out


# ═════════════════════════════════════════════════════════════════
# 15.  CROSS-SECTIONAL PRE-COMPUTATION
# ═════════════════════════════════════════════════════════════════

def _cdf_rank_dict(raw: dict) -> dict:
    if len(raw) < 3:
        return {k: 0.5 for k in raw}
    syms = list(raw.keys())
    vals = np.array([raw[s] for s in syms], dtype=float)
    pcts = rankdata(vals, method="average") / len(vals)
    return {s: float(p) for s, p in zip(syms, pcts)}


def compute_cs_ranks(st: dict) -> None:
    cache = st.get("raw_data_cache", {})
    if len(cache) < 3:
        return

    r5r: dict = {}; r20r: dict = {}
    for sym, df in cache.items():
        c = df["close"]
        if len(c) >= 6:  r5r[sym]  = float(c.iloc[-1] / c.iloc[-6]  - 1)
        if len(c) >= 21: r20r[sym] = float(c.iloc[-1] / c.iloc[-21] - 1)
    st["cs_rs_5d"]  = _cdf_rank_dict(r5r)
    st["cs_rs_20d"] = _cdf_rank_dict(r20r)

    bbr: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < 30: continue
            bw = (2.0 * c.rolling(20).std() / c.rolling(20).mean().replace(0, np.nan)).dropna()
            if len(bw) < 10: continue
            bbr[sym] = float((bw.iloc[:-1] <= float(bw.iloc[-1])).mean())
        except Exception:
            pass
    st["cs_bb_squeeze"] = _cdf_rank_dict(bbr)

    vdr: dict = {}
    for sym, df in cache.items():
        try:
            v = df["volume"].replace(0, np.nan).dropna()
            if len(v) < 25: continue
            ratio = float(v.tail(5).mean()) / (float(v.tail(20).mean()) + 1e-9)
            h_r   = (v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)).dropna()
            vdr[sym] = float((h_r >= ratio).mean()) if len(h_r) >= 5 else float(np.clip(1.0 - ratio, 0.0, 1.0))
        except Exception:
            pass
    st["cs_vol_dryup"] = _cdf_rank_dict(vdr)

    clvr: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]; hh = df["high"]; ll = df["low"]; vv = df["volume"]
            if len(c) < 25: continue
            hl  = (hh - ll).replace(0, np.nan)
            clv = ((c - ll) - (hh - c)) / hl
            mf  = clv.fillna(0) * vv
            mfn = mf / vv.rolling(20).mean().replace(0, np.nan)
            rmf = mfn.rolling(20).sum().dropna()
            if len(rmf) < 5: continue
            clvr[sym] = float((rmf.iloc[:-1] <= float(rmf.iloc[-1])).mean())
        except Exception:
            pass
    st["cs_clv_accum"] = _cdf_rank_dict(clvr)

    vcpr: dict = {}
    for sym, df in cache.items():
        try:
            if len(df) < 60: continue
            c = df["close"]; hh = df["high"]; ll = df["low"]; vv = df["volume"]
            tr  = pd.concat([hh - ll, (hh - c.shift(1)).abs(), (ll - c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / SCORE_CFG.ATR_PERIOD, adjust=False).mean()
            vcpr[sym] = float(detect_vcp(c, hh, ll, vv, atr).get("vcp_score", 0.0))
        except Exception:
            pass
    st["cs_vcp"] = _cdf_rank_dict(vcpr)

    _ab = 0; _tot = 0
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < 20: continue
            e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            _tot += 1
            if float(c.iloc[-1]) > e20: _ab += 1
        except Exception:
            pass
    if _tot >= 10:
        br = _ab / _tot
        st["breadth_cache"] = br
        st["breadth_hist"]  = (st.get("breadth_hist", []) + [br])[-200:]

    r5a: dict = {}; r10a: dict = {}
    for sym, df in cache.items():
        try:
            sec = get_sector(sym)
            if sec is None: continue
            c = df["close"]
            if len(c) >= 6:  r5a.setdefault(sec, []).append(float(c.iloc[-1] / c.iloc[-6]  - 1))
            if len(c) >= 11: r10a.setdefault(sec, []).append(float(c.iloc[-1] / c.iloc[-11] - 1))
        except Exception:
            pass
    st["sector_returns"]     = {**st.get("mkt", {}).get("sector_returns",     {}),
                                 **{s: float(np.mean(v)) for s, v in r5a.items()}}
    st["sector_returns_10d"] = {**st.get("mkt", {}).get("sector_returns_10d", {}),
                                 **{s: float(np.mean(v)) for s, v in r10a.items()}}


def _apply_coverage_score(df_out: pd.DataFrame) -> pd.DataFrame:
    if df_out.empty or "Score" not in df_out.columns:
        return df_out
    scores = df_out["Score"].values.astype(float)
    df_out["score_percentile"] = (rankdata(scores, method="average") / max(len(scores), 1) * 100).round(1)
    return df_out


# ═════════════════════════════════════════════════════════════════
# 16.  BACKGROUND EXTRACTION
# ═════════════════════════════════════════════════════════════════

def bootstrap_calibration_from_db(st: dict, db_path: pathlib.Path) -> None:
    """Load per-stock and per-setup win rates from calibration DB into STATE on startup."""
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.cursor().execute(
            "SELECT symbol, score, forward_ret, setup FROM calibration ORDER BY ts ASC"
        ).fetchall()
        con.close()
    except Exception:
        return

    wr_h: dict = {}; wr_t: dict = {}; sv = []
    sw_h: dict = {}; sw_t: dict = {}   # per-setup win rate accumulators

    for sym, score, fwd, setup in rows:
        if score is not None: sv.append(score)
        if fwd is not None:
            # per-stock
            wr_t[sym] = wr_t.get(sym, 0) + 1
            if fwd > 0: wr_h[sym] = wr_h.get(sym, 0) + 1
            # per-setup
            if setup:
                sw_t[setup] = sw_t.get(setup, 0) + 1
                if fwd > 0: sw_h[setup] = sw_h.get(setup, 0) + 1

    # Per-stock win rate (requires ≥3 outcomes to be reliable)
    wr = {s: float(np.clip(wr_h.get(s, 0) / wr_t[s], 0.30, 0.75))
          for s in wr_t if wr_t[s] >= 3}
    if wr:
        st.setdefault("per_stock_winrate", {}).update(wr)

    # Per-setup win rate (requires ≥5 outcomes)
    sw = {s: float(np.clip(sw_h.get(s, 0) / sw_t[s], 0.30, 0.75))
          for s in sw_t if sw_t[s] >= 5}
    st["_setup_winrate"] = sw   # e.g. {"Breakout": 0.54, "Pullback": 0.48}

    if len(sv) >= 10:
        bx = [sum(1 for s in sv[i:i+20] if s > 50) / max(len(sv[i:i+20]), 1)
              for i in range(0, len(sv), 20)]
        st["breadth_hist"] = (st.get("breadth_hist", []) + bx)[-200:]


def run_extraction(targets_dict: dict, min_avg_vol: int) -> None:
    cfg = SCORE_CFG
    with STATE_LOCK:
        s = STATE["extraction_status"]
        s.update({"running": True, "done": 0, "total": len(targets_dict),
                  "errors": 0, "rate_limited": 0, "log": []})
        STATE["raw_data_cache"] = {}; STATE["score_cache"] = {}
        STATE["_row_stream_queue"] = []   # clear any leftover rows from previous run
        for k in ("cs_rs_5d","cs_rs_20d","cs_bb_squeeze","cs_vol_dryup","cs_clv_accum","cs_vcp"):
            STATE[k] = {}
        STATE["breadth_cache"] = None

    status   = STATE["extraction_status"]
    end_dt   = datetime.now().strftime('%Y-%m-%d')
    start_dt = (datetime.now() - timedelta(days=cfg.HISTORY_DAYS)).strftime('%Y-%m-%d')

    live_q = fetch_live_quotes(list(targets_dict.values()))
    with STATE_LOCK:
        STATE["live_quotes_cache"] = live_q
        STATE["last_live_refresh"] = time.time()

    sym_keys = list(targets_dict.items())
    if min_avg_vol > 0 and live_q:
        sym_keys = [(s, k) for s, k in sym_keys
                    if (lq := live_q.get(normalize_key(k))) is None
                    or lq.get("volume") is None
                    or float(lq["volume"]) >= min_avg_vol * 0.20]
    status["total"] = len(sym_keys)

    def _fetch_one(pair):
        sym, key = pair
        url   = (f"https://api.upstox.com/v2/historical-candle/"
                 f"{urllib.parse.quote(key)}/day/{end_dt}/{start_dt}")
        delay = cfg.FETCH_BACKOFF
        hdr   = get_headers()
        for attempt in range(cfg.FETCH_RETRIES + 1):
            try:
                time.sleep(cfg.FETCH_DELAY)
                r = requests.get(url, headers=hdr, timeout=15)
                if r.status_code == 429:
                    if attempt < cfg.FETCH_RETRIES:
                        time.sleep(delay); delay *= 2; continue
                    return sym, None, "HTTP 429"
                if r.status_code != 200:
                    return sym, None, f"HTTP {r.status_code}"
                raw = r.json().get("data", {}).get("candles", [])
                if not raw:
                    return sym, None, "empty"
                df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume","oi"])
                df = to_ascending(df)
                lq = live_q.get(normalize_key(key))
                if lq:
                    df = patch_live_bar(df, lq)
                return sym, df, None
            except requests.exceptions.Timeout:
                if attempt < cfg.FETCH_RETRIES:
                    time.sleep(delay); delay *= 2; continue
                return sym, None, "timeout"
            except Exception as e:
                return sym, None, str(e)
        return sym, None, "max retries"

    with ThreadPoolExecutor(max_workers=cfg.FETCH_WORKERS) as executor:
        futs = {executor.submit(_fetch_one, p): p for p in sym_keys}
        for fut in as_completed(futs):
            sym, df, err = fut.result()
            status["done"] += 1
            if err:
                if "429" in str(err): status["rate_limited"] += 1
                elif err not in ("empty",): status["errors"] += 1
                continue
            if df is None: continue
            df_c = df[df["volume"] > 0].copy() if "volume" in df.columns else df.copy()
            if len(df_c) < 30: continue
            if min_avg_vol > 0 and len(df_c) >= 5:
                if float(df_c["volume"].tail(20).mean()) < min_avg_vol:
                    continue
            with STATE_LOCK:
                STATE["raw_data_cache"][sym] = df_c

            # ── Progressive score: score this stock immediately with whatever
            # CS ranks are available so far. The result is provisional — RS
            # percentile ranks will shift as more stocks arrive — but it lets
            # the frontend show a live-updating table rather than a blank screen.
            # A final re-score pass runs after compute_cs_ranks() completes.
            try:
                mkt      = STATE.get("mkt") or {}
                live_lq  = live_q.get(normalize_key(targets_dict.get(sym, "")), {})
                result   = score_stock_dual(sym, df_c, live_lq,
                                            mkt.get("nifty_r5"), mkt.get("nifty_r20"))
                if result is not None:
                    ltp_now = live_lq.get("ltp") or float(df_c["close"].iloc[-1])
                    vol_now = live_lq.get("volume") or (float(df_c["volume"].iloc[-1]) if "volume" in df_c.columns else 0)
                    prev_close = float(df_c["close"].iloc[-2]) if len(df_c) >= 2 else ltp_now
                    day_chg = round((ltp_now - prev_close) / (prev_close + 1e-9) * 100, 2)
                    row = {
                        "Ticker": sym, "LTP": round(float(ltp_now), 2),
                        "DayChg_pct": day_chg,
                        "DayHigh": round(float(live_lq.get("high", df_c["high"].iloc[-1])), 2),
                        "DayLow":  round(float(live_lq.get("low",  df_c["low"].iloc[-1])),  2),
                        "LiveVol": int(live_lq["volume"]) if live_lq.get("volume") else None,
                        **result,
                    }
                    # Sanitize NaN/inf for JSON serialisation
                    row = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                           for k, v in row.items()}
                    with STATE_LOCK:
                        STATE["score_cache"][sym] = {
                            "result": result, "ltp": ltp_now, "vol": vol_now,
                            "rsi_period": STATE.get("rsi_period", 7),
                        }
                        STATE["_row_stream_queue"].append(row)
            except Exception:
                pass

    compute_cs_ranks(STATE)
    with STATE_LOCK:
        status["running"] = False
        # Invalidate provisional scores so the next GET /api/screener re-scores
        # with the full cross-sectional RS ranks now that all stocks are loaded.
        STATE["score_cache"] = {}
        # Signal the SSE stream that the final full-universe rescore is ready.
        # The frontend will call loadScreener() once on this event to get definitive ranks.
        STATE["_row_stream_queue"].append({"__event__": "rescore_complete"})

    try:
        universe = STATE.get("_last_universe", "unknown")
        rows_to_save = []
        with STATE_LOCK:
            cs = dict(STATE["raw_data_cache"])
            sc = dict(STATE["score_cache"])
            tg = dict(STATE["targets"])
            lq = dict(STATE["live_quotes_cache"])
        for sym, df_raw in cs.items():
            cached = sc.get(sym, {}).get("result")
            if cached:
                live = lq.get(normalize_key(tg.get(sym, "")), {})
                rows_to_save.append({"Ticker": sym,
                                     "LTP": round(live.get("ltp") or float(df_raw["close"].iloc[-1]), 2),
                                     **cached})
        if rows_to_save:
            save_snapshot(rows_to_save, universe)
    except Exception:
        pass


def refresh_live_prices_bg() -> None:
    if time.time() - STATE["last_live_refresh"] < SCORE_CFG.LIVE_REFRESH_SEC:
        return
    if not STATE["raw_data_cache"] or not STATE["targets"]:
        return
    live = fetch_live_quotes(list(STATE["targets"].values()))
    if not live:
        return
    with STATE_LOCK:
        for sym, df in STATE["raw_data_cache"].items():
            key = STATE["targets"].get(sym)
            if not key: continue
            lq = live.get(normalize_key(key))
            if not lq: continue
            STATE["raw_data_cache"][sym] = patch_live_bar(df, lq)
        STATE["live_quotes_cache"] = live
        STATE["last_live_refresh"]  = time.time()
        # Selective cache invalidation: evict entries where LTP moved >1% OR rsi_period changed.
        # The old approach wiped the entire score_cache, forcing a full VCP/Darvas/RS rescore
        # for all stocks on every price tick.  Indicators that don't depend on LTP (MA structure, VCP,
        # Darvas, RS percentile ranks) are still valid after a small price move.
        current_rsi = STATE.get("rsi_period", 7)
        to_evict = []
        for sym, cached_e in STATE["score_cache"].items():
            if cached_e.get("rsi_period") != current_rsi:
                to_evict.append(sym); continue
            lq = live.get(normalize_key(STATE["targets"].get(sym, "")), {})
            ltp_now = lq.get("ltp")
            if ltp_now is not None:
                cached_ltp = cached_e.get("ltp", 0)
                if cached_ltp > 0 and abs(ltp_now - cached_ltp) / cached_ltp > 0.01:
                    to_evict.append(sym)
        for sym in to_evict:
            STATE["score_cache"].pop(sym, None)


# ═════════════════════════════════════════════════════════════════
# 17.  DATABASE
# ═════════════════════════════════════════════════════════════════

DB_PATH  = pathlib.Path("monarch_data.db")
_db_local = threading.local()


def get_db() -> sqlite3.Connection:
    conn = getattr(_db_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL mode and pragmas set once at connection creation — not re-issued on reuse.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster than FULL
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA cache_size=-8000")     # 8 MB page cache per connection
        _db_local.conn = conn
    return conn


def init_db() -> None:
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT NOT NULL,
            universe TEXT NOT NULL,
            row_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts);

        CREATE TABLE IF NOT EXISTS calibration (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            score       REAL,
            forward_ret REAL,
            horizon     TEXT,
            setup       TEXT,
            regime      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cal_sym     ON calibration(symbol);
        CREATE INDEX IF NOT EXISTS idx_cal_pending ON calibration(ts)
            WHERE forward_ret IS NULL;

        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            note   TEXT,
            added  TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            cond   TEXT NOT NULL,
            value  REAL NOT NULL,
            fired  INTEGER DEFAULT 0,
            ts     TEXT NOT NULL
        );
    """)
    conn.commit()


init_db()
bootstrap_calibration_from_db(STATE, DB_PATH)


def save_snapshot(rows: list, universe: str) -> None:
    if not rows: return
    ts = _dt.now().isoformat(); conn = get_db()
    conn.executemany(
        "INSERT INTO snapshots(ts,universe,row_json) VALUES(?,?,?)",
        [(ts, universe, _json.dumps(row)) for row in rows]
    )
    conn.commit()


# ═════════════════════════════════════════════════════════════════
# 18.  API ROUTES
# ═════════════════════════════════════════════════════════════════

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("static/login.html", encoding="utf-8") as f: return f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f: return f.read()

@app.get("/options", response_class=HTMLResponse)
async def options_page():
    with open("static/pages/options.html", encoding="utf-8") as f: return f.read()

@app.get("/fundamentals", response_class=HTMLResponse)
async def fundamentals_page():
    with open("static/pages/fundamentals.html", encoding="utf-8") as f: return f.read()

@app.get("/ml", response_class=HTMLResponse)
async def ml_page():
    with open("static/pages/ml.html", encoding="utf-8") as f: return f.read()

@app.get("/polymarket", response_class=HTMLResponse)
async def polymarket_page():
    with open("static/pages/polymarket.html", encoding="utf-8") as f: return f.read()

# ── Token ─────────────────────────────────────────────────────────
@app.post("/api/token")
async def set_token(body: dict):
    tok = body.get("token", "").strip()
    if not tok: raise HTTPException(400, "Empty token")
    STATE["token"] = tok
    try:
        with open(".upstox_token", "w") as f: f.write(tok)
    except Exception: pass
    return {"status": "ok", "token_prefix": tok[:16]}

@app.get("/api/token/status")
async def token_status():
    if not STATE["token"]:
        try:
            if os.path.exists(".upstox_token"):
                t = open(".upstox_token").read().strip()
                if t: STATE["token"] = t
        except Exception: pass
    return {"connected": bool(STATE["token"]),
            "prefix": STATE["token"][:16] if STATE["token"] else ""}

# ── Universe / Extract ────────────────────────────────────────────
@app.get("/api/universe")
async def get_universe(universe: str = "Nifty 50"):
    master_df = get_live_master()
    if master_df.empty: raise HTTPException(503, "Could not load instrument master")
    eq = master_df[(master_df["exchange"] == "NSE") & (master_df["instrument_type"] == "EQ")]
    if universe == "Nifty 50":
        df = eq[eq["trading_symbol"].isin(get_nifty50_live())]
    elif universe == "F&O Stocks":
        fo = master_df[master_df["segment"].astype(str).str.contains("FO", na=False)]
        df = eq[eq["trading_symbol"].isin(fo["underlying_symbol"].dropna().astype(str).unique())]
    else:
        df = eq
    targets = {r["trading_symbol"]: r["instrument_key"] for _, r in df.iterrows()}
    STATE["targets"] = targets
    return {"count": len(targets), "symbols": list(targets.keys())[:20]}

@app.post("/api/extract")
async def start_extraction(body: dict, background_tasks: BackgroundTasks):
    universe = body.get("universe", "Nifty 50")
    min_vol  = body.get("min_avg_vol", 100_000)
    rsi_p    = body.get("rsi_period",  7)
    STATE["rsi_period"]          = int(rsi_p)
    STATE["min_avg_vol"]         = int(min_vol)
    STATE["sector_cap_enabled"]  = body.get("sector_cap_enabled", False)
    if STATE["extraction_status"]["running"]:
        return {"status": "already_running"}
    master_df = get_live_master()
    if master_df.empty: raise HTTPException(503, "Master list unavailable")
    eq = master_df[(master_df["exchange"] == "NSE") & (master_df["instrument_type"] == "EQ")]
    if universe == "Nifty 50":
        df = eq[eq["trading_symbol"].isin(get_nifty50_live())]
    elif universe == "F&O Stocks":
        fo = master_df[master_df["segment"].astype(str).str.contains("FO", na=False)]
        df = eq[eq["trading_symbol"].isin(fo["underlying_symbol"].dropna().astype(str).unique())]
    else:
        df = eq
    targets = {r["trading_symbol"]: r["instrument_key"] for _, r in df.iterrows()}
    STATE["targets"] = targets
    mkt = get_market_context()
    STATE["mkt"]               = mkt
    STATE["sector_returns"]    = mkt.get("sector_returns",     {})
    STATE["sector_returns_10d"]= mkt.get("sector_returns_10d", {})
    STATE["top_sectors"]       = mkt.get("top_sectors",         set())
    STATE["_last_universe"]    = universe
    background_tasks.add_task(run_extraction, targets, int(min_vol))
    return {"status": "started", "total": len(targets)}

@app.get("/api/extraction/status")
async def extraction_status():
    s = STATE["extraction_status"]
    return {**s, "cached": len(STATE["raw_data_cache"]),
            "live_quotes": len(STATE["live_quotes_cache"]),
            "last_refresh": STATE["last_live_refresh"]}


@app.get("/api/extraction/stream")
async def extraction_stream():
    """
    Server-Sent Events endpoint.
    Emits:
      event: status  — extraction progress (same shape as /api/extraction/status)
      event: prices  — live LTP/chg/vol patch { TICKER: {ltp, chg, vol} }
                       pushed whenever live_quotes_cache is refreshed.
    Falls back silently if the client disconnects.
    """
    import asyncio

    async def _generator():
        last_refresh = -1
        last_status_hash = None
        while True:
            try:
                # ── status event ───────────────────────────────────
                s = STATE["extraction_status"]
                payload = {
                    **s,
                    "cached":       len(STATE["raw_data_cache"]),
                    "live_quotes":  len(STATE["live_quotes_cache"]),
                    "last_refresh": STATE["last_live_refresh"],
                }
                h = (payload["done"], payload["total"], payload["running"])
                if h != last_status_hash:
                    last_status_hash = h
                    yield f"event: status\ndata: {json.dumps(payload)}\n\n"

                # ── progressive row events ─────────────────────────
                # Drain whatever has been queued by run_extraction and push
                # each scored row immediately so the frontend can show it.
                with STATE_LOCK:
                    pending = STATE["_row_stream_queue"][:]
                    STATE["_row_stream_queue"].clear()
                for row in pending:
                    evt = row.get("__event__")
                    if evt == "rescore_complete":
                        # Tell the frontend the full-universe CS ranks are ready;
                        # it should call loadScreener() once to get definitive scores.
                        yield f"event: rescore_complete\ndata: {{}}\n\n"
                    else:
                        yield f"event: row\ndata: {json.dumps(row, default=str)}\n\n"

                # ── prices event when live quotes updated ──────────
                cur_refresh = STATE["last_live_refresh"]
                if cur_refresh != last_refresh and cur_refresh > 0:
                    last_refresh = cur_refresh
                    prices: dict = {}
                    for sym, df_raw in STATE["raw_data_cache"].items():
                        live = STATE["live_quotes_cache"].get(
                            normalize_key(STATE["targets"].get(sym, "")), {}
                        )
                        if not live:
                            continue
                        ltp_now = live.get("ltp")
                        if ltp_now is None:
                            continue
                        prev_close = (float(df_raw["close"].iloc[-2])
                                      if len(df_raw) >= 2 else float(ltp_now))
                        chg = round((float(ltp_now) - prev_close)
                                    / (prev_close + 1e-9) * 100, 2)
                        prices[sym] = {
                            "ltp": round(float(ltp_now), 2),
                            "chg": chg,
                            "vol": int(live["volume"]) if live.get("volume") else None,
                        }
                    if prices:
                        yield f"event: prices\ndata: {json.dumps(prices)}\n\n"

                await asyncio.sleep(0.3)   # tighter loop during extraction for snappier updates
            except asyncio.CancelledError:
                break
            except Exception:
                break

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )

@app.get("/api/screener")
async def get_screener(sort_by: str = "CompositeRank", horizon: str = "ALL"):
    # Kick off a background live-price refresh without blocking the response.
    # refresh_live_prices_bg() is a blocking network call (fetch_live_quotes can
    # take 1-3 s); calling it inline was hanging every screener page load.
    threading.Thread(target=refresh_live_prices_bg, daemon=True).start()
    mkt      = STATE["mkt"] or get_market_context()
    nifty_r5 = mkt.get("nifty_r5"); nifty_r20 = mkt.get("nifty_r20")
    rows     = []
    _min_vol = STATE.get("min_avg_vol", 0)
    # Snapshot mutable shared dicts before the loop — prevents RuntimeError
    # if another thread mutates them (e.g. refresh_live_prices_bg) mid-iteration.
    raw_snap  = dict(STATE["raw_data_cache"])
    lq_snap   = dict(STATE["live_quotes_cache"])
    tgt_snap  = dict(STATE["targets"])
    sc_snap   = dict(STATE["score_cache"])
    # Compute universe-wide BO saturation fraction from last score_cache pass.
    _cached_results = [v.get("result") for v in sc_snap.values() if v.get("result")]
    _bo_count  = sum(1 for r in _cached_results if r.get("SetupType") == "Breakout")
    _tot_cache = max(len(_cached_results), 1)
    STATE["_bo_saturation_frac"] = _bo_count / _tot_cache

    for sym, df_raw in raw_snap.items():
        try:
            if _min_vol > 0 and "volume" in df_raw.columns and len(df_raw) >= 5:
                if float(df_raw["volume"].tail(20).mean()) < _min_vol: continue
            live     = lq_snap.get(normalize_key(tgt_snap.get(sym, "")), {})
            ltp_now  = live.get("ltp") or float(df_raw["close"].iloc[-1])
            vol_now  = live.get("volume") or (float(df_raw["volume"].iloc[-1]) if "volume" in df_raw.columns else 0)
            cached_e = sc_snap.get(sym)
            # Cache hit: LTP within 1% AND rsi_period unchanged
            rsi_match = cached_e and cached_e.get("rsi_period") == STATE.get("rsi_period", 7)
            ltp_match = cached_e and abs(cached_e.get("ltp", 0) - ltp_now) < 0.01 * ltp_now
            if rsi_match and ltp_match and cached_e.get("result") is not None:
                result = cached_e["result"]
            else:
                result = score_stock_dual(sym, df_raw, live, nifty_r5, nifty_r20)
                STATE["score_cache"][sym] = {
                    "result": result, "ltp": ltp_now, "vol": vol_now,
                    "rsi_period": STATE.get("rsi_period", 7),
                }
            if result is None: continue
            prev_close = float(df_raw["close"].iloc[-2]) if len(df_raw) >= 2 else ltp_now
            day_chg    = round((ltp_now - prev_close) / (prev_close + 1e-9) * 100, 2) if prev_close else None
            rows.append({
                "Ticker": sym, "LTP": round(float(ltp_now), 2),
                "DayChg_pct": day_chg,
                "DayHigh": round(float(live.get("high", df_raw["high"].iloc[-1])), 2),
                "DayLow":  round(float(live.get("low",  df_raw["low"].iloc[-1])),  2),
                "LiveVol": int(live["volume"]) if live.get("volume") else None,
                **result,
            })
        except Exception:
            continue
    if not rows:
        return {"rows": [], "regime": mkt.get("regime", "BULL"),
                "vix": mkt.get("vix_level"), "market_ok": mkt.get("market_ok", True),
                "top_sectors": list(STATE.get("top_sectors", set())),
                "nifty_above_50dma": mkt.get("nifty_above_50dma", True),
                "vix_falling": mkt.get("vix_falling", True),
                "market_notes": mkt.get("market_notes", [])}
    df_out = pd.DataFrame(rows)
    df_out  = _apply_coverage_score(df_out)
    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by, ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", df_out.index + 1)
    if STATE.get("sector_cap_enabled") and "Sector" in df_out.columns:
        seen = set(); capped = []
        for _, row in df_out.iterrows():
            sec = str(row.get("Sector", "?"))
            if sec == "?" or sec not in seen: capped.append(row)
            if sec != "?": seen.add(sec)
        df_out = pd.DataFrame(capped).reset_index(drop=True); df_out["Rank"] = df_out.index + 1
    if horizon != "ALL":
        df_out = df_out[df_out["Horizon"] == horizon].reset_index(drop=True); df_out["Rank"] = df_out.index + 1
    df_out = df_out.replace({np.nan: None, np.inf: None, -np.inf: None})
    sector_ret = {s: round(v * 100, 2) for s, v in STATE["sector_returns"].items()}
    return {
        "rows": df_out.to_dict("records"),
        "regime": mkt.get("regime", "BULL"),
        "vix": mkt.get("vix_level"),
        "market_ok": mkt.get("market_ok", True),
        "top_sectors": list(STATE.get("top_sectors", set())),
        "nifty_above_50dma": mkt.get("nifty_above_50dma", True),
        "vix_falling": mkt.get("vix_falling", True),
        "market_notes": mkt.get("market_notes", []),
        "sector_returns": sector_ret,
        "breadth": STATE.get("breadth_cache"),
        "total": len(rows),
        "live_quotes": len(STATE["live_quotes_cache"]),
        "last_refresh": STATE["last_live_refresh"],
    }

@app.post("/api/refresh_live")
async def refresh_live(background_tasks: BackgroundTasks):
    background_tasks.add_task(refresh_live_prices_bg)
    return {"status": "refreshing"}

@app.get("/api/config")
async def get_config():
    return {"rsi_period": STATE["rsi_period"], "min_avg_vol": STATE["min_avg_vol"],
            "sector_cap_enabled": STATE["sector_cap_enabled"]}

@app.post("/api/config")
async def set_config(body: dict):
    if "rsi_period" in body:
        rp = int(body["rsi_period"])
        if rp not in (7, 14): raise HTTPException(400, "rsi_period must be 7 or 14")
        if STATE.get("rsi_period") != rp:
            STATE["rsi_period"] = rp
            STATE["score_cache"] = {}   # rsi_period change invalidates all cached scores
    if "min_avg_vol" in body:
        mv = int(body["min_avg_vol"])
        if mv < 0: raise HTTPException(400, "min_avg_vol must be non-negative")
        STATE["min_avg_vol"] = mv
    if "sector_cap_enabled" in body:
        STATE["sector_cap_enabled"] = bool(body["sector_cap_enabled"])
    return {"status": "ok"}

# ── DB endpoints ──────────────────────────────────────────────────
@app.get("/api/db/snapshots")
async def list_snapshots(limit: int = 20):
    rows = get_db().execute(
        "SELECT DISTINCT ts, universe, COUNT(*) as n FROM snapshots GROUP BY ts ORDER BY ts DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return {"snapshots": [dict(r) for r in rows]}

@app.get("/api/db/snapshot")
async def get_snapshot(ts: str):
    rows = get_db().execute("SELECT row_json FROM snapshots WHERE ts=? ORDER BY rowid", (ts,)).fetchall()
    return {"ts": ts, "rows": [_json.loads(r["row_json"]) for r in rows], "count": len(rows)}

@app.delete("/api/db/snapshots/old")
async def purge_old_snapshots(keep_days: int = SCORE_CFG.SNAPSHOT_KEEP_DAYS):
    cutoff = (_dt.now() - timedelta(days=keep_days)).isoformat()
    conn = get_db()
    n = conn.execute("DELETE FROM snapshots WHERE ts < ?", (cutoff,)).rowcount
    conn.commit()
    return {"deleted": n}

@app.post("/api/db/calibration")
async def add_calibration(body: dict):
    sym = str(body.get("symbol", "")).upper().strip()
    if not sym: raise HTTPException(400, "symbol required")
    score = body.get("score")
    if score is not None:
        try: score = float(score)
        except (TypeError, ValueError): raise HTTPException(400, "score must be numeric")
    horizon = str(body.get("horizon", ""))[:32]
    setup   = str(body.get("setup",   ""))[:32]
    regime  = str(body.get("regime",  ""))[:16]
    conn = get_db()
    conn.execute(
        "INSERT INTO calibration(ts,symbol,score,forward_ret,horizon,setup,regime) VALUES(?,?,?,?,?,?,?)",
        (_dt.now().isoformat(), sym, score, body.get("forward_ret"), horizon, setup, regime)
    )
    conn.commit()
    return {"status": "ok"}

@app.get("/api/db/calibration/stats")
async def calibration_stats():
    rows = get_db().execute(
        "SELECT symbol, score, forward_ret, horizon, setup, regime "
        "FROM calibration WHERE forward_ret IS NOT NULL"
    ).fetchall()
    if not rows: return {"error": "no data yet"}
    data   = [dict(r) for r in rows]
    scores = [r["score"]       for r in data if r["score"]       is not None]
    rets   = [r["forward_ret"] for r in data if r["forward_ret"] is not None]
    if len(scores) < 4: return {"count": len(data), "note": "need at least 4 records"}
    q25, q50, q75 = float(np.percentile(scores, 25)), float(np.percentile(scores, 50)), float(np.percentile(scores, 75))
    qts = {"Q1 (0-25%)": [], "Q2 (25-50%)": [], "Q3 (50-75%)": [], "Q4 (75-100%)": []}
    for r in data:
        s, ret = r.get("score"), r.get("forward_ret")
        if s is None or ret is None: continue
        if   s <= q25: qts["Q1 (0-25%)"].append(ret)
        elif s <= q50: qts["Q2 (25-50%)"].append(ret)
        elif s <= q75: qts["Q3 (50-75%)"].append(ret)
        else:          qts["Q4 (75-100%)"].append(ret)
    result = {q: {"n": len(v), "avg_ret": round(float(np.mean(v)), 3),
                  "win_rate": round(float(np.mean([1 if x > 0 else 0 for x in v])) * 100, 1)}
              for q, v in qts.items() if v}
    return {
        "count": len(data),
        "overall_win_rate": round(float(np.mean([1 if r > 0 else 0 for r in rets])) * 100, 1),
        "overall_avg_ret":  round(float(np.mean(rets)), 3),
        "quartiles": result,
        "score_percentiles": {"p25": round(q25,1), "p50": round(q50,1), "p75": round(q75,1)},
    }

@app.post("/api/db/calibration/update_returns")
async def update_forward_returns():
    """Fill forward returns for pending calibration rows, then reload win rates into STATE."""
    conn    = get_db()
    pending = conn.execute("SELECT id, symbol, ts FROM calibration WHERE forward_ret IS NULL").fetchall()
    updated = 0
    for rec in pending:
        sym = rec["symbol"]
        lt  = _dt.fromisoformat(rec["ts"])
        df  = STATE["raw_data_cache"].get(sym)
        if df is None or "time" not in df.columns: continue
        df_t = df.copy(); df_t["time"] = pd.to_datetime(df_t["time"])
        df_t = df_t.sort_values("time").reset_index(drop=True)
        idx  = (df_t["time"] - lt).abs().idxmin()
        fi   = idx + SCORE_CFG.FORWARD_RETURN_DAYS
        if fi < len(df_t):
            entry  = float(df_t.loc[idx, "close"])
            exit_p = float(df_t.loc[fi,  "close"])
            conn.execute("UPDATE calibration SET forward_ret=? WHERE id=?",
                         (round((exit_p / entry - 1) * 100, 3), rec["id"]))
            updated += 1
    conn.commit()

    # ── Reload per-stock and per-setup win rates from full DB ──────────
    # This closes the feedback loop: Kelly sizing and setup confidence now
    # reflect actual observed outcomes rather than the theoretical formula.
    rows_all = conn.execute(
        "SELECT symbol, forward_ret, setup FROM calibration WHERE forward_ret IS NOT NULL"
    ).fetchall()
    wr_hits: dict = {}; wr_total: dict = {}
    sw_hits: dict = {}; sw_total: dict = {}
    for row in rows_all:
        sym = row["symbol"]; ret = row["forward_ret"]; setup = row["setup"] or ""
        wr_total[sym] = wr_total.get(sym, 0) + 1
        if ret > 0: wr_hits[sym] = wr_hits.get(sym, 0) + 1
        if setup:
            sw_total[setup] = sw_total.get(setup, 0) + 1
            if ret > 0: sw_hits[setup] = sw_hits.get(setup, 0) + 1

    new_wr = {s: float(np.clip(wr_hits.get(s, 0) / wr_total[s], 0.30, 0.75))
              for s in wr_total if wr_total[s] >= 3}
    new_sw = {s: float(np.clip(sw_hits.get(s, 0) / sw_total[s], 0.30, 0.75))
              for s in sw_total if sw_total[s] >= 5}

    with STATE_LOCK:
        STATE["per_stock_winrate"].update(new_wr)
        STATE["_setup_winrate"] = new_sw
        # Invalidate score cache so the next screener load picks up new Kelly fractions.
        # Only clear if we actually updated win rates — avoids redundant rescoring.
        if updated > 0:
            STATE["score_cache"] = {}

    return {"updated": updated, "pending": len(pending),
            "wr_stocks_loaded": len(new_wr), "wr_setups_loaded": len(new_sw),
            "setup_winrates": new_sw}

# ── Watchlist ─────────────────────────────────────────────────────
@app.get("/api/watchlist")
async def get_watchlist():
    return {"symbols": [dict(r) for r in get_db().execute("SELECT * FROM watchlist ORDER BY added DESC").fetchall()]}

@app.post("/api/watchlist")
async def add_to_watchlist(body: dict):
    sym = str(body.get("symbol", "")).upper().strip()
    if not sym: raise HTTPException(400, "Symbol required")
    if len(sym) > 20: raise HTTPException(400, "Symbol too long")
    note = str(body.get("note", ""))[:200]  # cap note length
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO watchlist(symbol,note,added) VALUES(?,?,?)",
                 (sym, note, _dt.now().isoformat()))
    conn.commit()
    return {"status": "ok", "symbol": sym}

@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    conn = get_db()
    conn.execute("DELETE FROM watchlist WHERE symbol=?", (symbol.upper(),))
    conn.commit()
    return {"status": "ok"}

# ── News ──────────────────────────────────────────────────────────
@app.get("/api/news")
async def get_news(symbol: str = ""):
    try:
        import feedparser, html as _html
    except ImportError:
        return {"articles": [], "symbol": symbol, "error": "feedparser not installed"}
    feeds = [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ]
    arts = []
    for url in feeds:
        try:
            for e in feedparser.parse(url).entries[:15]:
                title = _html.unescape(getattr(e, "title",   ""))
                summ  = _html.unescape(getattr(e, "summary", ""))[:200]
                link  = getattr(e, "link",      "#")
                pub   = getattr(e, "published", "")
                if symbol and symbol.upper() not in (title + summ).upper(): continue
                arts.append({"title": title, "summary": summ, "link": link,
                              "pub": pub, "source": url.split("/")[2]})
        except Exception:
            pass
    return {"articles": arts[:30], "symbol": symbol}

# ── Chart ─────────────────────────────────────────────────────────
@app.get("/api/chart/{symbol}")
async def get_chart_data(symbol: str):
    df = STATE["raw_data_cache"].get(symbol.upper())
    if df is None: raise HTTPException(404, f"{symbol} not in cache — run extraction first")
    df = df.copy().tail(SCORE_CFG.CHART_BARS)
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    c = df["close"]
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean().round(2)
    df["ema20"] = c.ewm(span=20, adjust=False).mean().round(2)
    df["ema50"] = c.ewm(span=50, adjust=False).mean().round(2)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"]  - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"]    = tr.ewm(span=14, adjust=False).mean().round(2)
    rsi_p        = STATE.get("rsi_period", 7)
    delta        = c.diff()
    gain         = delta.clip(lower=0).ewm(alpha=1 / rsi_p, adjust=False).mean()
    loss         = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_p, adjust=False).mean()
    df["rsi"]    = (100 - 100 / (1 + gain / loss.replace(0, float("nan")))).round(1)
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean().round(0)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
    cols = ["time","open","high","low","close","volume","ema9","ema20","ema50","atr","rsi","vol_ma20"]
    return {"symbol": symbol.upper(), "bars": df[[c for c in cols if c in df.columns]].to_dict("records")}

# ── Explain ───────────────────────────────────────────────────────
@app.get("/api/explain/{symbol}")
async def explain_score(symbol: str):
    sym    = symbol.upper()
    df_raw = STATE["raw_data_cache"].get(sym)
    if df_raw is None: raise HTTPException(404, "Run extraction first — symbol not in cache")
    live   = STATE["live_quotes_cache"].get(normalize_key(STATE["targets"].get(sym, "")), {})
    cached = STATE["score_cache"].get(sym, {}).get("result")
    if not cached:
        mkt = STATE.get("mkt") or {}
        try:
            cached = score_stock_dual(sym, df_raw, live, mkt.get("nifty_r5"), mkt.get("nifty_r20"))
            if cached:
                ltp_now = live.get("ltp") or float(df_raw["close"].iloc[-1])
                STATE["score_cache"][sym] = {
                    "result": cached, "ltp": ltp_now, "vol": 0,
                    "rsi_period": STATE.get("rsi_period", 7),
                }
        except Exception as e:
            raise HTTPException(500, f"Score computation failed: {e}")
    if not cached: raise HTTPException(404, "Score unavailable — insufficient data (need 60+ bars)")
    s   = cached
    ltp = live.get("ltp") or float(df_raw["close"].iloc[-1])

    def _fv(v, dec=2):
        if v is None: return "—"
        try: return str(round(float(v), dec))
        except: return str(v)

    lines = []
    def line(label, value, why, implication):
        lines.append({"label": label, "value": str(value) if value is not None else "—",
                      "why": why, "implication": implication})

    line("LTP",           f"₹{ltp:.2f}",                     "Last traded price",                                               "Current market price")
    line("Score",         _fv(s.get("Score"),1),              "Composite quality score 0–100 (signal_strength × coverage)",      "Higher = better setup quality; no signal gaps inflate score")
    line("Setup Type",    s.get("SetupType","—"),             "Pattern detected on price/volume analysis",                       "Defines entry strategy and target multiplier")
    line("Horizon",       s.get("Horizon","—"),               "Expected trade duration based on distance to trigger",            "Match to your holding capacity")
    line("EMI",           _fv(s.get("EMI"),3),                "Score × ATR% — reward-adjusted quality index",                   "Higher EMI = better quality per unit of volatility")
    line("Composite Rank",_fv(s.get("CompositeRank"),4),      "EMI×0.70 + LiquidityScore×0.20 + Stability×0.10",                "Lower = better rank in the universe")
    line("RS Score",      f"{_fv(s.get('RS'),1)}/15 pts",     "Vol-normalised alpha vs Nifty + cross-sectional universe rank",   "Higher = leading the market; look for >10/15")
    line("Sector RS",     f"{_fv(s.get('RS_Sector'),1)}/10",  "Stock's vol-normalised return vs sector average (0-10 pts)",     "Is this the best stock in its sector?")
    line("Volume",        f"{_fv(s.get('Volume'),1)}/15",      "Volume pattern score: surge (Breakout) or dryup (Pullback)",     "Breakout: high vol = good. Pullback: low vol = good")
    line("Coil",          f"{_fv(s.get('Coil'),1)}/10",       "Adaptive coiling quality (0-10 pts)",                            "Higher coil = tighter base = better setup")
    line("MA Structure",  f"{_fv(s.get('MA_Struct'),1)}/10",  "EMA9/20/50 alignment + slope (percentile-rank based)",           "All EMAs rising and stacked = bullish structure")
    line("Proximity",     f"{_fv(s.get('Proximity'),1)}/10",  "Distance to ideal entry relative to this stock's own history",   "Higher = closer to optimal entry price zone")
    line("BB Squeeze",    _fv(s.get("BBSqueeze"),1),          "Bollinger Band width vs universe CDF rank (0-8 pts)",            "Higher = tighter bands vs peers = coiling energy")
    line("Vol Dryup",     _fv(s.get("VolDryUp"),1),           "Volume drying up during consolidation (0-8 pts)",                "Higher = healthy coil with low-vol pullback")
    line("CLV Accum",     _fv(s.get("CLVAccum"),1),           "Close Location Value — buy pressure over 20 days (0-8 pts)",     "Higher = consistent institutional buying into close")
    line("VCP Score",     _fv(s.get("VCP"),1),                "Volatility Contraction Pattern composite (0-10 pts)",            "Higher = cleaner Minervini VCP pattern")
    line("VCP Detected",  "YES" if s.get("VCP_Detected") else "NO", "VCP confirmed (score≥0.55 + contraction≥0.4)",            "YES = textbook setup; expect breakout on volume")
    line("VCP Pullbacks", str(s.get("VCP_Pullbacks","—")),    "Number of contracting pullbacks detected",                       "3+ pullbacks each smaller = ideal VCP")
    line("Darvas In Box", "YES" if s.get("DarvasInBox") else "NO", "Price contained within Darvas box boundaries",              "YES = contained consolidation, potential breakout")
    line("Spread Pts",    _fv(s.get("SpreadPts"),1),          "Spread compression + rising close signal (0-11 pts)",            "Higher = quiet accumulation with upward drift")
    line("Up Vol Skew",   _fv(s.get("UpVolSkew"),1),          "Volume on up-days vs down-days (bonus signal)",                  "Higher = more volume on green days")
    line("OI Buildup",    _fv(s.get("OI_Buildup"),1),         "Open Interest rising during price coiling (F&O only)",           "Higher = smart money positioning for a move")
    line("Soft Penalty",  f"-{_fv(s.get('SoftPenalty',0),1)}","Sum of continuous penalties (overbought / illiquid / extended)", "Lower (closer to 0) = cleaner, fresher setup")
    line("Signal Coverage",f"{round((s.get('SignalPersist',1) or 1)*100,0):.0f}%",
         "Fraction of scoring signals with valid data",                                                                          "Low coverage = signal uncertainty; weight score accordingly")
    line("Liquidity Score",_fv(s.get("LiquidityScore"),3),    "ADV turnover sigmoid score (0-1)",                               "Higher = more liquid = easier fills, tighter stops")
    line("Stability",     _fv(s.get("Stability"),2),          "% of last 20 days that closed positive",                         ">0.6 = orderly trend; <0.4 = choppy/risky")
    line("Breakout Prob", f"{round((s.get('BreakoutProb',0) or 0)*100,1)}%", "Mean of BB/VDU/CLV/VCP/VC percentile scores",    "Higher = more factors aligned for imminent move")
    line("RSI",           _fv(s.get("RSI7"),1),               f"Relative Strength Index ({STATE.get('rsi_period',7)}-period)",  ">P90 overbought (adaptive), 40-60 ideal entry zone")
    line("ATR%",          f"{_fv(s.get('ATR%'),2)}%",         "Average True Range as % of price (14-day ATR)",                  "Use for stop sizing: stop = entry − 1-1.5× ATR")
    line("Vol Z-Score",   _fv(s.get("VolZ"),2),               "Today's volume vs 20-day avg in σ units",                        ">1.5σ = unusual accumulation; <-1σ = drying up")
    line("Vol Ratio",     _fv(s.get("VolRatio"),2),           "5-day avg volume / 20-day avg volume",                           ">1.5 = elevated activity; <0.7 = quiet coiling")
    line("Pos 52W",       f"{round((s.get('Pos52W',0) or 0)*100,1)}%", "Stock's position in its 52-week high-low range",        ">70% = near highs (strong trend); <30% = near lows")
    line("Above SMA200",  "YES" if s.get("AboveSMA200") else "NO", "Is LTP above 200-day simple moving average?",               "YES = long-term uptrend intact")
    line("EMA9",          f"₹{_fv(s.get('EMA9'),2)}",         "9-day Exponential Moving Average",                               "Fast trend line; acts as first support in uptrends")
    line("EMA20",         f"₹{_fv(s.get('EMA20'),2)}",        "20-day Exponential Moving Average",                              "Primary entry zone for pullback setups")
    line("EMA50",         f"₹{_fv(s.get('EMA50'),2)}",        "50-day Exponential Moving Average",                              "Stop reference for swing trades")
    line("Entry",         f"₹{_fv(s.get('Entry'),2)}",        "Suggested entry price based on setup type",                      "Buy at or just above this level on confirmation")
    line("Target",        f"₹{_fv(s.get('Target'),2)}",       "Price target based on ATR multiples + structure",                "Exit here to realise the expected gain")
    line("Stop Loss",     f"₹{_fv(s.get('Stop'),2)}",         "Stop-loss level — setup is invalidated below this",              "Exit immediately if price closes below stop")
    line("Risk:Reward",   f"{_fv(s.get('RR'),2)}x",           "Reward ÷ Risk ratio",                                            ">2.0x acceptable; >3.0x excellent; <1.5x skip")
    line("Kelly Size",    f"{round((s.get('KellyFrac',0) or 0)*100,1)}%", "Kelly criterion optimal position size",              "% of capital to risk; halve this for safety")
    line("Move%",         f"{_fv(s.get('Move%'),1)}%",        "Expected % move from entry to target",                           "Higher move% with good RR = better opportunity")
    line("Patterns",      s.get("Patterns","—"),               "Candlestick patterns detected on latest bar",                    "Confirm entry with reversal or continuation candle")

    regime = STATE.get("mkt", {}).get("regime", "BULL")
    vix    = STATE.get("mkt", {}).get("vix_level")
    line("Market Regime", regime,
         "BULL=Nifty>50DMA+rising, BEAR=below 50DMA, CHOP=mixed signals",
         "BULL: take all setups. CHOP: only Score>60. BEAR: Reversals only, reduce size.")
    if vix:
        line("India VIX", f"{vix:.1f}",
             "Fear/volatility index — higher = more uncertainty",
             "<14=calm. 14-20=normal. >20=elevated risk. >25=avoid new entries.")

    return {"symbol": sym, "explanation": lines,
            "score_summary": {"score": s.get("Score"), "setup": s.get("SetupType"),
                               "horizon": s.get("Horizon"), "regime": regime,
                               "total_factors": len(lines)}}

# ── Calibration snapshot ──────────────────────────────────────────
def _calibration_snapshot_bg() -> int:
    """Background worker: scores every cached stock and inserts into calibration DB.
    Separated from the route so the HTTP response returns immediately."""
    if not STATE["raw_data_cache"]: return 0
    mkt    = STATE.get("mkt") or {}
    regime = mkt.get("regime", "BULL")
    conn   = get_db(); logged = 0
    for sym, df_raw in dict(STATE["raw_data_cache"]).items():
        try:
            live   = STATE["live_quotes_cache"].get(normalize_key(STATE["targets"].get(sym, "")), {})
            result = score_stock_dual(sym, df_raw, live, mkt.get("nifty_r5"), mkt.get("nifty_r20"))
            if result is None: continue
            conn.execute(
                "INSERT INTO calibration(ts,symbol,score,forward_ret,horizon,setup,regime) VALUES(?,?,?,?,?,?,?)",
                (_dt.now().isoformat(), sym, result.get("Score"),
                 None, result.get("Horizon",""), result.get("SetupType",""), regime)
            )
            logged += 1
        except Exception:
            continue
    conn.commit()
    return logged

@app.post("/api/calibration/snapshot")
async def calibration_snapshot(background_tasks: BackgroundTasks):
    if not STATE["raw_data_cache"]: return {"error": "no data in cache"}
    # Fire the heavy scoring work in the background so the response returns instantly.
    # The old implementation ran score_stock_dual() for every stock synchronously,
    # which could block the request for 30+ seconds on a 500-stock universe.
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    logged = await loop.run_in_executor(None, _calibration_snapshot_bg)
    return {"logged": logged}

# ── Export ────────────────────────────────────────────────────────
@app.get("/api/export/screener")
async def export_screener_csv():
    if not STATE["raw_data_cache"]: raise HTTPException(404, "No data — run extraction first")
    rows = []
    cache_snap = dict(STATE["raw_data_cache"])
    score_snap = dict(STATE["score_cache"])
    lq_snap    = dict(STATE["live_quotes_cache"])
    tgt_snap   = dict(STATE["targets"])
    for sym, df_raw in cache_snap.items():
        cached = score_snap.get(sym, {}).get("result")
        if cached:
            live = lq_snap.get(normalize_key(tgt_snap.get(sym, "")), {})
            ltp  = live.get("ltp") or float(df_raw["close"].iloc[-1])
            rows.append({"Ticker": sym, "LTP": round(ltp, 2), **cached})
    if not rows: raise HTTPException(404, "No scored rows")
    df_out = pd.DataFrame(rows).replace({float("nan"): "", float("inf"): "", float("-inf"): ""})
    buf = _io.StringIO(); df_out.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.read()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=monarch_screener_{_dt.now().strftime('%Y%m%d_%H%M')}.csv"})

@app.get("/api/export/snapshot/{ts}")
async def export_snapshot_csv(ts: str):
    rows = get_db().execute("SELECT row_json FROM snapshots WHERE ts=?", (ts,)).fetchall()
    if not rows: raise HTTPException(404, "Snapshot not found")
    df_out = pd.DataFrame([_json.loads(r["row_json"]) for r in rows])
    buf = _io.StringIO(); df_out.to_csv(buf, index=False); buf.seek(0)
    ts_safe = ts.replace(":", "-").replace(".", "-")[:19]
    return StreamingResponse(iter([buf.read()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=snapshot_{ts_safe}.csv"})


# ═════════════════════════════════════════════════════════════════
# 19.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    try:
        import config as _c
        _port = _c.PORT; _host = _c.HOST
    except Exception:
        _port = 8000; _host = "0.0.0.0"
    uvicorn.run(app, host=_host, port=_port)
