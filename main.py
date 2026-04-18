"""
MONARCH PRO — FastAPI Backend
Replaces Streamlit. Same scoring engine, same UI logic.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests, gzip, json, time, io, urllib.parse
import pandas as pd
import numpy as np
import math
import sqlite3, pathlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import rankdata
import threading
import os

app = FastAPI(title="Monarch Pro")

# ══════════════════════════════════════════════════════════════════════════════
# SCORE CONFIGURATION — single source of truth for all magic numbers  [F-11]
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class _ScoreCfg:
    NSE_OPEN_MIN:   int   = 9 * 60 + 15
    NSE_CLOSE_MIN:  int   = 15 * 60 + 30
    ADV_THRESHOLD:  float = 2e7
    LIQ_CENTRE_LOG: float = math.log(5e7)
    RSI_OB_CAP:     float = 15.0
    VOL_LOW_CAP:    float = 12.0
    GAP_CAP:        float = 15.0
    SMA_CAP:        float = 20.0
    LIQ_CAP:        float = 15.0
    STAB_CAP:       float = 20.0
    EXT_CAP:        float = 15.0
    ALREADY_BO_CAP: float = 18.0
    BONUS_CAP:      float = 8.0
    SWEEP_CAP:      float = 4.0
    VCVE_CAP:       float = 3.0
    OI_CAP:         float = 3.0
    W_SPREAD:       float = 0.40
    W_VOL:          float = 0.40
    W_COIL:         float = 0.20
    MAX_SPREAD:     float = 11.0
    MAX_VOL:        float = 14.0
    MAX_COIL:       float = 10.0
    REG_HISTORY:    int   = 200
    MIN_BARS:       int   = 60

SCORE_CFG = _ScoreCfg()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── OAuth router (Upstox OTP login) ──────────────────────────────────────────
from upstox_auth import router as auth_router, init_state as _auth_init_state
app.include_router(auth_router)

# ── Feature routers ───────────────────────────────────────────────────────────
from routers.options      import router as options_router,      init_state as _options_init_state
from routers.fundamentals import router as fundamentals_router
from routers.ml           import router as ml_router,           init_state as _ml_init_state
app.include_router(options_router)
app.include_router(fundamentals_router)
app.include_router(ml_router)

# ── STATE ──
STATE = {
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
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [], "pullback_sigma": [],
        "stab_adj_scale": [], "stab_adj_obs": [], "pos52w_max": [],
    },
    "per_stock_winrate": {},
    "last_live_refresh": 0,
    "extraction_status": {"running": False, "done": 0, "total": 0, "errors": 0, "rate_limited": 0, "log": []},
    "mkt": {},
    "sector_returns": {},
    "sector_returns_10d": {},
    "top_sectors": set(),
    "rsi_period": 7,
    "min_avg_vol": 100000,
    "sector_cap_enabled": False,
}
STATE_LOCK = threading.Lock()

LIVE_REFRESH_SEC = 60

# Share STATE with all routers + auto-load saved token
_auth_init_state(STATE)
_options_init_state(STATE)
_ml_init_state(STATE)

# ── SECTOR MAPS ──
SECTOR_TICKERS = {
    "IT": "^CNXIT", "Bank": "^NSEBANK", "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA", "Metal": "^CNXMETAL", "Energy": "^CNXENERGY",
    "Infra": "^CNXINFRA", "FMCG": "^CNXFMCG", "Realty": "^CNXREALTY",
    "PSUBank": "^CNXPSUBANK", "Chemicals": "^CNXCHEMICALS",
    "ConsumerDur": "^CNXCONSUMER", "Insurance": "^CNXFINSERVICE",
    "Telecom": "^CNXTELECOM", "Retail": "^CNXCONSUMER", "Logistics": "^CNXINFRA",
}

STOCK_SECTOR_MAP = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT", "OFSS": "IT",
    "KPITTECH": "IT", "TATAELXSI": "IT", "MASTEK": "IT", "HEXAWARE": "IT",
    "HDFCBANK": "Bank", "ICICIBANK": "Bank", "KOTAKBANK": "Bank", "AXISBANK": "Bank",
    "INDUSINDBK": "Bank", "FEDERALBNK": "Bank", "IDFCFIRSTB": "Bank", "AUBANK": "Bank",
    "BAJFINANCE": "Bank", "BAJAJFINSV": "Bank", "RBLBANK": "Bank", "YESBANK": "Bank",
    "CSBBANK": "Bank", "DCBBANK": "Bank", "KARURVYSYA": "Bank",
    "SBIN": "PSUBank", "BANKBARODA": "PSUBank", "PNB": "PSUBank", "CANBK": "PSUBank",
    "UNIONBANK": "PSUBank", "BANKINDIA": "PSUBank", "MAHABANK": "PSUBank",
    "INDIANB": "PSUBank", "UCOBANK": "PSUBank", "CENTRALBK": "PSUBank",
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "HEROMOTOCO": "Auto", "EICHERMOT": "Auto", "TVSMOTORS": "Auto",
    "MOTHERSON": "Auto", "BOSCHLTD": "Auto", "BHARATFORG": "Auto", "BALKRISIND": "Auto",
    "APOLLOTYRE": "Auto", "MRF": "Auto", "CEATLTD": "Auto", "EXIDEIND": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "TORNTPHARM": "Pharma", "AUROPHARMA": "Pharma", "APOLLOHOSP": "Pharma",
    "LUPIN": "Pharma", "BIOCON": "Pharma", "ALKEM": "Pharma", "GLENMARK": "Pharma",
    "IPCALAB": "Pharma", "NATCOPHARM": "Pharma", "LAURUSLABS": "Pharma",
    "FORTIS": "Pharma", "METROPOLIS": "Pharma", "LALPATHLAB": "Pharma",
    "TATASTEEL": "Metal", "JSWSTEEL": "Metal", "HINDALCO": "Metal", "SAIL": "Metal",
    "VEDL": "Metal", "COALINDIA": "Metal", "NMDC": "Metal", "JINDALSTEL": "Metal",
    "APLAPOLLO": "Metal", "RATNAMANI": "Metal", "NATIONALUM": "Metal", "MOIL": "Metal",
    "ONGC": "Energy", "NTPC": "Energy", "POWERGRID": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "RELIANCE": "Energy", "HPCL": "Energy",
    "PETRONET": "Energy", "OIL": "Energy", "HINDPETRO": "Energy", "MGL": "Energy",
    "IGL": "Energy", "TATAPOWER": "Energy", "ADANIGREEN": "Energy", "ADANIENT": "Energy",
    "LT": "Infra", "ADANIPORTS": "Infra", "IRFC": "Infra", "RVNL": "Infra",
    "IRCON": "Infra", "NBCC": "Infra", "ULTRACEMCO": "Infra", "SHREECEM": "Infra",
    "AMBUJACEMENT": "Infra", "ACC": "Infra", "SIEMENS": "Infra", "ABB": "Infra",
    "BEL": "Infra", "HAL": "Infra", "BHEL": "Infra", "CUMMINSIND": "Infra",
    "THERMAX": "Infra", "KEC": "Infra", "KALPATPOWR": "Infra", "VOLTAS": "Infra",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "MARICO": "FMCG", "GODREJCP": "FMCG", "ASIANPAINT": "FMCG",
    "EMAMILTD": "FMCG", "COLPAL": "FMCG", "TATACONSUM": "FMCG", "UBL": "FMCG",
    "RADICO": "FMCG", "VBL": "FMCG",
    "DLF": "Realty", "LODHA": "Realty", "OBEROIRLTY": "Realty", "PHOENIXLTD": "Realty",
    "GODREJPROP": "Realty", "PRESTIGE": "Realty", "BRIGADE": "Realty", "SOBHA": "Realty",
    "PIDILITIND": "Chemicals", "SRF": "Chemicals", "DEEPAKNTR": "Chemicals",
    "AARTIIND": "Chemicals", "NAVINFLUOR": "Chemicals", "ALKYLAMINE": "Chemicals",
    "FINEORG": "Chemicals", "VINATIORGA": "Chemicals", "BALRAMCHIN": "Chemicals",
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance", "ICICIPRULI": "Insurance",
    "LICIHSGFIN": "Insurance", "MUTHOOTFIN": "Insurance", "CHOLAFIN": "Insurance",
    "ICICIGI": "Insurance", "NIACL": "Insurance", "GICRE": "Insurance",
    "HDFCAMC": "Insurance", "NAM-INDIA": "Insurance", "ABSLAMC": "Insurance",
    "BHARTIARTL": "Telecom", "IDEA": "Telecom", "TATACOMM": "Telecom", "INDUSTOWER": "Telecom",
    "HAVELLS": "ConsumerDur", "CROMPTON": "ConsumerDur", "TITAN": "ConsumerDur",
    "TRENT": "Retail", "DMART": "Retail",
    "CONCOR": "Logistics", "BLUEDART": "Logistics",
}


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def normalize_key(k: str) -> str:
    return k.replace("%7C", "|").replace(":", "|")

def to_ascending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[::-1].reset_index(drop=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    return df

def get_headers():
    return {"Authorization": f"Bearer {STATE['token']}", "Accept": "application/json"}

def rsi_wilder(close: pd.Series, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def atr14(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, adjust=False).mean()

def percentile_last(series: pd.Series, window: int):
    """CDF rank of last value vs window. O(n log n) via rankdata.  [F-02]"""
    s = series.tail(window).dropna()
    if len(s) < 2:
        return np.nan
    ranks = rankdata(s.values, method="average")
    return float(ranks[-1] / len(s))

def get_sector(ticker):
    return STOCK_SECTOR_MAP.get(ticker.upper(), None)

def get_sector_return(ticker):
    sect = get_sector(ticker)
    if sect:
        r5 = STATE["sector_returns"].get(sect)
        r10 = STATE["sector_returns_10d"].get(sect)
        if r5 is not None:
            return r5, r10, sect
    return None, None, None


# ─────────────────────────────────────────────
# DARVAS BOX
# ─────────────────────────────────────────────

def darvas_box_score(df: pd.DataFrame, atr_v: float) -> dict:
    null = {"darvas_score": 0.0, "box_high": np.nan, "box_low": np.nan,
            "in_box": False, "bars_in_box": 0, "box_atr_ratio": np.nan}
    if len(df) < 20:
        return null
    hh, hl, hc = df["high"], df["low"], df["close"]
    _tr = pd.concat([(hh - hl), (hh - hc.shift(1)).abs(), (hl - hc.shift(1)).abs()], axis=1).max(axis=1)
    _atr = float(_tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]) if len(_tr) >= 5 else atr_v
    _daily_move = float(_tr.tail(20).median()) if len(_tr) >= 20 else (_atr * 0.5)
    _bars_per_atr = max(3, int(_atr / (_daily_move + 1e-9)))
    _box_confirm_n = int(np.clip(_bars_per_atr, 3, 10))
    _box_high = _box_low = _box_start_idx = None
    for i in range(len(hh) - 1, _box_confirm_n * 2, -1):
        _ws = max(0, i - _box_confirm_n)
        _candidate_high = float(hh.iloc[_ws:i+1].max())
        _cs = i + 1
        _ce = min(len(hh), _cs + _box_confirm_n)
        if _ce > len(hh):
            continue
        if float(hh.iloc[_cs:_ce].max()) > _candidate_high:
            continue
        _box_high = _candidate_high
        _box_low = float(hl.iloc[_ws:].min())
        _box_start_idx = _ws
        break
    if _box_high is None:
        return null
    _ltp = float(hc.iloc[-1])
    _box_width = _box_high - _box_low + 1e-9
    _bars_in_box = len(hh) - _box_start_idx
    _in_box = (_ltp <= _box_high) and (_ltp >= _box_low)
    _box_atr_ratio = _box_width / (atr_v + 1e-9)
    _tightness = float(1.0 / (1.0 + _box_atr_ratio))
    _pos_in_box = (_ltp - _box_low) / _box_width
    _pos_score = float(np.clip(1.0 - 4.0 * (_pos_in_box - 0.5) ** 2, 0.0, 1.0))
    _n = _box_confirm_n
    _roll_max_cur = hh.rolling(_n).max()
    _roll_max_prev = hh.rolling(_n).max().shift(_n)
    _coiling_mask = (_roll_max_cur <= _roll_max_prev).fillna(False)
    _coiling_frac = float(_coiling_mask.tail(60).mean()) if len(_coiling_mask) >= 10 else 0.5
    _typical_box_dur = max(int(_coiling_frac * 20), 3)
    _time_score = float(np.clip(_bars_in_box / (_typical_box_dur * 2.0 + 1e-9), 0.0, 1.0))
    darvas_score = round((_tightness * 0.40 + _pos_score * 0.35 + _time_score * 0.25) * 10.0, 1)
    return {"darvas_score": darvas_score, "box_high": round(_box_high, 2), "box_low": round(_box_low, 2),
            "in_box": _in_box, "bars_in_box": _bars_in_box, "box_atr_ratio": round(_box_atr_ratio, 2)}


# ─────────────────────────────────────────────
# CANDLE PATTERNS
# ─────────────────────────────────────────────

def detect_candle_patterns(o, h, l, c, prev_o, prev_h, prev_l, prev_c):
    body = abs(c - o); rng = h - l + 1e-9
    upper_w = h - max(o, c); lower_w = min(o, c) - l
    prev_body = abs(prev_c - prev_o); prev_rng = prev_h - prev_l + 1e-9
    patterns = []; pts = 0
    if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
        patterns.append("Engulfing"); pts += 3
    if lower_w >= 2 * body and upper_w <= 0.4 * rng and c > o:
        patterns.append("Hammer"); pts += 2.5
    if h <= prev_h and l >= prev_l:
        patterns.append("InsideBar"); pts += 1.5
    if h > prev_h and l < prev_l and c > o and c > (h + l) / 2:
        patterns.append("OutsideBar"); pts += 2
    if body / rng > 0.60 and c > o and (c - l) / rng > 0.75:
        patterns.append("StrongGreen"); pts += 2
    if body / rng < 0.10 and lower_w > 1.5 * upper_w:
        patterns.append("BullDoji"); pts += 1
    if prev_c < prev_o and prev_body / prev_rng > 0.5 and c > o and c > (prev_o + prev_c) / 2:
        patterns.append("MorningStar"); pts += 2.5
    _gap_min = max(prev_body * 0.5, (prev_rng * 0.01))
    if o > prev_c + _gap_min and c > o:
        patterns.append("GapContinue"); pts += 2
    return min(pts, 10), patterns


# ─────────────────────────────────────────────
# CONSOLIDATION & BB WIDTH & VCP & CLV
# ─────────────────────────────────────────────

def consolidation_score(df, window=15):
    if len(df) < window * 2:
        return 0.0
    base = df.iloc[-window:]
    prior = df.iloc[-(window*2):-window]
    base_rng = (base["high"].max() - base["low"].min())
    prior_rng = (prior["high"].max() - prior["low"].min())
    if prior_rng == 0:
        return 0.0
    compression = 1.0 - (base_rng / (prior_rng + 1e-9))
    flat_hi = base["high"].std() / (base["high"].mean() + 1e-9)
    flat_lo = base["low"].std() / (base["low"].mean() + 1e-9)
    cv_avg = (flat_hi + flat_lo) / 2.0
    if len(df) >= window * 4:
        _cv_hist = pd.Series([
            ((df["high"].iloc[max(0,i-window):i].std() / (df["high"].iloc[max(0,i-window):i].mean() + 1e-9)) +
             (df["low"].iloc[max(0,i-window):i].std()  / (df["low"].iloc[max(0,i-window):i].mean()  + 1e-9))) / 2.0
            for i in range(window, min(window*4, len(df)))
        ], dtype=float).dropna()
        if len(_cv_hist) >= 5:
            _cv_centre = float(_cv_hist.median()); _cv_sigma = float(_cv_hist.std())
            _cv_k = 1.0 / max(_cv_sigma, 0.002)
        else:
            _cv_centre, _cv_k = 0.01, 200.0
    else:
        _cv_centre, _cv_k = 0.01, 200.0
    flatness = float(1.0 / (1.0 + np.exp(_cv_k * (cv_avg - _cv_centre))))
    return max(0.0, min(1.0, (compression * 0.6 + flatness * 0.4)))

def bb_width_compression_score(c_series: pd.Series, window: int = 20, mult: float = 2.0):
    if len(c_series) < window + 10:
        return 0.5, 0.5
    sma = c_series.rolling(window).mean()
    std = c_series.rolling(window).std()
    bb_width = (2.0 * mult * std) / (sma.replace(0, np.nan))
    bb_width = bb_width.dropna()
    if len(bb_width) < 10:
        return 0.5, 0.5
    current_bw = float(bb_width.iloc[-1])
    hist_bw = bb_width.iloc[:-1]
    bb_width_pct = float((hist_bw <= current_bw).mean())
    return round(bb_width_pct, 4), round(1.0 - bb_width_pct, 4)

def volume_dryup_score(v_series, window_short=5, window_long=20):
    if len(v_series) < window_long + window_short:
        return 1.0, 0.5
    v = v_series.replace(0, np.nan).dropna()
    if len(v) < window_long:
        return 1.0, 0.5
    short_avg = float(v.tail(window_short).mean())
    long_avg = float(v.tail(window_long).mean())
    dryup_ratio = short_avg / (long_avg + 1e-9)
    v_hist = v.iloc[:-1]
    if len(v_hist) < window_long:
        return round(dryup_ratio, 4), round(float(np.clip(1.0 - dryup_ratio, 0.0, 1.0)), 4)
    _short_roll = v_hist.rolling(window_short).mean()
    _long_roll = v_hist.rolling(window_long).mean()
    _ratio_hist = (_short_roll / (_long_roll + 1e-9)).dropna()
    if len(_ratio_hist) < 5:
        return round(dryup_ratio, 4), round(float(np.clip(1.0 - dryup_ratio, 0.0, 1.0)), 4)
    pct_rank = float((_ratio_hist <= dryup_ratio).mean())
    return round(dryup_ratio, 4), round(1.0 - pct_rank, 4)

def clv_accumulation_score(c_series, h_series, l_series, v_series, window=20):
    if len(c_series) < window + 5:
        return 0.0, 0.5
    hl_range = (h_series - l_series).replace(0, np.nan)
    clv = ((c_series - l_series) - (h_series - c_series)) / hl_range
    clv = clv.fillna(0.0)
    mf = clv * v_series
    avg_v = v_series.rolling(window).mean().replace(0, np.nan)
    mf_norm = mf / avg_v
    roll_mf = mf_norm.rolling(window).sum().dropna()
    if len(roll_mf) < 5:
        return float(clv.iloc[-1]) if pd.notna(clv.iloc[-1]) else 0.0, 0.5
    current_mf = float(roll_mf.iloc[-1])
    hist_mf = roll_mf.iloc[:-1]
    accum_pct = float((hist_mf <= current_mf).mean())
    return round(current_mf, 4), round(accum_pct, 4)

def detect_vcp(c, h, l, v, atr):
    _NEUTRAL = {"vcp_score": 0.0, "vcp_pullback_n": 0, "vcp_contraction": 0.5,
                "vcp_vol_comp": 0.5, "vcp_vol_dryup": 0.5, "vcp_tightness": 0.5,
                "vcp_position": 0.5, "vcp_detected": False}
    if len(c) < 60:
        return _NEUTRAL
    hist_c, hist_h, hist_l, hist_v, hist_atr = c.iloc[:-1], h.iloc[:-1], l.iloc[:-1], v.iloc[:-1], atr.iloc[:-1]
    daily_range = (hist_h - hist_l).replace(0, np.nan).dropna()
    if len(daily_range) < 20:
        return _NEUTRAL
    median_range = float(daily_range.tail(60).median())
    current_atr = float(hist_atr.dropna().tail(1).iloc[0]) if len(hist_atr.dropna()) >= 1 else median_range
    if median_range <= 0 or current_atr <= 0:
        return _NEUTRAL
    atr_cycle_ratio = current_atr / (median_range + 1e-9)
    swing_window = max(3, min(25, int(round(float(np.clip(atr_cycle_ratio * 5.0, 3.0, 25.0))))))
    n = len(hist_c)
    if n < swing_window * 4:
        return _NEUTRAL
    roll_max = hist_h.rolling(2 * swing_window + 1, center=True).max()
    roll_min = hist_l.rolling(2 * swing_window + 1, center=True).min()
    sh_idx = hist_h.index[hist_h == roll_max].tolist()
    sl_idx = hist_l.index[hist_l == roll_min].tolist()
    pullbacks = []
    for shi in sh_idx:
        sh_pos = hist_h.index.get_loc(shi)
        sh_val = float(hist_h.loc[shi])
        subsequent_lows = [sli for sli in sl_idx if hist_l.index.get_loc(sli) > sh_pos]
        if not subsequent_lows:
            continue
        sli = subsequent_lows[0]
        sl_val = float(hist_l.loc[sli])
        if sh_val <= 0:
            continue
        depth = (sh_val - sl_val) / (sh_val + 1e-9)
        if depth < 0 or depth > 0.99:
            continue
        pullbacks.append({"depth": depth, "sh_pos": sh_pos, "sl_pos": hist_l.index.get_loc(sli), "sh_val": sh_val, "sl_val": sl_val})
    if len(pullbacks) < 2:
        return _NEUTRAL
    all_depths = [p["depth"] for p in pullbacks]
    recent_pbs = [p for p in pullbacks if p["sh_pos"] >= n - 60]
    if len(recent_pbs) < 2:
        recent_pbs = pullbacks[-min(4, len(pullbacks)):]
    recent_depths = [p["depth"] for p in recent_pbs]
    if len(recent_depths) < 2:
        return _NEUTRAL
    x = np.arange(len(recent_depths), dtype=float)
    slope, _ = np.polyfit(x, recent_depths, 1) if len(recent_depths) >= 2 else (0.0, 0.0)
    all_slopes = []
    for i in range(2, len(all_depths)):
        _x = np.arange(i, dtype=float)
        _s, _ = np.polyfit(_x, all_depths[:i], 1)
        all_slopes.append(_s)
    if len(all_slopes) >= 3:
        contraction_pct = float((np.array(all_slopes) >= slope).mean())
        contraction_score = 1.0 - contraction_pct
    else:
        contraction_score = float(np.clip((-slope / (np.std(all_depths) + 1e-9)), 0, 1)) if np.std(all_depths) > 0 else 0.5
    _, vol_comp_score = bb_width_compression_score(hist_c, window=min(20, len(hist_c)//3))
    _, vol_dryup_score_val = volume_dryup_score(hist_v)
    last_sh = recent_pbs[-1]["sh_val"]
    last_sl = recent_pbs[-1]["sl_val"]
    consolidation_range = last_sh - last_sl + 1e-9
    if len(hist_atr.dropna()) >= 1:
        _atr_cons = float(hist_atr.dropna().iloc[-1])
    else:
        _atr_cons = consolidation_range * 0.5
    tightness_ratio = consolidation_range / (_atr_cons + 1e-9)
    _tight_hist = []
    for i in range(len(recent_pbs)-1, -1, -1):
        _pb = recent_pbs[i]
        _r = _pb["sh_val"] - _pb["sl_val"]
        _atr_at = float(hist_atr.iloc[min(_pb["sh_pos"], len(hist_atr)-1)]) if _pb["sh_pos"] < len(hist_atr) else _atr_cons
        _tight_hist.append(_r / (_atr_at + 1e-9))
    if len(_tight_hist) >= 3:
        tightness_score = float(np.clip(1.0 - tightness_ratio / (np.percentile(_tight_hist, 75) + 1e-9), 0.0, 1.0))
    else:
        tightness_score = float(np.clip(1.0 - tightness_ratio / 3.0, 0.0, 1.0))
    ltp_now = float(c.iloc[-1])
    position_score = float(np.clip((ltp_now - last_sl) / (consolidation_range), 0.0, 1.0))
    sub_scores = [contraction_score, vol_comp_score, vol_dryup_score_val, tightness_score, position_score]
    vcp_score = float(np.clip(float(np.mean(sub_scores)), 0.0, 1.0))
    vcp_detected = (vcp_score >= 0.55 and len(recent_pbs) >= 2 and contraction_score >= 0.4)
    return {"vcp_score": round(vcp_score, 3), "vcp_pullback_n": len(recent_pbs),
            "vcp_contraction": round(contraction_score, 3), "vcp_vol_comp": round(vol_comp_score, 3),
            "vcp_vol_dryup": round(vol_dryup_score_val, 3), "vcp_tightness": round(tightness_score, 3),
            "vcp_position": round(position_score, 3), "vcp_detected": vcp_detected}


# ─────────────────────────────────────────────
# RELATIVE STRENGTH HELPERS
# ─────────────────────────────────────────────

def volume_surge(v_today, v_series, window=20):
    """Returns ratio of today's volume to N-day average."""
    avg = v_series.tail(window).mean()
    if avg == 0:
        return 0.0
    return float(v_today) / float(avg)

def _robust_iqr_width(s: pd.Series, fallback: float = 1.5) -> float:
    """Robust σ estimator via IQR."""
    s = s.dropna()
    if len(s) < 10:
        return fallback
    q75, q25 = np.percentile(s, 75), np.percentile(s, 25)
    return float(max((q75 - q25) / 1.35, 0.3))

def relative_strength(c_series, nifty_r5, nifty_r20, window5=5, window20=20, regime="BULL"):
    """Absolute vol-normalised alpha vs Nifty. Returns 0-1 score."""
    if len(c_series) < 23:
        return 0.5
    base_6  = float(c_series.iloc[-7])  if len(c_series) >= 7  else 0
    base_21 = float(c_series.iloc[-22]) if len(c_series) >= 22 else 0
    end_p   = float(c_series.iloc[-2])
    stock_r5  = float(end_p / base_6  - 1) if base_6  != 0 else 0
    stock_r20 = float(end_p / base_21 - 1) if base_21 != 0 else 0
    r5_beat   = stock_r5  - (nifty_r5  or 0)
    r20_beat  = stock_r20 - (nifty_r20 or 0)
    daily_rets = c_series.pct_change().dropna()
    ret_std_5  = float(daily_rets.tail(5).std())  if len(daily_rets) >= 5  else 0.01
    ret_std_20 = float(daily_rets.tail(20).std()) if len(daily_rets) >= 20 else 0.01
    ret_std_5  = max(ret_std_5,  0.001)
    ret_std_20 = max(ret_std_20, 0.001)
    alpha_5  = r5_beat  / (ret_std_5  * np.sqrt(5)  + 1e-9)
    alpha_20 = r20_beat / (ret_std_20 * np.sqrt(20) + 1e-9)
    _alpha5_hist  = (c_series.pct_change(5).dropna()  - (nifty_r5  or 0)) / (ret_std_5  * np.sqrt(5)  + 1e-9)
    _alpha20_hist = (c_series.pct_change(20).dropna() - (nifty_r20 or 0)) / (ret_std_20 * np.sqrt(20) + 1e-9)
    _w5  = _robust_iqr_width(_alpha5_hist)
    _w20 = _robust_iqr_width(_alpha20_hist)
    rs5  = float(0.5 * (1.0 + np.tanh(alpha_5  / _w5)))
    rs20 = float(0.5 * (1.0 + np.tanh(alpha_20 / _w20)))
    if regime == "BULL":
        return rs5 * 0.25 + rs20 * 0.75
    elif regime == "BEAR":
        return rs5 * 0.15 + rs20 * 0.85
    else:
        return rs5 * 0.20 + rs20 * 0.80


# ─────────────────────────────────────────────
# MAIN SCORING ENGINE  (exact match: 1_Live_screener.py v6/v7)
# ─────────────────────────────────────────────

def score_stock_dual(ticker: str, df: pd.DataFrame, live: dict, nifty_r5, nifty_r20):
    """
    Live scoring path — mirrors score_stock_dual in 1_Live_screener.py exactly.
    st.session_state → STATE (thread-safe via STATE_LOCK reads).
    bt_mode parameters are not needed here (live path only).
    """
    if len(df) < 60:
        return None

    df = df.copy()

    # ── LIVE PATCH ──
    _live_ltp = live.get("ltp");   ltp     = float(_live_ltp if _live_ltp is not None else df["close"].iloc[-1])
    _live_vol = live.get("volume"); day_vol = float(_live_vol if _live_vol is not None else df["volume"].iloc[-1])
    _live_hi  = live.get("high");   day_hi  = float(_live_hi  if _live_hi  is not None else df["high"].iloc[-1])
    _live_lo  = live.get("low");    day_lo  = float(_live_lo  if _live_lo  is not None else df["low"].iloc[-1])
    _live_o   = live.get("open");   day_o   = float(_live_o   if _live_o   is not None else df["open"].iloc[-1])

    df.at[df.index[-1], "close"] = ltp
    df.at[df.index[-1], "high"]  = max(float(df["high"].iloc[-1]),  day_hi)
    df.at[df.index[-1], "low"]   = min(float(df["low"].iloc[-1]),   day_lo)

    # ── VOLUME SCALING (intraday extrapolation) ──
    _NSE_OPEN_MIN  = 9 * 60 + 15
    _NSE_CLOSE_MIN = 15 * 60 + 30
    _SESSION_MINS  = _NSE_CLOSE_MIN - _NSE_OPEN_MIN
    _now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    _now_min = _now_ist.hour * 60 + _now_ist.minute
    _elapsed = _now_min - _NSE_OPEN_MIN
    _elapsed_frac = float(np.clip(_elapsed / _SESSION_MINS, 0.10, 1.0))
    if _elapsed >= 6 and _elapsed < _SESSION_MINS:
        day_vol_scaled = day_vol / _elapsed_frac
    else:
        day_vol_scaled = day_vol
    df.at[df.index[-1], "volume"] = day_vol_scaled

    c = df["close"]; h = df["high"]; l = df["low"]
    v = df["volume"]; o = df["open"]
    day_vol = day_vol_scaled

    # ── HISTORICAL SLICE (T-1) — no look-ahead ──
    hist = df.iloc[:-1]
    hc = hist["close"]; hh = hist["high"]; hl = hist["low"]; hv = hist["volume"]

    e9   = ema(hc, 9);  e20 = ema(hc, 20);  e50 = ema(hc, 50)
    e5   = ema(hc, 5)
    atr  = atr14(hist)
    _rsi_period = int(STATE.get("rsi_period", 7))
    rsi  = rsi_wilder(hc, _rsi_period)

    e9_v  = float(e9.iloc[-1]);  e20_v = float(e20.iloc[-1])
    e50_v = float(e50.iloc[-1]); atr_v = float(atr.iloc[-1])
    rsi_v = float(rsi.iloc[-1]); rsi_p = float(rsi.iloc[-2])
    e9_y  = float(e9.iloc[-2]);  e20_y = float(e20.iloc[-2])

    ltp_score = float(hc.iloc[-1])   # T-1 close = scoring price

    vol_ma20 = float(hv.rolling(20).mean().iloc[-1]) if len(hv) >= 20 else float(hv.mean())
    atr_pct  = (atr_v / ltp_score) * 100 if ltp_score > 0 else 0

    if atr_v == 0 or vol_ma20 == 0:
        return None
    if ltp_score <= 0:
        return None

    _soft_penalty = 0.0

    # ── RSI overbought penalty ──
    _rsi_hist_full = rsi_wilder(hc, _rsi_period)
    _rsi_p90       = float(_rsi_hist_full.tail(60).quantile(0.90)) if len(_rsi_hist_full) >= 20 else 80.0
    if rsi_v > _rsi_p90:
        _rsi_ob_z      = (rsi_v - _rsi_p90) / max(float(_rsi_hist_full.tail(20).std()), 1.0)
        _soft_penalty += float(np.clip(8.0 * np.tanh(_rsi_ob_z), 0.0, 15.0))

    # ── Low-volume penalty ──
    _vol_p05   = float(hv.tail(60).quantile(0.05)) if len(hv) >= 20 else vol_ma20 * 0.10
    _prev_vol  = float(hv.iloc[-1])
    if _prev_vol < _vol_p05:
        _vol_low_z     = (_vol_p05 - _prev_vol) / (max(float(hv.tail(20).std()), 1.0))
        _soft_penalty += float(np.clip(6.0 * np.tanh(_vol_low_z), 0.0, 12.0))

    # ── Low ATR% penalty ──
    if atr_pct < 0.5:
        _soft_penalty += float(np.clip((0.5 - atr_pct) * 10.0, 0.0, 5.0))

    # ── Turnover liquidity penalty ──
    _price_for_adv = float(hc.tail(20).median()) if len(hc) >= 20 else ltp_score
    _adv_turnover  = vol_ma20 * _price_for_adv
    _ADV_THRESHOLD = 2e7
    if _adv_turnover < _ADV_THRESHOLD:
        _liq_ratio     = _adv_turnover / (_ADV_THRESHOLD + 1e-9)
        _soft_penalty += float(np.clip(15.0 * (1.0 - _liq_ratio), 0.0, 15.0))

    # ── Gap penalty ──
    prev_close = float(hc.iloc[-1])
    prev_open  = float(hist["open"].iloc[-1]) if "open" in hist.columns else prev_close
    if prev_close > 0 and atr_v > 0:
        _gap = prev_open - float(hc.iloc[-2]) if len(hc) >= 2 else 0.0
        _gap_hist_atr = ((hh.shift(1) - hc.shift(1)).abs() / (atr + 1e-9)).dropna().tail(60)
        _gap_p90 = float(_gap_hist_atr.quantile(0.90)) if len(_gap_hist_atr) >= 20 else 2.0
        _gap_abs_atr = abs(_gap) / (atr_v + 1e-9)
        if _gap_abs_atr > _gap_p90:
            _gap_excess    = _gap_abs_atr - _gap_p90
            _soft_penalty += float(np.clip(8.0 * np.tanh(_gap_excess / (_gap_p90 + 1e-9)), 0.0, 15.0))

    # ── vol_pts guaranteed initialisation — no dir() hack  [F-05] ──
    vol_pts = 0.0

    # ── BASE RANGE ──
    base_hi  = float(hh.tail(20).max())
    base_lo  = float(hl.tail(20).min())
    base_rng = base_hi - base_lo + 1e-9
    breakout_ext = (ltp_score - base_hi) / (atr_v + 1e-9)

    # ── SMA200 ATR-normalised penalty ──
    _n_bars_sma  = min(200, len(hc))
    _sma200      = float(hc.tail(_n_bars_sma).mean())
    _sma200_gap_atr = (ltp_score - _sma200) / (atr_v + 1e-9)
    if _sma200_gap_atr < -0.5:
        _sma_excess    = abs(_sma200_gap_atr + 0.5)
        _soft_penalty += float(np.clip(10.0 * np.tanh(_sma_excess / 2.0), 0.0, 20.0))
    above_long_trend = ltp_score > _sma200 - 0.5 * atr_v

    # ── EMA proximity ──
    above_ema50  = ltp_score > e50_v - atr_v
    near_e9_y    = abs(ltp_score - e9_v)  / (atr_v + 1e-9) < 1.0
    near_e20_y   = abs(ltp_score - e20_v) / (atr_v + 1e-9) < 1.0

    # ── Volume distribution ──
    vol_series_20 = hv.tail(20) if len(hv) >= 5 else hv
    vol_mu        = float(vol_series_20.mean()) if len(vol_series_20) > 0 else float(vol_ma20)
    vol_sigma     = float(vol_series_20.std())  if len(vol_series_20) > 1 else vol_mu * 0.3
    vol_sigma     = max(vol_sigma, vol_mu * 0.05)
    _vol_q_hist   = hv.tail(min(60, len(hv))) if len(hv) >= 10 else hv
    vol_bo_thresh = float(_vol_q_hist.quantile(0.85))

    # ── Setup extension history ──
    _ext_hist = ((hc - hh.rolling(20).max().shift(1)) / (atr.iloc[:-1] + 1e-9)).dropna()
    _ext_p10  = float(_ext_hist.quantile(0.10)) if len(_ext_hist) >= 20 else -1.5
    _ext_p90  = float(_ext_hist.quantile(0.90)) if len(_ext_hist) >= 20 else  0.3
    _ext_p10  = min(_ext_p10, -0.3)
    _ext_p90  = max(_ext_p90,  0.5)

    # ── REVERSAL detection ──
    _vol3 = hv.tail(3)
    _t1_vol_ratio_rev = float(_vol3.max()) / (vol_ma20 + 1e-9)
    _hi10d_rev        = float(hh.tail(10).max())
    _washout_depth    = (_hi10d_rev - ltp_score) / (atr_v + 1e-9)
    _t1_bar_range = float(hh.iloc[-1]) - float(hl.iloc[-1])
    _t1_close_pos = ((float(hc.iloc[-1]) - float(hl.iloc[-1])) / (_t1_bar_range + 1e-9))
    _is_reversal = (rsi_v < 40.0 and _t1_vol_ratio_rev >= 1.3 and _washout_depth >= 1.5)

    if _is_reversal:
        setup_type = "Reversal"
    elif breakout_ext >= _ext_p10 and breakout_ext <= _ext_p90 and day_vol >= vol_bo_thresh:
        setup_type = "Breakout"
    elif above_ema50 and (near_e9_y or near_e20_y):
        _hi10d       = float(hh.tail(10).max())
        _real_pullback = ltp_score < (_hi10d - 0.3 * atr_v)
        setup_type = "Pullback" if _real_pullback else "Breakout"
    elif breakout_ext > _ext_p90:
        setup_type = "Breakout"
        _ext_excess    = (breakout_ext - _ext_p90) / (max(_ext_p90 - _ext_p10, 0.5))
        _soft_penalty += float(np.clip(10.0 * np.tanh(_ext_excess), 0.0, 15.0))
    elif breakout_ext >= _ext_p10 and breakout_ext <= _ext_p90:
        setup_type = "Breakout"
    else:
        setup_type = "Pullback"

    # ── Already-broke-out penalty ──
    if setup_type == "Breakout" and ltp_score >= base_hi - 0.2 * atr_v:
        _t1_vol     = float(hv.iloc[-1])
        _vol_rank   = float((hv.iloc[:-1] <= _t1_vol).mean())
        if _vol_rank >= 0.85:
            _already_broke_z = (_vol_rank - 0.85) / 0.15
            _soft_penalty   += float(np.clip(12.0 * np.tanh(_already_broke_z * 2.0), 0.0, 18.0))

    # ── Parameter registry — in-place, no copy overhead  [F-09] ──
    _reg = STATE.setdefault("param_registry", {
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [], "pullback_sigma": [],
        "stab_adj_scale": [], "stab_adj_obs": [], "pos52w_max": [],
    })

    def _tanh_w(z_series, fallback=None):
        s = z_series.dropna()
        _reg_vals = _reg.setdefault("tanh_w", [])
        _fallback = float(np.median(_reg_vals)) if len(_reg_vals) >= 5 else 1.5
        if len(s) < 10:
            return _fallback
        q75, q25 = np.percentile(s, 75), np.percentile(s, 25)
        w = float(max((q75 - q25) / 1.35, 0.3))
        _reg_vals.append(w)
        _reg["tanh_w"] = _reg_vals[-SCORE_CFG.REG_HISTORY:]
        return w

    # ── Market context ──
    _regime                = STATE["mkt"].get("regime", "BULL")
    _resolved_sect_ret     = STATE.get("sector_returns", {})
    _resolved_sect_ret_10d = STATE.get("sector_returns_10d", {})
    _resolved_vix_level    = STATE["mkt"].get("vix_level")
    _resolved_vix_falling  = STATE["mkt"].get("vix_falling", True)
    _resolved_vix_median   = STATE["mkt"].get("vix_median", 14.5)
    _resolved_vix_sigma    = STATE["mkt"].get("vix_sigma",  4.5)
    _resolved_above_20dma  = STATE["mkt"].get("nifty_above_20dma", True)

    # ── Regime weights ──
    if _regime == "BULL":
        _w5d, _w20d = 0.25, 0.75
    elif _regime == "BEAR":
        _w5d, _w20d = 0.15, 0.85
    else:
        _w5d, _w20d = 0.20, 0.80

    # ═══ F1 — RS vs UNIVERSE + RS vs NIFTY (0-15 pts) ═══
    _cs5  = STATE["cs_rs_5d"].get(ticker,  None)
    _cs20 = STATE["cs_rs_20d"].get(ticker, None)
    if _cs5 is not None and _cs20 is not None:
        cs_rs_score = _cs5 * _w5d + _cs20 * _w20d
    else:
        cs_rs_score = _cs5 if _cs5 is not None else (_cs20 if _cs20 is not None else 0.5)
    _rs_slope_score = float(np.clip(0.5 + ((_cs5 or 0.5) - (_cs20 or 0.5)) * 2.0, 0.0, 1.0))
    abs_rs_score = relative_strength(hc, nifty_r5, nifty_r20, regime=_regime)
    rs_score = cs_rs_score * 0.40 + _rs_slope_score * 0.35 + abs_rs_score * 0.25
    rs_pts   = round(rs_score * 15, 1)

    # RS acceleration
    _velocity      = e5 - e20
    _acceleration  = _velocity.diff()
    _acc_hist      = _acceleration.iloc[:-1]
    _acc_z_series  = (_acc_hist - _acc_hist.mean()) / (_acc_hist.std() + 1e-9)
    _acc_w         = _tanh_w(_acc_z_series)
    _acc_today_z   = float((_acceleration.iloc[-2] - _acc_hist.mean()) / (_acc_hist.std() + 1e-9)) \
                     if len(_acceleration) >= 2 else 0.0
    acc_score      = float(0.5 * (1.0 + np.tanh(_acc_today_z / _acc_w)))
    rs_accel_bonus = round(acc_score * 4, 1)
    rs_accel       = float(_acceleration.iloc[-2]) if len(_acceleration) >= 2 else 0.0

    # RS divergence bonus
    _rs_div = (_cs5 - _cs20) if (_cs5 is not None and _cs20 is not None) else 0.0
    _rs_div_hist = STATE.get("rs_div_hist", {})
    _prev_divs   = _rs_div_hist.get(ticker, [])
    _prev_divs   = (_prev_divs + [_rs_div])[-60:]
    _rs_div_hist[ticker] = _prev_divs
    STATE["rs_div_hist"] = _rs_div_hist
    if len(_prev_divs) >= 10:
        _div_arr = np.array(_prev_divs)
        _div_pct = float((_div_arr <= _rs_div).mean())
        rs_div_bonus = round(_div_pct * 3, 1)
    else:
        rs_div_bonus = 1.5

    # ═══ F2 — RS vs SECTOR (0-10 pts) ═══
    sect_ret, sect_ret_10d, sect_name = (None, None, None)
    if ticker:
        sect = get_sector(ticker)
        if sect:
            r5  = _resolved_sect_ret.get(sect)
            r10 = _resolved_sect_ret_10d.get(sect)
            if r5 is not None:
                sect_ret, sect_ret_10d, sect_name = r5, r10, sect
    _resolved_top_sectors = {k for k,_ in sorted(_resolved_sect_ret.items(), key=lambda x:x[1], reverse=True)[:3]} if _resolved_sect_ret else set()
    if sect_ret is not None and len(hc) >= 7:
        _hc_base = float(hc.iloc[-6])
        stock_r5 = float(hc.iloc[-1] / _hc_base - 1) if _hc_base != 0 else 0.0
        _daily_rets   = hc.pct_change().dropna()
        _stock_5d_vol = float(_daily_rets.tail(20).std() * np.sqrt(5)) if len(_daily_rets) >= 10 else 0.02
        _stock_5d_vol = max(_stock_5d_vol, 0.005)
        sect_beat = stock_r5 - sect_ret
        _sb_z     = sect_beat / _stock_5d_vol
        rs_sect_sc = float(0.5 * (1.0 + np.tanh(_sb_z)))
        _n_s = 0
        if _resolved_sect_ret:
            _all_sect_vals = sorted(_resolved_sect_ret.values())
            _n_s = len(_all_sect_vals)
            if _n_s > 1:
                _sect_rank_pct = sum(1 for v_s in _all_sect_vals if v_s <= sect_ret) / _n_s
                rs_sect_sc = min(1.0, rs_sect_sc + 0.15 * _sect_rank_pct)
        if sect_ret_10d is not None:
            _sect_accel = sect_ret - sect_ret_10d
            _sect_cross_vol = float(pd.Series(list(_resolved_sect_ret.values())).std()) \
                              if _n_s > 2 else max(abs(sect_ret) * 0.5, 1e-4)
            _sect_cross_vol = max(_sect_cross_vol, 1e-4)
            _accel_z = _sect_accel / _sect_cross_vol
            rs_sect_sc = float(np.clip(rs_sect_sc + 0.10 * np.tanh(_accel_z), 0.0, 1.0))
        rs_sect_pts = round(rs_sect_sc * 10, 1)
    else:
        rs_sect_pts = 0.0
        sect_name   = "?"

    # ═══ F3 — VOLUME (0-15 pts) ═══
    vol_ratio = volume_surge(float(hv.iloc[-1]), hv.iloc[:-1])
    vol_z     = (float(hv.iloc[-1]) - vol_mu) / (vol_sigma + 1e-9)
    _vol_z_hist = ((hv - hv.rolling(20).mean()) / (hv.rolling(20).std() + 1e-9)).tail(60)
    _vol_tanh_w = _tanh_w(_vol_z_hist)
    if len(hv) >= 8:
        _v5 = hv.tail(5).values.astype(float)
        _v5_slope = float(np.polyfit(np.arange(5, dtype=float), _v5, 1)[0]) / (vol_mu + 1e-9)
        _vol_trend_pct = float((hv.rolling(5).mean().dropna() <= float(np.mean(_v5))).mean())
        _vol_signal = float(np.clip(0.5 + _v5_slope * 2.0, 0.0, 1.0)) * 0.60 + _vol_trend_pct * 0.40
    else:
        _vol_signal = 0.5
    if setup_type == "Breakout":
        _raw_vol_pts = float(np.clip(15.0 * _vol_signal, 0.0, 15.0))
        _t1_vol_now = float(hv.iloc[-1])
        _vol_60d    = hv.tail(60).dropna()
        _vol_p90    = float(_vol_60d.quantile(0.90)) if len(_vol_60d) >= 10 else vol_ma20 * 2.5
        if _t1_vol_now > _vol_p90:
            _spike_excess = (_t1_vol_now - _vol_p90) / max(_vol_p90, 1.0)
            _spike_decay  = float(np.clip(_spike_excess / 2.0, 0.0, 0.7))
            _raw_vol_pts  = max(_raw_vol_pts * (1.0 - _spike_decay), 0.0)
        vol_pts = round(_raw_vol_pts, 1)
    else:
        vol_pts = round(max(float(np.clip(15.0 * (1.0 - _vol_signal), 0.0, 15.0)), 5.0), 1)

    # Intraday vol velocity
    _vol_velocity_score = 0.0
    if _elapsed >= 30 and _elapsed < _SESSION_MINS:
        _velocity_ratio = day_vol / (vol_mu + 1e-9)
        _vol_velocity_score = float(np.clip(3.0 * np.tanh((_velocity_ratio - 1.0)), 0.0, 3.0))

    # ═══ F4 — PRE-BREAKOUT ACCUMULATION (0-10 pts) ═══
    _hist5 = list(hv.tail(4).values) + [day_vol]
    inst_ratio = float(np.mean(_hist5)) / (vol_ma20 + 1e-9)
    _inst_hist = (hv.rolling(5).mean() / (hv.rolling(20).mean() + 1e-9)).dropna()
    if len(_inst_hist) >= 20:
        _inst_centre = float(_inst_hist.tail(60).median())
        _inst_sigma  = float(_inst_hist.tail(60).std())
        _reg_inst = _reg.setdefault("inst_sigma", [])
        _reg_inst.append(_inst_sigma)
        _reg["inst_sigma"] = _reg_inst[-SCORE_CFG.REG_HISTORY:]
        _inst_sigma_floor = float(np.percentile(_reg_inst, 5)) if len(_reg_inst) >= 10 else 0.05
        _inst_k      = 1.0 / max(_inst_sigma, _inst_sigma_floor)
    else:
        _inst_centre = 1.2
        _inst_k      = 3.0
    inst_pts = round(10.0 / (1.0 + np.exp(-_inst_k * (inst_ratio - _inst_centre))), 1)
    inst_pts = max(0.0, min(10.0, inst_pts))

    # ═══ F5 — VOLATILITY CONTRACTION + RANGE COMPRESSION (0-10 pts) ═══
    _tr_series  = pd.concat([hh - hl, (hh - hc.shift(1)).abs(), (hl - hc.shift(1)).abs()], axis=1).max(axis=1)
    _atr5_hist  = _tr_series.rolling(5).mean()
    _atr20_hist = _tr_series.rolling(20).mean()
    _vc_series  = _atr5_hist / (_atr20_hist.replace(0, np.nan))
    _vc_pct     = percentile_last(_vc_series, min(250, len(_vc_series)))
    if pd.isna(_vc_pct):
        _vc_pct = 0.5
    vc_pts = round((1.0 - _vc_pct) * 5, 1)

    _rng_series = (hh.rolling(5).max() - hl.rolling(5).min()) / \
                  (hh.rolling(20).max() - hl.rolling(20).min() + 1e-9)
    _rci_pct    = percentile_last(_rng_series, min(250, len(_rng_series)))
    if pd.isna(_rci_pct):
        _rci_pct = 0.5
    rci     = float(_rng_series.iloc[-1]) if pd.notna(_rng_series.iloc[-1]) else 1.0
    rci_pts = round((1.0 - _rci_pct) * 5, 1)
    vc_pts  = vc_pts + rci_pts   # combined 0-10
    rci_vc_pts = vc_pts  # alias for return dict

    atr5_h  = float(_tr_series.iloc[-5:].mean())  if len(_tr_series) >= 5  else atr_v
    atr20_h = float(_tr_series.iloc[-20:].mean()) if len(_tr_series) >= 20 else atr_v
    vc_ratio = atr5_h / (atr20_h + 1e-9)

    # VCVE bonus
    vcve = inst_ratio * (1.0 - min(vc_ratio, 1.0))
    if len(_inst_hist) >= 10 and len(_vc_series) >= 10:
        _n_common   = min(60, len(_inst_hist), len(_vc_series))
        _inst_tail  = _inst_hist.iloc[-_n_common:].values
        _vc_tail    = _vc_series.iloc[-_n_common:].values
        _vcve_arr   = _inst_tail * np.clip(1.0 - _vc_tail, 0.0, 1.0)
        _vcve_arr   = _vcve_arr[~np.isnan(_vcve_arr)]
        _vcve_hist  = pd.Series(_vcve_arr, dtype=float)
    else:
        _vcve_hist  = pd.Series(dtype=float)
    _vcve_sat = float(_vcve_hist.quantile(0.75)) if len(_vcve_hist) >= 10 else 0.4
    _vcve_sat = max(_vcve_sat, 0.1)
    vcve_bonus = round(float(np.clip(3.0 * np.tanh(vcve / _vcve_sat), 0.0, 3.0)), 1)

    # ═══ UNIVERSAL LEADING SIGNALS ═══

    # L1: Upside Volume Skew (0-3 pts)
    _uv_bonus = 0.0
    if len(hc) >= 20:
        _up_mask   = hc.diff() > 0
        _dn_mask   = hc.diff() < 0
        _uv_window = min(20, len(hc))
        _up_vol    = float(hv[_up_mask].tail(_uv_window).sum())
        _dn_vol    = float(hv[_dn_mask].tail(_uv_window).sum())
        _uv_ratio  = _up_vol / (_dn_vol + 1e-9)
        _uv_hist = pd.Series([
            hv[_up_mask].iloc[max(0, i-20):i].sum() /
            (hv[_dn_mask].iloc[max(0, i-20):i].sum() + 1e-9)
            for i in range(20, min(60, len(hc)))
        ], dtype=float)
        if len(_uv_hist) >= 5:
            _uv_pct   = float((_uv_hist <= _uv_ratio).mean())
            _uv_bonus = round(_uv_pct * 3.0, 1)
        else:
            _uv_bonus = 1.5 if _uv_ratio > 1.0 else 0.5

    # L2: Close Position Rank (0-3 pts)
    _cpr_bonus = 0.0
    if len(hc) >= 10:
        _hl_range  = (hh - hl).replace(0, np.nan)
        _cpr_raw   = ((hc - hl) / _hl_range).dropna()
        _cpr_10    = float(_cpr_raw.tail(10).mean())
        _cpr_hist  = _cpr_raw.rolling(10).mean().dropna()
        if len(_cpr_hist) >= 10:
            _cpr_pct   = float((_cpr_hist <= _cpr_10).mean())
            _cpr_bonus = round(_cpr_pct * 3.0, 1)
        else:
            _cpr_bonus = round(_cpr_10 * 3.0, 1)

    # L3: Spread Compression + Rising Close (0-3 pts)
    _sc_bonus = 0.0
    if len(hc) >= 15:
        _range_5d  = (hh.tail(5).max() - hl.tail(5).min())
        _range_10d = (hh.tail(10).max() - hl.tail(10).min())
        _compression = 1.0 - (_range_5d / (_range_10d + 1e-9))
        try:
            _close_slope = float(np.polyfit(range(5), hc.tail(5).values, 1)[0]) / (atr_v + 1e-9)
        except (np.linalg.LinAlgError, ValueError):
            _close_slope = 0.0
        _quiet_accum = max(0.0, _compression) * max(0.0, _close_slope)
        _w = len(hc)
        if _w >= 20:
            _x    = np.arange(5, dtype=float)
            _sx   = _x.sum(); _sx2 = (_x**2).sum(); _n = 5
            _denom = _n * _sx2 - _sx**2
            _c_vals = hc.values.astype(float)
            _strides = np.lib.stride_tricks.sliding_window_view(_c_vals, 5)
            _sxy_arr = (_strides * _x).sum(axis=1)
            _sy_arr  = _strides.sum(axis=1)
            _slopes  = (_n * _sxy_arr - _sx * _sy_arr) / (_denom + 1e-9)
            _atr_vals = atr.iloc[:-1].values.astype(float)
            _atr_win  = _atr_vals[4:]
            _n_win    = min(len(_slopes), len(_atr_win))
            _slopes   = _slopes[:_n_win]; _atr_win = _atr_win[:_n_win]
            _range_5_arr  = np.array([hh.values[i:i+5].max() - hl.values[i:i+5].min() for i in range(_n_win)])
            _range_10_arr = np.array([hh.values[max(0,i-4):i+6].max() - hl.values[max(0,i-4):i+6].min() + 1e-9 for i in range(_n_win)])
            _comp_arr  = np.clip(1.0 - _range_5_arr / _range_10_arr, 0.0, 1.0)
            _slope_norm = np.clip(_slopes / (_atr_win + 1e-9), 0.0, None)
            _sc_hist_arr = _comp_arr * _slope_norm
            _sc_hist = pd.Series(_sc_hist_arr[-60:], dtype=float).dropna()
            if len(_sc_hist) >= 5:
                _sc_pct   = float((_sc_hist <= _quiet_accum).mean())
                _sc_bonus = round(_sc_pct * 3.0, 1)
            else:
                _sc_bonus = round(float(np.clip(_quiet_accum * 3.0, 0.0, 3.0)), 1)
        else:
            _sc_bonus = round(float(np.clip(_quiet_accum * 3.0, 0.0, 3.0)), 1)

    # L4: ATR Expansion Onset (0-3 pts)
    _atr_exp_bonus = 0.0
    if len(_tr_series) >= 25:
        _atr5_series  = _tr_series.rolling(5).mean().dropna()
        _atr20_series = _tr_series.rolling(20).mean().dropna()
        _common_idx   = _atr5_series.index.intersection(_atr20_series.index)
        _vc_ratio_ser = (_atr5_series.loc[_common_idx] / (_atr20_series.loc[_common_idx] + 1e-9))
        _vc_p30 = float(np.percentile(_vc_ratio_ser.values[~np.isnan(_vc_ratio_ser.values)], 30)) \
                  if len(_vc_ratio_ser.dropna()) >= 10 else 0.85
        if len(_vc_ratio_ser) >= 10:
            _vc_arr   = _vc_ratio_ser.values
            _vc_diff  = np.diff(_vc_arr)
            _was_compressed = False; _bars_since_onset = None
            for j in range(len(_vc_diff) - 1, -1, -1):
                if _vc_arr[j] < _vc_p30:
                    _was_compressed = True
                if _was_compressed and _vc_diff[j] > 0:
                    _bars_since_onset = len(_vc_diff) - j
                    break
            if _bars_since_onset is not None and _bars_since_onset <= 5:
                _atr_exp_bonus = round(float(np.clip(3.0 / _bars_since_onset, 0.0, 3.0)), 1)

    # L5: OI Buildup (0-3 pts, F&O only)
    oi_bonus = 0.0
    if "oi" in df.columns and len(df) >= 10:
        _oi = df["oi"].dropna()
        _oi_nonzero = _oi[_oi > 0]
        if len(_oi_nonzero) >= 5:
            _oi_now   = float(_oi.iloc[-1])
            _oi_5d    = float(_oi.iloc[-6]) if len(_oi) >= 6 else float(_oi.iloc[0])
            _oi_avg   = float(_oi_nonzero.tail(20).mean())
            _oi_std   = float(_oi_nonzero.tail(20).std())
            _oi_std   = max(_oi_std, _oi_avg * 0.05)
            _oi_rising = _oi_now > _oi_5d
            _oi_z      = (_oi_now - _oi_avg) / (_oi_std + 1e-9)
            _vc_p30_oi      = float(_vc_series.dropna().quantile(0.30)) \
                              if len(_vc_series.dropna()) >= 10 else 0.85
            _price_coiling  = vc_ratio < _vc_p30_oi
            if _oi_rising and _price_coiling and _oi_z > 0.5:
                _compression_strength = 1.0 - min(vc_ratio, 1.0)
                oi_bonus = round(float(np.clip(3.0 * np.tanh(_oi_z * _compression_strength), 0.0, 3.0)), 1)

    range5  = float((hh.tail(5).max()  - hl.tail(5).min()))
    range20 = float((hh.tail(20).max() - hl.tail(20).min()))

    # ═══ F6 — BASE / COIL QUALITY (0-10 pts) ═══
    _swing_window = 20
    if len(hh) >= 40:
        _peaks    = (hh.rolling(3, center=True).max() == hh).astype(int)
        _peak_idx = hh.index[_peaks == 1].tolist()
        if len(_peak_idx) >= 3:
            _peak_gaps = [_peak_idx[i+1] - _peak_idx[i] for i in range(len(_peak_idx)-1)
                          if isinstance(_peak_idx[i+1] - _peak_idx[i], (int, float))]
            if _peak_gaps:
                _swing_window = int(np.clip(np.median(_peak_gaps), 10, 40))

    if setup_type == "Breakout":
        rng5b  = float(hh.tail(5).max()) - float(hl.tail(5).min())
        tight  = 1.0 - min(1.0, rng5b / (base_rng + 1e-9))
        rec_hi = hh.tail(8)
        hi_spread = (rec_hi.max() - rec_hi.min()) / (atr_v + 1e-9)
        flatness  = max(0.0, 1.0 - min(hi_spread / 1.0, 1.0))
        _tight_hist = (1.0 - (hh.rolling(5).max() - hl.rolling(5).min()).tail(60) / (base_rng + 1e-9)).dropna()
        _flat_hist  = pd.Series([
            max(0.0, 1.0 - (hh.iloc[max(0,i-7):i].max() - hh.iloc[max(0,i-7):i].min()) / (atr_v + 1e-9))
            for i in range(max(8, len(hh)-60), len(hh))
        ], dtype=float).dropna()
        _tight_var = float(_tight_hist.var()) if len(_tight_hist) >= 5 else 0.5
        _flat_var  = float(_flat_hist.var())  if len(_flat_hist)  >= 5 else 0.5
        _total_var = _tight_var + _flat_var + 1e-9
        _w_tight   = _tight_var / _total_var
        _w_flat    = _flat_var  / _total_var
        coil_sc    = tight * _w_tight + flatness * _w_flat
        base_pos   = (ltp_score - base_lo) / (base_rng + 1e-9)
        _bpos_hist = ((hc - hl.rolling(_swing_window).min()) /
                      (hh.rolling(_swing_window).max() - hl.rolling(_swing_window).min() + 1e-9)).dropna().tail(60)
        _bpos_centre = float(_bpos_hist.quantile(0.75)) if len(_bpos_hist) >= 10 else 0.80
        _bpos_sigma  = float(max((_bpos_hist.quantile(0.90) - _bpos_hist.quantile(0.60)) / 1.35, 0.05)) \
                       if len(_bpos_hist) >= 10 else 0.10
        _bp_sigmoid_k = 1.0 / _bpos_sigma
        _bp_bonus = 0.2 / (1.0 + np.exp(-_bp_sigmoid_k * (base_pos - _bpos_centre)))
        coil_sc   = min(1.0, coil_sc + float(_bp_bonus))
    else:
        psw_hi  = float(hh.tail(_swing_window).max())
        psw_lo  = float(hl.tail(_swing_window).min())
        pm      = psw_hi - psw_lo + 1e-9
        pb_dep  = (psw_hi - float(hc.iloc[-1])) / pm
        if len(hh) >= 40:
            _sw = _swing_window
            _pb_depths = []
            for _i in range(_sw, min(len(hh), _sw + 120)):
                _ph = float(hh.iloc[_i - _sw : _i].max())
                _pl = float(hl.iloc[_i - _sw : _i].min())
                _pm = _ph - _pl + 1e-9
                _pd = (_ph - float(hc.iloc[_i])) / _pm
                _pb_depths.append(_pd)
            _pb_arr = np.array([x for x in _pb_depths if 0.0 <= x <= 1.0])
            if len(_pb_arr) >= 10:
                _pb_centre = float(np.median(_pb_arr))
                _pb_q75, _pb_q25 = np.percentile(_pb_arr, 75), np.percentile(_pb_arr, 25)
                _pb_sigma  = float(max((_pb_q75 - _pb_q25) / 1.35, 0.05))
            else:
                _pb_centre = 0.382; _pb_sigma = 0.125
        else:
            _pb_centre = 0.382; _pb_sigma = 0.125
        coil_sc  = float(np.clip(np.exp(-0.5 * ((pb_dep - _pb_centre) / _pb_sigma) ** 2), 0.0, 1.0))
        base_pos = (ltp_score - base_lo) / (base_rng + 1e-9)
    coil_pts = round(coil_sc * 10, 1)

    # ── 52-week position ──
    if len(hc) >= 50:
        _n250  = min(250, len(hc))
        _hi250 = float(hh.tail(_n250).max()); _lo250 = float(hl.tail(_n250).min())
        _pos_now = (ltp_score - _lo250) / (_hi250 - _lo250 + 1e-9)
        _pos_series = (hc - hc.rolling(_n250).min()) / (hc.rolling(_n250).max() - hc.rolling(_n250).min() + 1e-9)
        _pos_pct    = percentile_last(_pos_series, min(250, len(_pos_series)))
        if pd.notna(_pos_pct):
            _pos52w_reg = _reg.setdefault("pos52w_max", [])
            _pos52w_max = float(np.median(_pos52w_reg)) if len(_pos52w_reg) >= 10 else 3.0
            _pos52w_max = float(np.clip(_pos52w_max, 1.0, 5.0))
            pos52w_bonus = round(_pos_pct * _pos52w_max, 1)
            _pos52w_reg.append(_pos52w_max); _reg["pos52w_max"] = _pos52w_reg[-SCORE_CFG.REG_HISTORY:]
            pos52w = round(_pos_now, 3)
        else:
            pos52w_bonus = 0.0; pos52w = round(base_pos, 3)
    else:
        pos52w_bonus = 0.0; pos52w = round(base_pos, 3)

    # ── Liquidity Sweep ──
    sweep_bonus = 0.0
    if len(hc) >= 5:
        prior_support = float(hl.tail(5).min())
        prior_close   = float(hc.iloc[-1])
        lower_wick    = min(day_o, ltp_score) - day_lo
        _vol_z_hist_sw = ((hv - hv.rolling(20).mean()) / (hv.rolling(20).std() + 1e-9)).tail(60)
        _vol_z_p60     = float(_vol_z_hist_sw.quantile(0.60)) if len(_vol_z_hist_sw) >= 20 else 1.0
        if (day_lo < prior_support and ltp_score > prior_close and
                lower_wick >= 0.5 * atr_v and vol_z >= _vol_z_p60):
            _wick_atr_ratio = lower_wick / (atr_v + 1e-9)
            sweep_bonus = round(float(np.clip(4.0 * np.tanh(_wick_atr_ratio), 0.0, 4.0)), 1)

    # ── VWMA-20 ──
    vwap_bonus = 0
    if "volume" in df.columns and len(df) >= 20:
        typical    = (h + l + c) / 3
        cum_tv     = (typical * v).rolling(20).sum()
        cum_v_roll = v.rolling(20).sum()
        vwma20_val = float((cum_tv / cum_v_roll.replace(0, np.nan)).iloc[-1])
        if not np.isnan(vwma20_val):
            if ltp_score > vwma20_val:
                vwap_bonus = 2
            vwma20_prev = float((cum_tv / cum_v_roll.replace(0, np.nan)).iloc[-2]) if len(df) >= 21 else vwma20_val
            if not np.isnan(vwma20_prev) and vwma20_val > vwma20_prev:
                vwap_bonus += 1

    stab_bonus = 0.0

    # ── Momentum Stability ──
    if len(hc) >= 21:
        returns_20    = hc.iloc[-20:].pct_change().dropna()
        positive_days = int((returns_20 > 0).sum())
        stability     = positive_days / max(len(returns_20), 1)
    elif len(hc) >= 11:
        returns_10    = hc.iloc[-10:].pct_change().dropna()
        positive_days = int((returns_10 > 0).sum())
        stability     = positive_days / max(len(returns_10), 1)
    else:
        stability = 0.5
    if len(hc) >= 40:
        _stab_hist = hc.pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / max(len(x.dropna()), 1), raw=False
        ).dropna()
        _stab_pct = percentile_last(_stab_hist, min(60, len(_stab_hist)))
    else:
        _stab_pct = None
    if stability < 0.20 and (rsi_v < 40 or rsi_v <= rsi_p):
        _stab_kill_z   = (0.20 - stability) / 0.20
        _soft_penalty += float(np.clip(15.0 * _stab_kill_z, 0.0, 20.0))
    if _stab_pct is not None and pd.notna(_stab_pct):
        _stab_deviation = _stab_pct - 0.50
        _stab_w = float(max(np.std(list(_stab_hist)) if len(_stab_hist) >= 10 else 0.15, 0.05))
        _stab_z = _stab_deviation / _stab_w
        _reg_stab = _reg.setdefault("stab_adj_scale", [])
        _stab_scale = float(np.median(_reg_stab)) if len(_reg_stab) >= 10 else 5.0
        _reg_stab_adj = _reg.setdefault("stab_adj_obs", [])
        if len(_reg_stab_adj) >= 10:
            _stab_clip_lo = float(np.percentile(_reg_stab_adj, 5))
            _stab_clip_hi = float(np.percentile(_reg_stab_adj, 95))
        else:
            _stab_clip_lo, _stab_clip_hi = -8.0, 2.0
        stab_adj = float(np.clip(np.tanh(_stab_z) * _stab_scale, _stab_clip_lo, _stab_clip_hi))
        _reg_stab_adj.append(stab_adj); _reg["stab_adj_obs"] = _reg_stab_adj[-SCORE_CFG.REG_HISTORY:]
        _this_scale = abs(_stab_z) * atr_pct / 100.0 if _stab_z != 0 else 5.0
        _reg_stab.append(float(np.clip(_this_scale, 1.0, 15.0))); _reg["stab_adj_scale"] = _reg_stab[-SCORE_CFG.REG_HISTORY:]
        stab_bonus = stab_adj
    else:
        stab_bonus = 1.0 if stability >= 0.60 else 0.0

    # ═══ F7 — MA STRUCTURE (0-10 pts) ═══
    if len(e9) >= 4:
        _e9_slope = (float(e9.iloc[-1]) - float(e9.iloc[-4])) / (atr_v * 3.0 + 1e-9)
        _e9_slope_score = float(np.clip(0.5 + _e9_slope * 2.0, 0.0, 1.0))
    else:
        _e9_slope_score = 0.5
    _gap_now  = (e9_v - e20_v) / (atr_v + 1e-9)
    _gap_prev = (e9_y - e20_y) / (atr_v + 1e-9)
    _cross_prox = float(np.exp(-abs(_gap_now) * 2.0))
    _converge_score = float(np.clip(_cross_prox * (1.2 if _gap_now > _gap_prev else 0.8), 0.0, 1.0))
    _above_e50_score = float(np.clip((ltp_score - e50_v) / (atr_v + 1e-9) + 0.5, 0.0, 1.0))
    ma_pts = round((_e9_slope_score * 0.35 + _converge_score * 0.45 + _above_e50_score * 0.20) * 10.0, 1)

    # ═══ F8 — BREAKOUT PROXIMITY (0-10 pts) ═══
    if len(hh) >= 30:
        _dist_hist_bo = ((hh.rolling(20).max().shift(1) - hc) / (atr.iloc[:-1] + 1e-9)).dropna()
        _dist_hist_bo = _dist_hist_bo[_dist_hist_bo > 0].tail(60)
        _median_dist  = float(_dist_hist_bo.median()) if len(_dist_hist_bo) >= 10 else 0.7
        _median_dist  = max(_median_dist, 0.2)
        _prox_lambda  = float(np.log(2) / _median_dist)
        _reg_lam = _reg.setdefault("prox_lambda", [])
        _reg_lam.append(_prox_lambda); _reg["prox_lambda"] = _reg_lam[-SCORE_CFG.REG_HISTORY:]
        if len(_reg_lam) >= 10:
            _lam_lo = float(np.percentile(_reg_lam, 5)); _lam_hi = float(np.percentile(_reg_lam, 95))
        else:
            _lam_lo, _lam_hi = 0.5, 3.0
        _prox_lambda  = float(np.clip(_prox_lambda, _lam_lo, _lam_hi))
    else:
        _prox_lambda = 1.0

    if setup_type == "Breakout":
        d_trig = (base_hi - ltp_score) / (atr_v + 1e-9)
        _resist_series = hh.rolling(20).max().shift(1)
        _n_bo = min(len(hc), len(_resist_series.dropna()), len(atr) - 1)
        _hc_bo   = hc.values[-_n_bo:].astype(float)
        _res_bo  = _resist_series.values[-_n_bo:].astype(float)
        _atr_bo  = atr.values[-(_n_bo + 1):-1].astype(float)
        _valid   = ~np.isnan(_res_bo)
        _below_v = _valid & (_hc_bo < _res_bo)
        _above_n = np.zeros(len(_hc_bo), dtype=bool)
        _above_n[:-1] = _valid[:-1] & (_hc_bo[1:] > _res_bo[:-1])
        _bo_entry_v = _below_v & _above_n
        _dist_bo = np.where(_valid & (_atr_bo > 0), (_res_bo - _hc_bo) / _atr_bo, np.nan)
        if _bo_entry_v.sum() >= 3:
            _d = _dist_bo[_bo_entry_v]; _d = _d[(_d > 0) & ~np.isnan(_d)]
            _IDEAL_D = float(np.median(_d)) if len(_d) >= 2 else float(1.0 / _prox_lambda)
        else:
            _d = _dist_bo[(_dist_bo > 0) & ~np.isnan(_dist_bo)]
            _IDEAL_D = float(np.median(_d)) if len(_d) >= 5 else float(1.0 / _prox_lambda)
        _d_adj = abs(d_trig - _IDEAL_D)
        if d_trig < 0:
            _d_adj += abs(d_trig)
        prox_pts = round(max(0.0, min(10.0, 10.0 * np.exp(-_prox_lambda * _d_adj))), 1)
    else:
        _de9  = ltp_score - e9_v; _de20 = ltp_score - e20_v
        _closest_ema_dist = _de9 if abs(_de9) <= abs(_de20) else _de20
        _n_common = min(len(hc), len(e9), len(e20), len(atr) - 1)
        _hc_arr   = hc.values[-_n_common:].astype(float)
        _e9_arr   = e9.values[-_n_common:].astype(float)
        _e20_arr  = e20.values[-_n_common:].astype(float)
        _atr_arr  = atr.values[-(_n_common + 1):-1].astype(float)
        _cross_mask = np.zeros(len(_e9_arr), dtype=bool)
        if len(_e9_arr) >= 2:
            _cross_mask[1:] = (_e9_arr[1:] > _e20_arr[1:]) & (_e9_arr[:-1] <= _e20_arr[:-1])
        _pre_cross_mask = np.zeros(len(_e9_arr), dtype=bool)
        if _cross_mask.sum() >= 3:
            cross_idxs = np.where(_cross_mask)[0]
            pre_idxs   = cross_idxs[cross_idxs > 0] - 1
            _pre_cross_mask[pre_idxs] = True
        _dist_arr = (_hc_arr - _e20_arr) / (_atr_arr + 1e-9)
        if _pre_cross_mask.sum() >= 2:
            _pb_entry_dists = _dist_arr[_pre_cross_mask]
            _pb_entry_dists = _pb_entry_dists[_pb_entry_dists > 0]
            _IDEAL_D_PB = float(np.median(_pb_entry_dists)) if len(_pb_entry_dists) >= 2 else float(1.0 / _prox_lambda)
        else:
            _pos_dists = _dist_arr[_dist_arr > 0]
            _IDEAL_D_PB = float(np.median(_pos_dists)) if len(_pos_dists) >= 5 else float(1.0 / _prox_lambda)
        _d_from_ideal = (_closest_ema_dist / (atr_v + 1e-9)) - _IDEAL_D_PB
        prox_dist = abs(_d_from_ideal) if _d_from_ideal >= 0 else abs(_d_from_ideal) * 1.5
        prox_pts  = round(max(0.0, min(10.0, 10.0 * np.exp(-_prox_lambda * prox_dist))), 1)

    # ═══ F9 — ATR POTENTIAL (0-5 pts) ═══
    atr_hist_pct = atr14(df).iloc[:-1] / c.iloc[:-1] * 100
    atr_hist_pct = atr_hist_pct.tail(60).dropna()
    if len(atr_hist_pct) >= 10:
        atr_pct_rank = float((atr_hist_pct <= atr_pct).mean())
        atp_pts = round((1.0 - atr_pct_rank) * 5, 1)
    else:
        _atr_lo = float(atr_hist_pct.min()) if len(atr_hist_pct) > 0 else 0.0
        _atr_hi = float(atr_hist_pct.max()) if len(atr_hist_pct) > 0 else 5.0
        _atr_hi = max(_atr_hi, _atr_lo + 0.1)
        atp_pts = round(float(np.clip((1.0 - (atr_pct - _atr_lo) / (_atr_hi - _atr_lo)) * 5, 0.0, 5.0)), 1)

    # ═══ F10 — CANDLESTICK TRIGGER (0-5 pts) ═══
    raw_cdl, candle_names = detect_candle_patterns(
        day_o, day_hi, day_lo, ltp_score,
        float(o.iloc[-2]), float(h.iloc[-2]), float(l.iloc[-2]), float(c.iloc[-2])
    )
    cdl_pts = min(round(raw_cdl * 0.5, 1), 5.0)

    # ── DARVAS BOX ──
    _darvas_result = darvas_box_score(hist, atr_v)
    _d_box_hi = _darvas_result.get("box_high", np.nan)
    _d_box_lo = _darvas_result.get("box_low",  np.nan)
    if not (np.isnan(_d_box_hi) or np.isnan(_d_box_lo)):
        _d_width = _d_box_hi - _d_box_lo + 1e-9
        _d_pos   = (ltp_score - _d_box_lo) / _d_width
        _d_pos_score = float(np.clip(_d_pos, 0.0, 1.0)) if setup_type == "Breakout" \
                       else float(np.clip(1.0 - _d_pos, 0.0, 1.0))
        _d_atr_ratio = float(_darvas_result.get("box_atr_ratio", 1.0) or 1.0)
        if math.isnan(_d_atr_ratio): _d_atr_ratio = 1.0
        _d_tightness = 1.0 / (1.0 + _d_atr_ratio)
        _d_time = float(np.clip(_darvas_result.get("bars_in_box", 0) / 20.0, 0.0, 1.0))
        darvas_pts = round((_d_tightness*0.40 + _d_pos_score*0.35 + _d_time*0.25)*10.0, 1)
    else:
        darvas_pts = _darvas_result["darvas_score"]
    darvas_bonus = 0.0

    # ── Cross-sectional factors (from pre-computed universe ranks) ──
    _bb_cs_pct  = STATE.get("cs_bb_squeeze",  {}).get(ticker, None)
    if _bb_cs_pct is None:
        _, _bb_self = bb_width_compression_score(hc); _bb_cs_pct = _bb_self
    bb_pts = round(float(_bb_cs_pct) * 8.0, 1)

    _vdu_cs_pct = STATE.get("cs_vol_dryup", {}).get(ticker, None)
    if _vdu_cs_pct is None:
        _, _vdu_self = volume_dryup_score(hv); _vdu_cs_pct = _vdu_self
    if setup_type == "Breakout":
        vol_dryup_pts = round(float(_vdu_cs_pct) * 8.0, 1)
    else:
        _vdu_tent = float(np.clip(1.0 - abs(_vdu_cs_pct - 0.60) / 0.40, 0.0, 1.0))
        vol_dryup_pts = round(_vdu_tent * 5.0, 1)

    _clv_cs_pct = STATE.get("cs_clv_accum", {}).get(ticker, None)
    if _clv_cs_pct is None:
        _, _clv_self = clv_accumulation_score(hc, hh, hl, hv); _clv_cs_pct = _clv_self
    clv_pts = round(float(_clv_cs_pct) * 8.0, 1)

    _vcp_cs_pct = STATE.get("cs_vcp", {}).get(ticker, None)
    if _vcp_cs_pct is None:
        _vcp_result_inline = detect_vcp(c, h, l, v, atr)
        _vcp_cs_pct = _vcp_result_inline["vcp_score"]; _vcp_detail = _vcp_result_inline
    else:
        _vcp_detail = detect_vcp(c, h, l, v, atr)
    vcp_pts = round(float(_vcp_cs_pct) * 10.0, 1)

    # ── Vol-quiet (primary signal) ──
    _vr_hist    = (hv.iloc[:-1] / (hv.iloc[:-1].rolling(20).mean() + 1e-9)).dropna()
    _vr_now     = float(hv.iloc[-1]) / (vol_mu + 1e-9)
    _quiet_pct  = float((_vr_hist >= _vr_now).mean()) if len(_vr_hist) >= 10 else float(np.clip(1.0 - _vr_now, 0.0, 1.0))
    vol_quiet_pts = round(_quiet_pct * 14.0, 1)

    # SpreadComp as primary factor (0-11 pts)
    spread_pts = round(float(np.clip(_sc_bonus / 3.0, 0.0, 1.0)) * 11.0, 1)

    # ── Liquidity score (sigmoid on log-ADV) ──
    _liq_logadv    = float(np.log(_adv_turnover + 1.0))
    _LIQ_CENTRE    = float(np.log(5e7))
    _LIQ_SCALE     = 1.0
    liquidity_score = float(1.0 / (1.0 + np.exp(-_LIQ_SCALE * (_liq_logadv - _LIQ_CENTRE))))

    # ═══ REVERSAL SCORING ═══
    _rev_rsi_pts = _rev_coil_pts = _rev_prox_pts = _rev_spread_pts = 0.0
    _rev_vol_pts = _rev_wash_pts = _rev_tail_pts = _rev_support_pts = 0.0
    if setup_type == "Reversal":
        _rsi_p90_rev = float(_rsi_hist_full.tail(60).quantile(0.90)) if len(_rsi_hist_full) >= 20 else 70.0
        _rsi_p10_rev = float(_rsi_hist_full.tail(60).quantile(0.10)) if len(_rsi_hist_full) >= 20 else 25.0
        _rsi_range   = max(_rsi_p90_rev - _rsi_p10_rev, 10.0)
        _rsi_oversold_depth = float(np.clip((_rsi_p90_rev - rsi_v) / _rsi_range, 0.0, 1.0))
        _rev_rsi_pts = round(40.0 * _rsi_oversold_depth, 1)
        _rev_coil_pts = round(30.0 * float(np.clip(coil_pts / 10.0, 0.0, 1.0)), 1)
        _rev_prox_pts = round(20.0 * float(np.clip(prox_pts / 10.0, 0.0, 1.0)), 1)
        _rev_spread_pts = round(10.0 * float(np.clip(_sc_bonus / 3.0, 0.0, 1.0)), 1)
        _rev_vol_pct = float((hv.iloc[:-1] <= float(hv.iloc[-1])).mean())
        _rev_vol_pts = round(5.0 * _rev_vol_pct, 1)
        _rev_wash_score = float(np.clip((_washout_depth - 1.5) / 4.0, 0.0, 1.0))
        _rev_wash_pts   = round(10.0 * _rev_wash_score, 1)
        _rev_tail_pts   = round(5.0 * float(np.clip((_t1_close_pos - 0.30) / 0.70, 0.0, 1.0)), 1)
        _rev_support_pts = _rev_prox_pts
        _rev_penalty = 0.0
        if _sma200_gap_atr < -5.0:
            _rev_penalty += float(np.clip((_sma200_gap_atr + 5.0) * -3.0, 0.0, 15.0))
        if liquidity_score < 0.2:
            _rev_penalty += 10.0 * (0.2 - liquidity_score) / 0.2
        _rev_raw = _rev_rsi_pts + _rev_coil_pts + _rev_prox_pts + _rev_spread_pts
        total    = round(max(0.0, min(100.0, _rev_raw - _rev_penalty)), 1)
        # Reversal uses its own EMI/rank — NOT multiplied by coverage  [F-04]
        emi            = round(total * atr_pct / 100, 3)
        composite_rank = round((total / 100.0) * 0.75 + liquidity_score * 0.25, 4)
        _bb_norm  = float(_bb_cs_pct)  if _bb_cs_pct  is not None else 0.5
        _vdu_norm = float(_vdu_cs_pct) if _vdu_cs_pct is not None else 0.5
        _clv_norm = float(_clv_cs_pct) if _clv_cs_pct is not None else 0.5
        _vcp_norm = float(_vcp_cs_pct) if _vcp_cs_pct is not None else _vcp_detail["vcp_score"]
        _vc_norm  = 1.0 - _vc_pct
        breakout_prob  = float(np.mean([_bb_norm, _vdu_norm, _clv_norm, _vcp_norm, _vc_norm]))
        volume_stability = float(np.clip(stability, 0.0, 1.0))
        # ── Horizon (reversal) ──
        _rsi_turning_rev = rsi_v > rsi_p
        if _rsi_turning_rev and _t1_close_pos >= 0.60 and raw_cdl >= 1:
            horizon = "Intraday"
            hz_note = (f"Capitulation bottom confirmed — RSI {rsi_v:.0f} turning, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Pattern: {', '.join(candle_names) if candle_names else 'hammer/wick'}. "
                       f"Buy on open, tight stop below {float(hl.iloc[-1]):.2f}.")
        elif _rsi_turning_rev:
            horizon = "Swing 2-5D"
            hz_note = (f"Washout in progress — RSI {rsi_v:.0f} showing first turn, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Enter on next green candle above {ltp_score:.2f}.")
        else:
            horizon = "Mid 5-14D"
            hz_note = (f"Panic selling extreme — RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Wait for RSI to tick up + candle confirmation before entry.")
        _atr_cv   = float(atr.iloc[-20:].std() / (atr.iloc[-20:].mean() + 1e-9)) if len(atr) >= 20 else 0.3
        _cv_scale = float(np.clip(1.0 + _atr_cv, 0.7, 1.5))
        entry      = round(ltp, 2)
        entry_note = (f"Buy at open — reversal from panic low. "
                      f"RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                      f"Stop below {float(hl.iloc[-1]):.2f}")
        stp  = round(float(hl.iloc[-1]) - 0.25 * atr_v, 2)
        tgt  = max(round(e20_v, 2), round(entry + 1.5 * atr_v, 2))
        risk_raw   = max(entry - stp,  0.01)
        reward_raw = max(tgt  - entry, 0.01)
        rr         = round(reward_raw / risk_raw, 2)
        _wr_prior  = float(STATE["per_stock_winrate"].get(ticker, None) or
                          np.clip(0.40 + 0.20 * cs_rs_score + 0.10 * stability, 0.35, 0.70))
        kelly_frac = round(float(np.clip(
            0.5 * (_wr_prior * max(rr, 0.5) - (1.0 - _wr_prior)) / (max(rr, 0.5) + 1e-9), 0.0, 0.25)), 3)
        move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0
        # No flush needed - _reg is in-place  [F-09]
        return {
            "SetupType":  setup_type, "Score": total, "EMI": emi,
            "CompositeRank": composite_rank, "Horizon": horizon, "HorizonNote": hz_note,
            "Entry": entry, "Target": tgt, "Stop": stp,
            "Risk": round(risk_raw, 1), "Reward": round(reward_raw, 1),
            "RR": rr, "KellyFrac": kelly_frac, "Move%": move_pct, "EntryNote": entry_note,
            "RS": round(rs_pts, 1), "RS_Sector": round(rs_sect_pts, 1),
            "Volume": round(vol_pts, 1), "InstVol": round(inst_pts, 1),
            "VolCont": round(rci_vc_pts, 1), "RCI": round(rci, 3),
            "VolQuiet": round(vol_quiet_pts, 1), "SpreadPts": round(spread_pts, 1),
            "VolDryUp": round(vol_dryup_pts, 1), "CLVAccum": round(clv_pts, 1),
            "BBSqueeze": round(bb_pts, 1), "VCP": round(vcp_pts, 1),
            "BreakoutProb": round(breakout_prob, 3), "SignalPersist": 1.0,
            "VCP_Detected": _vcp_detail["vcp_detected"],
            "VCP_Pullbacks": _vcp_detail["vcp_pullback_n"],
            "VCP_Contraction": round(_vcp_detail["vcp_contraction"], 3),
            "VCP_VolComp": round(_vcp_detail["vcp_vol_comp"], 3),
            "VCP_VolDryup": round(_vcp_detail["vcp_vol_dryup"], 3),
            "VCP_Tightness": round(_vcp_detail["vcp_tightness"], 3),
            "VCP_Position": round(_vcp_detail["vcp_position"], 3),
            "RS_raw": round(rs_pts, 1), "Sect_raw": round(rs_sect_pts, 1),
            "Vol_raw": round(vol_pts, 1), "Inst_raw": round(inst_pts, 1),
            "VC_raw": round(vc_pts, 1), "Coil_raw": round(coil_pts, 1),
            "MA_raw": round(ma_pts, 1), "Prox_raw": round(prox_pts, 1),
            "Darvas": round(darvas_pts, 1),
            "DarvasBox": _darvas_result.get("box_high", np.nan),
            "DarvasLow": _darvas_result.get("box_low", np.nan),
            "DarvasInBox": _darvas_result.get("in_box", False),
            "ADVTurnover": round(_adv_turnover / 1e7, 2),
            "LiquidityScore": round(liquidity_score, 3),
            "SoftPenalty": round(_soft_penalty, 1),
            "AboveSMA200": above_long_trend,
            "Coil": round(coil_pts, 1), "MA_Struct": round(ma_pts, 1),
            "Proximity": round(prox_pts, 1), "ATR_Pot": round(atp_pts, 1),
            "Candle": round(cdl_pts, 1),
            "Patterns": ", ".join(candle_names) if candle_names else "—",
            "RS_Accel": round(rs_accel, 4), "AccelScore": round(acc_score * 100, 1),
            "VCVE": round(vcve, 3), "BasePos": round(base_pos, 3),
            "Pos52W": round(pos52w, 3), "Stability": round(stability, 2),
            "Sweep": False, "VWMA20_OK": False, "DarvasBO": 0.0,
            "UpVolSkew": round(_uv_bonus, 1), "CPR": round(_cpr_bonus, 1),
            "SpreadComp": round(_sc_bonus, 1), "ATRExpOnset": round(_atr_exp_bonus, 1),
            "OI_Buildup": round(oi_bonus, 1), "VolVelocity": 0.0,
            "RSDivergence": round(rs_div_bonus, 1),
            "CSRank5d": round(cs_rs_score, 3), "AbsRS": round(abs_rs_score, 3),
            "RSI7": round(rsi_v, 1), "VolRatio": round(vol_ratio, 2),
            "Rev_RSI_Pts": round(_rev_rsi_pts, 1), "Rev_Vol_Pts": round(_rev_vol_pts, 1),
            "Rev_Wash_Pts": round(_rev_wash_pts, 1), "Rev_Tail_Pts": round(_rev_tail_pts, 1),
            "Rev_Support_Pts": round(_rev_support_pts, 1),
            "WashoutDepth": round(_washout_depth, 2),
            "CandleTailPos": round(_t1_close_pos, 3),
            "VolZ": round(vol_z, 2),
            "VolBOThr": round(vol_bo_thresh / vol_mu, 2),
            "InstRatio": round(inst_ratio, 2), "VC_Ratio": round(vc_ratio, 2),
            "ATR%": round(atr_pct, 2), "RS_vs_Nifty": round(rs_score * 100, 1),
            "BO_Ext_ATR": round(breakout_ext, 2),
            "Sector": sect_name or "?",
            "EMA9": round(e9_v, 2), "EMA20": round(e20_v, 2), "EMA50": round(e50_v, 2),
        }

    # ═══ UNIFIED SCORE ASSEMBLY (Breakout + Pullback only — Reversal returned above) ═══
    _t1_vol_ratio_pb = float(hv.iloc[-1]) / (vol_ma20 + 1e-9)
    vol_surge_pts    = round(float(np.clip((_t1_vol_ratio_pb / 3.0) * 14.0, 0.0, 14.0)), 1)

    _primary_vol_pts = vol_quiet_pts   # Breakout/Pullback always use vol_quiet
    _primary_vol_max = SCORE_CFG.MAX_VOL

    _W_SPREAD    = SCORE_CFG.W_SPREAD; _W_VOL_QUIET = SCORE_CFG.W_VOL; _W_COIL = SCORE_CFG.W_COIL
    _MAX_SPREAD  = SCORE_CFG.MAX_SPREAD; _MAX_VOL_QUIET = SCORE_CFG.MAX_VOL; _MAX_COIL = SCORE_CFG.MAX_COIL
    _weighted_raw = _W_SPREAD * spread_pts + _W_VOL_QUIET * _primary_vol_pts + _W_COIL * coil_pts
    _weighted_max = _W_SPREAD * _MAX_SPREAD + _W_VOL_QUIET * _primary_vol_max + _W_COIL * _MAX_COIL
    total_base    = round(_weighted_raw * (100.0 / max(_weighted_max, 1e-9)), 1)
    total = round(total_base, 1)

    # Preserve raw values
    rs_pts_raw = rs_pts; vol_pts_raw = vol_pts; rs_sect_pts_raw = rs_sect_pts
    inst_pts_raw = inst_pts; vc_pts_raw = vc_pts; coil_pts_raw = coil_pts
    ma_pts_raw = ma_pts; prox_pts_raw = prox_pts

    # ── Bonuses (Breakout + Pullback only) ──
    _persist_factor = 1.0
    if len(hc) >= 6:
        _vc_last3 = _vc_series.iloc[-4:-1].dropna()
        _vc_p40   = float(_vc_series.dropna().quantile(0.40)) if len(_vc_series.dropna()) >= 10 else 0.9
        _compressed_days = int((_vc_last3 < _vc_p40).sum())
        _persist_factor = float(np.clip(0.5 + _compressed_days * 0.25, 0.50, 1.0))

    _bonus_raw = (
        _uv_bonus + _cpr_bonus + _sc_bonus + _atr_exp_bonus +
        oi_bonus + vcve_bonus + sweep_bonus + stab_bonus
    ) * _persist_factor
    _BONUS_CAP_ABS = SCORE_CFG.BONUS_CAP
    bonuses = round((_bonus_raw * (_BONUS_CAP_ABS / _bonus_raw)
                     if _bonus_raw > _BONUS_CAP_ABS else _bonus_raw), 1) if _bonus_raw > 0 else 0.0
    total += bonuses

    # ── Soft penalties ──
    total = max(0.0, total - _soft_penalty)

    # ── Breadth / VIX adjustment ──
    _nifty_breadth_adj = 0.0
    _breadth_cached = STATE.get("breadth_cache", None)
    if _breadth_cached is not None:
        _breadth = _breadth_cached
        _breadth_hist_list = STATE.get("breadth_hist", [])
        _breadth_mu  = float(np.mean(_breadth_hist_list)) if len(_breadth_hist_list) >= 5  else 0.50
        _breadth_sig = float(np.std(_breadth_hist_list))  if len(_breadth_hist_list) >= 10 else 0.12
        _breadth_sig = max(_breadth_sig, 0.03)
        _breadth_z   = (_breadth - _breadth_mu) / _breadth_sig
        _raw_breadth_adj = float(np.clip(6.0 * np.tanh(_breadth_z), -8.0, 4.0))
        if _raw_breadth_adj < 0:
            _penalty_scale = max(0.0, 1.0 - cs_rs_score)
            _nifty_breadth_adj = _raw_breadth_adj * _penalty_scale
        else:
            _nifty_breadth_adj = _raw_breadth_adj
    elif nifty_r5 is not None and nifty_r20 is not None:
        _n5  = nifty_r5  or 0.0; _n20 = nifty_r20 or 0.0
        _nifty_breadth_adj = float(np.clip((_n5 + _n20 * 0.5) * 100, -8.0, 4.0))
    elif not _resolved_above_20dma:
        _nifty_breadth_adj = -8.0

    _vix_adj = 0.0
    if _resolved_vix_level is not None:
        _vix_z   = (_resolved_vix_level - _resolved_vix_median) / (_resolved_vix_sigma + 1e-9)
        _vix_adj = float(np.clip(-6.0 * np.tanh(_vix_z), -8.0, 2.0))
        if not _resolved_vix_falling:
            _vix_adj = float(np.clip(_vix_adj - 2.0 * abs(np.tanh(_vix_z)), -8.0, 0.0))
    elif not _resolved_vix_falling:
        _vix_adj = -5.0

    total += _nifty_breadth_adj + _vix_adj

    # ── Regime penalty for breakout setups ──
    if setup_type == "Breakout":
        _new_factor_weight = (bb_pts + vol_dryup_pts) / (8.0 + 8.0 + 1e-9)
        if _regime == "BEAR":
            total -= _new_factor_weight * 8.0
        elif _regime == "CHOP":
            total -= _new_factor_weight * 4.0

    total = max(0, min(100, round(total, 1)))
    emi   = round(total * atr_pct / 100, 3)

    volume_stability = float(np.clip(stability, 0.0, 1.0))
    _bb_norm   = float(_bb_cs_pct)   if _bb_cs_pct  is not None else 0.5
    _vdu_norm  = float(_vdu_cs_pct)  if _vdu_cs_pct is not None else 0.5
    _clv_norm  = float(_clv_cs_pct)  if _clv_cs_pct is not None else 0.5
    _vcp_norm  = float(_vcp_cs_pct)  if _vcp_cs_pct is not None else _vcp_detail["vcp_score"]
    _vc_norm   = 1.0 - _vc_pct
    breakout_prob = float(np.mean([_bb_norm, _vdu_norm, _clv_norm, _vcp_norm, _vc_norm]))

    composite_rank = round(emi * 0.70 + liquidity_score * 0.20 + volume_stability * 0.10, 4)

    # No explicit flush needed — _reg is in-place reference  [F-09]

    # ═══ HORIZON CLASSIFICATION ═══
    imminence = prox_pts + vc_pts
    if len(hh) >= 40:
        _bo_dist_hist   = ((hh.rolling(20).max() - hc) / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
        _p20_bo = float(np.percentile(_bo_dist_hist, 20)) if len(_bo_dist_hist) >= 10 else 0.25
        _p50_bo = float(np.percentile(_bo_dist_hist, 50)) if len(_bo_dist_hist) >= 10 else 1.0
        _p80_bo = float(np.percentile(_bo_dist_hist, 80)) if len(_bo_dist_hist) >= 10 else 3.0
        _pb_dist_hist = ((e20.iloc[:-1] - hc) / (atr.iloc[:-1] + 1e-9)).clip(0).dropna().tail(60)
        _p20_pb = float(np.percentile(_pb_dist_hist, 20)) if len(_pb_dist_hist) >= 10 else 0.3
        _p50_pb = float(np.percentile(_pb_dist_hist, 50)) if len(_pb_dist_hist) >= 10 else 1.0
        _p80_pb = float(np.percentile(_pb_dist_hist, 80)) if len(_pb_dist_hist) >= 10 else 2.5
    else:
        _p20_bo, _p50_bo, _p80_bo = 0.25, 1.0, 3.0
        _p20_pb, _p50_pb, _p80_pb = 0.3,  1.0, 2.5

    _atr_cv    = float(atr.iloc[-20:].std() / (atr.iloc[-20:].mean() + 1e-9)) if len(atr) >= 20 else 0.3
    _cv_scale  = float(np.clip(1.0 + _atr_cv, 0.7, 1.5))
    _tgt_mult  = {
        "Imminent BO": round(0.6 * _cv_scale, 2),
        "Intraday":    round(0.6 * _cv_scale, 2),
        "Swing 2-5D":  round(1.8 * _cv_scale, 2),
        "Mid 5-14D":   round(3.2 * _cv_scale, 2),
        "Long 14-30D": round(4.5 * _cv_scale, 2),
    }

    if setup_type == "Breakout":
        d_trig_atr = (base_hi - ltp_score) / (atr_v + 1e-9)
        if d_trig_atr <= _p20_bo and day_vol >= vol_bo_thresh:
            horizon = "Imminent BO"
            hz_note = f"AT TRIGGER — vol {vol_ratio:.1f}× avg (threshold {vol_bo_thresh/vol_mu:.1f}×). Enter now or market open."
        elif d_trig_atr <= 0.0 and vol_ratio >= 1.5 and rsi_v < float(_rsi_hist_full.tail(60).quantile(0.60)):
            horizon = "Intraday"
            hz_note = f"Breaking today — RSI {rsi_v:.0f}, vol {vol_ratio:.1f}×. Trail stop above base low."
        elif d_trig_atr <= _p20_bo:
            horizon = "Swing 2-5D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Place limit above {base_hi:.1f}."
        elif d_trig_atr <= _p50_bo:
            horizon = "Mid 5-14D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Coiling — alert for vol expansion."
        elif d_trig_atr <= _p80_bo:
            horizon = "Long 14-30D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Base building — add to watchlist."
        else:
            horizon = "Long 14-30D"
            hz_note = f"{d_trig_atr:.2f} ATR from trigger. Base still forming."
    elif setup_type == "Reversal":
        _rsi_turning_rev = rsi_v > rsi_p
        if _rsi_turning_rev and _t1_close_pos >= 0.60 and raw_cdl >= 1:
            horizon = "Intraday"
            hz_note = (f"Capitulation bottom confirmed — RSI {rsi_v:.0f} turning, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Pattern: {', '.join(candle_names) if candle_names else 'hammer/wick'}. "
                       f"Buy on open, tight stop below {float(hl.iloc[-1]):.2f}.")
        elif _rsi_turning_rev:
            horizon = "Swing 2-5D"
            hz_note = (f"Washout in progress — RSI {rsi_v:.0f} showing first turn, "
                       f"vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Enter on next green candle above {ltp_score:.2f}.")
        else:
            horizon = "Mid 5-14D"
            hz_note = (f"Panic selling extreme — RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                       f"Wait for RSI to tick up + candle confirmation before entry.")
    else:
        rsi_turning  = rsi_v > rsi_p
        pb_depth_atr = (e20_v - ltp_score) / (atr_v + 1e-9)
        if pb_depth_atr <= _p20_pb and rsi_turning and vol_ratio <= 0.8:
            horizon = "Intraday"
            hz_note = f"EMA20 support + RSI turning ({rsi_v:.0f}↑). Vol dry = clean pullback. Buy near {e20_v:.1f}."
        elif pb_depth_atr <= _p20_pb and rsi_turning and raw_cdl >= 2:
            horizon = "Imminent BO"
            hz_note = f"Reversal candle at EMA. RSI {rsi_v:.0f}↑, pattern: {', '.join(candle_names) if candle_names else 'none'}."
        elif pb_depth_atr <= _p50_pb and rsi_v >= 40:
            horizon = "Swing 2-5D"
            hz_note = f"Approaching EMA20. RSI {rsi_v:.0f}. Wait for reversal candle + vol confirmation."
        elif pb_depth_atr <= _p80_pb:
            horizon = "Mid 5-14D"
            hz_note = f"Pullback deepening ({pb_depth_atr:.1f} ATR below EMA20). Do not enter yet."
        else:
            horizon = "Long 14-30D"
            hz_note = f"Extended correction ({pb_depth_atr:.1f} ATR below EMA20). Watch for base formation."

    tgt_mult = _tgt_mult.get(horizon, round(1.8 * _cv_scale, 2))

    # ── TRADE LEVELS ──
    if setup_type == "Breakout":
        _entry_buffer = atr_v * 0.1 * max(0.5, vc_ratio)
        entry = round(base_hi + _entry_buffer, 2) if ltp < base_hi else round(ltp, 2)
        entry_note = (f"Buy above {entry:.2f} ({_entry_buffer:.2f} above base high {base_hi:.2f})"
                      if ltp < base_hi else f"Breaking now — buy on close above {base_hi:.2f}")
        tgt = round(entry + tgt_mult * atr_v, 2)
        _stop_buf = atr_v * max(0.3, min(0.7, vc_ratio))
        stp = round(base_lo - _stop_buf, 2)
    elif setup_type == "Reversal":
        entry      = round(ltp, 2)
        entry_note = (f"Buy at open — reversal from panic low. "
                      f"RSI {rsi_v:.0f}, vol {_t1_vol_ratio_rev:.1f}× avg. "
                      f"Stop below {float(hl.iloc[-1]):.2f}")
        stp  = round(float(hl.iloc[-1]) - 0.25 * atr_v, 2)
        tgt  = max(round(e20_v, 2), round(entry + 1.5 * atr_v, 2))
    else:
        entry = round(ltp, 2)
        entry_note = f"Buy near EMA20 ({e20_v:.2f}) on reversal candle"
        tgt_struct = round(float(hh.tail(20).max()) * 0.997, 2)
        tgt_atr    = round(entry + tgt_mult * atr_v, 2)
        tgt        = max(tgt_struct, tgt_atr)
        stp = round(e50_v - atr_v, 2)

    risk_raw   = max(entry - stp,  0.01)
    reward_raw = max(tgt  - entry, 0.01)
    rr         = round(reward_raw / risk_raw, 2)
    risk       = round(risk_raw,   1)
    reward     = round(reward_raw, 1)
    move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0

    # ── Kelly ──
    _wr_prior = float(STATE["per_stock_winrate"].get(ticker, None) or
                      np.clip(0.40 + 0.20 * cs_rs_score + 0.10 * stability, 0.35, 0.70))
    kelly_frac = round(float(np.clip(0.5 * (_wr_prior * max(rr, 0.5) - (1.0 - _wr_prior)) / (max(rr, 0.5) + 1e-9), 0.0, 0.25)), 3)

    return {
        "SetupType":  setup_type,
        "Score":      total,
        "EMI":        emi,
        "CompositeRank": composite_rank,
        "Horizon":    horizon,
        "HorizonNote": hz_note,
        "Entry":      entry,
        "Target":     tgt,
        "Stop":       stp,
        "Risk":       risk,
        "Reward":     reward,
        "RR":         rr,
        "KellyFrac":  kelly_frac,
        "Move%":      move_pct,
        "EntryNote":  entry_note,
        "RS":         round(rs_pts,       1),
        "RS_Sector":  round(rs_sect_pts,  1),
        "Volume":     round(vol_pts,      1),
        "InstVol":    round(inst_pts,     1),
        "VolCont":    round(rci_vc_pts,   1),
        "RCI":        round(rci, 3),
        "VolQuiet":   round(vol_quiet_pts, 1),
        "SpreadPts":  round(spread_pts,   1),
        "VolDryUp":   round(vol_dryup_pts, 1),
        "CLVAccum":   round(clv_pts,      1),
        "VCP":        round(vcp_pts,      1),
        "BreakoutProb": round(breakout_prob, 3),
        "SignalPersist": round(_persist_factor, 2),
        "VCP_Detected":    _vcp_detail["vcp_detected"],
        "VCP_Pullbacks":   _vcp_detail["vcp_pullback_n"],
        "VCP_Contraction": round(_vcp_detail["vcp_contraction"], 3),
        "VCP_VolComp":     round(_vcp_detail["vcp_vol_comp"],    3),
        "VCP_VolDryup":    round(_vcp_detail["vcp_vol_dryup"],   3),
        "VCP_Tightness":   round(_vcp_detail["vcp_tightness"],   3),
        "VCP_Position":    round(_vcp_detail["vcp_position"],    3),
        "RS_raw":    round(rs_pts_raw,      1),
        "Sect_raw":  round(rs_sect_pts_raw, 1),
        "Vol_raw":   round(vol_pts_raw,     1),
        "Inst_raw":  round(inst_pts_raw,    1),
        "VC_raw":    round(vc_pts_raw,      1),
        "Coil_raw":  round(coil_pts_raw,    1),
        "MA_raw":    round(ma_pts_raw,      1),
        "Prox_raw":  round(prox_pts_raw,    1),
        "Darvas":     round(darvas_pts,  1),
        "DarvasBox":  _darvas_result.get("box_high", np.nan),
        "DarvasLow":  _darvas_result.get("box_low",  np.nan),
        "DarvasInBox": _darvas_result.get("in_box",  False),
        "ADVTurnover":    round(_adv_turnover / 1e7, 2),
        "LiquidityScore": round(liquidity_score, 3),
        "SoftPenalty":    round(_soft_penalty, 1),
        "AboveSMA200":    above_long_trend,
        "Coil":      round(coil_pts,  1),
        "MA_Struct": round(ma_pts,    1),
        "Proximity": round(prox_pts,  1),
        "ATR_Pot":   round(atp_pts,   1),
        "Candle":    round(cdl_pts,   1),
        "Patterns":  ", ".join(candle_names) if candle_names else "—",
        "RS_Accel":   round(rs_accel, 4),
        "AccelScore": round(acc_score * 100, 1),
        "VCVE":       round(vcve, 3),
        "BasePos":    round(base_pos, 3),
        "Pos52W":     round(pos52w, 3),
        "Stability":  round(stability, 2),
        "Sweep":      sweep_bonus > 0,
        "VWMA20_OK":  vwap_bonus > 0,
        "DarvasBO":   round(darvas_bonus, 1),
        "UpVolSkew":   round(_uv_bonus, 1),
        "CPR":         round(_cpr_bonus, 1),
        "SpreadComp":  round(_sc_bonus, 1),
        "ATRExpOnset": round(_atr_exp_bonus, 1),
        "OI_Buildup":  round(oi_bonus, 1),
        "VolVelocity": round(_vol_velocity_score, 1),
        "RSDivergence": round(rs_div_bonus, 1),
        "CSRank5d":  round(cs_rs_score, 3),
        "AbsRS":     round(abs_rs_score, 3),
        "RSI7":      round(rsi_v, 1),
        "VolRatio":  round(vol_ratio, 2),
        "Rev_RSI_Pts":     round(_rev_rsi_pts,  1),
        "Rev_Vol_Pts":     round(_rev_vol_pts,  1),
        "Rev_Wash_Pts":    round(_rev_wash_pts, 1),
        "Rev_Tail_Pts":    round(_rev_tail_pts, 1),
        "Rev_Support_Pts": round(_rev_support_pts, 1),
        "WashoutDepth":    round(_washout_depth, 2),
        "CandleTailPos":   round(_t1_close_pos, 3),
        "VolZ":      round(vol_z, 2),
        "VolBOThr":  round(vol_bo_thresh / vol_mu, 2),
        "InstRatio": round(inst_ratio, 2),
        "VC_Ratio":  round(vc_ratio, 2),
        "ATR%":      round(atr_pct, 2),
        "RS_vs_Nifty": round(rs_score * 100, 1),
        "BO_Ext_ATR":  round(breakout_ext, 2),
        "Sector":    sect_name or "?",
        "EMA9":      round(e9_v, 2),
        "EMA20":     round(e20_v, 2),
        "EMA50":     round(e50_v, 2),
    }


# ─────────────────────────────────────────────
# LIVE QUOTE FETCH
# ─────────────────────────────────────────────

def fetch_live_quotes(all_keys: list) -> dict:
    url = "https://api.upstox.com/v2/market-quote/quotes"
    live_map = {}
    CHUNK = 50
    headers = get_headers()
    for i in range(0, len(all_keys), CHUNK):
        batch = all_keys[i:i + CHUNK]
        params = {"instrument_key": ",".join(batch)}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json().get("data", {})
            for _resp_key, v in data.items():
                ikey = v.get("instrument_token", "")
                if not ikey:
                    continue
                nk = normalize_key(ikey)
                ltp = v.get("last_price")
                if ltp is None:
                    continue
                ohlc = v.get("ohlc", {})
                live_map[nk] = {
                    "ltp": float(ltp),
                    "open": float(ohlc.get("open", ltp)),
                    "high": float(ohlc.get("high", ltp)),
                    "low": float(ohlc.get("low", ltp)),
                    "volume": float(v["volume"]) if v.get("volume") else None,
                    "oi": float(v["oi"]) if v.get("oi") else None,
                    "last_trade_time": v.get("last_trade_time"),
                }
        except Exception:
            pass
        time.sleep(0.12)
    return live_map


def patch_live_bar(df: pd.DataFrame, live: dict) -> pd.DataFrame:
    if not live:
        return df
    df = df.copy()
    last_idx = df.index[-1]
    _pre_high = float(df.at[last_idx, "high"]); _pre_low = float(df.at[last_idx, "low"])
    ltp = live.get("ltp"); high = live.get("high"); low = live.get("low")
    volume = live.get("volume"); oi = live.get("oi")
    if ltp is not None: df.at[last_idx, "close"] = ltp
    if high is not None: df.at[last_idx, "high"] = max(_pre_high, high)
    if low is not None: df.at[last_idx, "low"] = min(_pre_low, low)
    if volume is not None: df.at[last_idx, "volume"] = volume
    if oi is not None and "oi" in df.columns: df.at[last_idx, "oi"] = oi
    _ph = float(df.at[last_idx, "high"]); _pl = float(df.at[last_idx, "low"]); _pc = float(df.at[last_idx, "close"])
    if _ph < _pl: df.at[last_idx, "high"] = _pre_high; df.at[last_idx, "low"] = _pre_low
    if _pc > float(df.at[last_idx, "high"]): df.at[last_idx, "high"] = _pc
    if _pc < float(df.at[last_idx, "low"]): df.at[last_idx, "low"] = _pc
    return df


# ─────────────────────────────────────────────
# MASTER INSTRUMENTS
# ─────────────────────────────────────────────

_master_cache = {"df": None, "ts": 0}

def get_live_master():
    if _master_cache["df"] is not None and time.time() - _master_cache["ts"] < 3600:
        return _master_cache["df"]
    try:
        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
                data = json.load(gz)
            df = pd.DataFrame(data)
            _master_cache["df"] = df; _master_cache["ts"] = time.time()
            return df
    except Exception:
        pass
    return _master_cache["df"] or pd.DataFrame()

_nifty50_cache = {"syms": None, "ts": 0}

def get_nifty50_live():
    if _nifty50_cache["syms"] is not None and time.time() - _nifty50_cache["ts"] < 14400:
        return _nifty50_cache["syms"]
    try:
        headers_nse = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://www.nseindia.com/"}
        r = requests.get("https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050", headers=headers_nse, timeout=10)
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
        "HDFCLIFE","APOLLOHOSP","BAJAJ-AUTO","UPL","SHREECEM","HINDUNILVR","TATACONSUM"
    }
    _nifty50_cache["syms"] = fallback; _nifty50_cache["ts"] = time.time()
    return fallback

_mkt_cache = {"data": {}, "ts": 0}

def get_market_context():
    if _mkt_cache["data"] and time.time() - _mkt_cache["ts"] < 900:
        return _mkt_cache["data"]
    out = dict(nifty_r5=None, nifty_r20=None, nifty_above_20dma=True, nifty_above_50dma=True,
               regime="BULL", vix_level=None, vix_falling=True, vix_median=14.5, vix_sigma=4.5,
               sector_returns={}, sector_returns_10d={}, top_sectors=set(), market_ok=True, market_notes=[])
    try:
        import yfinance as yf
        n = yf.download("^NSEI", period="365d", interval="1d", progress=False)
        if not n.empty:
            c = n["Close"].squeeze()
            out["nifty_r5"] = float(c.iloc[-1]/c.iloc[-6]-1) if len(c)>=6 else None
            out["nifty_r20"] = float(c.iloc[-1]/c.iloc[-21]-1) if len(c)>=21 else None
            dma20 = float(c.tail(20).mean()); dma50 = float(c.tail(50).mean()) if len(c)>=50 else dma20
            out["nifty_above_20dma"] = float(c.iloc[-1]) > dma20
            out["nifty_above_50dma"] = float(c.iloc[-1]) > dma50
            _nifty_atr = float(c.diff().abs().tail(14).mean())
            _dma20_now = float(c.tail(20).mean())
            _dma20_10d = float(c.iloc[-11:-1].mean()) if len(c)>=11 else _dma20_now
            _gap = float(c.iloc[-1]) - dma50
            if _gap > 0 and (_dma20_now - _dma20_10d) > 0: out["regime"] = "BULL"
            elif _gap < -_nifty_atr: out["regime"] = "BEAR"
            else: out["regime"] = "CHOP"
    except Exception:
        pass
    try:
        import yfinance as yf
        v = yf.download("^INDIAVIX", period="365d", interval="1d", progress=False)
        if not v.empty:
            vc = v["Close"].squeeze()
            out["vix_level"] = round(float(vc.iloc[-1]), 2)
            if len(vc) >= 5:
                _vix_slope = float(np.polyfit(range(5), vc.tail(5).values, 1)[0])
                out["vix_falling"] = _vix_slope < 0
            out["vix_median"] = round(float(vc.median()), 2) if len(vc)>=20 else 14.5
            out["vix_sigma"] = round(float(vc.std()), 2) if len(vc)>=20 else 4.5
    except Exception:
        pass

    sr_5d = {}; sr_10d = {}
    def _fetch_sector(name_ticker):
        _name, _ticker = name_ticker
        try:
            import yfinance as yf
            s = yf.download(_ticker, period="60d", interval="1d", progress=False)
            if not s.empty:
                sc = s["Close"].squeeze()
                r5 = float(sc.iloc[-1]/sc.iloc[-6]-1) if len(sc)>=6 else None
                r10 = float(sc.iloc[-1]/sc.iloc[-11]-1) if len(sc)>=11 else None
                return _name, r5, r10
        except Exception:
            pass
        return _name, None, None

    with ThreadPoolExecutor(max_workers=8) as exc:
        for _sname, _r5, _r10 in exc.map(_fetch_sector, SECTOR_TICKERS.items()):
            if _r5 is not None: sr_5d[_sname] = _r5
            if _r10 is not None: sr_10d[_sname] = _r10
    out["sector_returns"] = sr_5d; out["sector_returns_10d"] = sr_10d
    if sr_5d:
        out["top_sectors"] = {k for k,_ in sorted(sr_5d.items(), key=lambda x:x[1], reverse=True)[:3]}
    out["market_ok"] = out["nifty_above_20dma"]
    _mkt_cache["data"] = out; _mkt_cache["ts"] = time.time()
    return out


# ─────────────────────────────────────────────
# CROSS-SECTIONAL PRE-COMPUTATION  [F-02, F-10]
# O(n log n) CDF ranking via scipy rankdata.
# Replaces the old min-max inline block in run_extraction.
# ─────────────────────────────────────────────

def _cdf_rank_dict(raw_dict: dict) -> dict:
    """Convert raw-value dict → universe CDF percentile dict. O(n log n).  [F-02]"""
    if len(raw_dict) < 3:
        return {k: 0.5 for k in raw_dict}
    syms = list(raw_dict.keys())
    vals = np.array([raw_dict[s] for s in syms], dtype=float)
    pcts = rankdata(vals, method="average") / len(vals)
    return {s: float(p) for s, p in zip(syms, pcts)}


def compute_cs_ranks(STATE: dict) -> None:
    """
    Compute ALL cross-sectional ranks across the loaded universe.
    Populates:
      cs_rs_5d, cs_rs_20d          — momentum RS (CDF, not min-max)
      cs_bb_squeeze                — BB width compression universe rank
      cs_vol_dryup                 — volume dry-up universe rank
      cs_clv_accum                 — CLV accumulation universe rank
      cs_vcp                       — VCP score universe rank
      breadth_cache, breadth_hist  — market breadth
      sector_returns, sector_returns_10d
    """
    cache = STATE.get("raw_data_cache", {})
    if len(cache) < 3:
        return

    # 1. RS 5d and 20d — true CDF, not min-max  [F-02]
    _r5_raw:  dict = {}
    _r20_raw: dict = {}
    for sym, df in cache.items():
        c = df["close"]
        if len(c) >= 6:
            _r5_raw[sym]  = float(c.iloc[-1] / c.iloc[-6]  - 1)
        if len(c) >= 21:
            _r20_raw[sym] = float(c.iloc[-1] / c.iloc[-21] - 1)
    STATE["cs_rs_5d"]  = _cdf_rank_dict(_r5_raw)
    STATE["cs_rs_20d"] = _cdf_rank_dict(_r20_raw)

    # 2. BB squeeze (self-calibrated → universe CDF)
    _bb_raw: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < 30: continue
            sma = c.rolling(20).mean()
            std = c.rolling(20).std()
            bb_w = (2.0 * std / sma.replace(0, np.nan)).dropna()
            if len(bb_w) < 10: continue
            cur = float(bb_w.iloc[-1])
            _bb_raw[sym] = float((bb_w.iloc[:-1] <= cur).mean())
        except Exception:
            pass
    STATE["cs_bb_squeeze"] = _cdf_rank_dict(_bb_raw)

    # 3. Volume dry-up (self-calibrated → universe CDF)
    _vdu_raw: dict = {}
    for sym, df in cache.items():
        try:
            v = df["volume"].replace(0, np.nan).dropna()
            if len(v) < 25: continue
            ratio = float(v.tail(5).mean()) / (float(v.tail(20).mean()) + 1e-9)
            hist_ratios = (v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)).dropna()
            if len(hist_ratios) >= 5:
                _vdu_raw[sym] = float((hist_ratios >= ratio).mean())
            else:
                _vdu_raw[sym] = float(np.clip(1.0 - ratio, 0.0, 1.0))
        except Exception:
            pass
    STATE["cs_vol_dryup"] = _cdf_rank_dict(_vdu_raw)

    # 4. CLV accumulation (self-calibrated → universe CDF)
    _clv_raw: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
            if len(c) < 25: continue
            hl_rng  = (h - l).replace(0, np.nan)
            clv     = ((c - l) - (h - c)) / hl_rng
            mf      = clv.fillna(0) * v
            mf_norm = mf / v.rolling(20).mean().replace(0, np.nan)
            roll_mf = mf_norm.rolling(20).sum().dropna()
            if len(roll_mf) < 5: continue
            cur = float(roll_mf.iloc[-1])
            _clv_raw[sym] = float((roll_mf.iloc[:-1] <= cur).mean())
        except Exception:
            pass
    STATE["cs_clv_accum"] = _cdf_rank_dict(_clv_raw)

    # 5. VCP raw score → universe CDF
    _vcp_raw: dict = {}
    for sym, df in cache.items():
        try:
            if len(df) < 60: continue
            c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]
            tr  = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False).mean()
            res = detect_vcp(c, h, l, v, atr)
            _vcp_raw[sym] = float(res.get("vcp_score", 0.0))
        except Exception:
            pass
    STATE["cs_vcp"] = _cdf_rank_dict(_vcp_raw)

    # 6. Market breadth (% stocks above own EMA20)
    _above = 0; _total = 0
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < 20: continue
            e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            _total += 1
            if float(c.iloc[-1]) > e20:
                _above += 1
        except Exception:
            pass
    if _total >= 10:
        breadth = _above / _total
        STATE["breadth_cache"] = breadth
        STATE["breadth_hist"]  = (STATE.get("breadth_hist", []) + [breadth])[-200:]

    # 7. Sector returns from universe cache
    _r5a: dict = {}; _r10a: dict = {}
    for sym, df in cache.items():
        try:
            sec = get_sector(sym)
            if sec is None: continue
            c = df["close"]
            if len(c) >= 6:  _r5a.setdefault(sec, []).append(float(c.iloc[-1]/c.iloc[-6]-1))
            if len(c) >= 11: _r10a.setdefault(sec, []).append(float(c.iloc[-1]/c.iloc[-11]-1))
        except Exception:
            pass
    _cache_sr5  = {s: float(np.mean(v)) for s, v in _r5a.items()}
    _cache_sr10 = {s: float(np.mean(v)) for s, v in _r10a.items()}
    STATE["sector_returns"]     = {**STATE.get("mkt", {}).get("sector_returns", {}),     **_cache_sr5}
    STATE["sector_returns_10d"] = {**STATE.get("mkt", {}).get("sector_returns_10d", {}), **_cache_sr10}


def _apply_coverage_score(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process screener output — add score_percentile column.
    O(n log n) via rankdata.  [F-03]
    """
    if df_out.empty or "Score" not in df_out.columns:
        return df_out
    _scores = df_out["Score"].values.astype(float)
    df_out["score_percentile"] = (
        rankdata(_scores, method="average") / max(len(_scores), 1) * 100
    ).round(1)
    for _col in ("signal_strength", "coverage"):
        if _col in df_out.columns:
            _vals = df_out[_col].values.astype(float)
            df_out[f"{_col}_pct"] = (
                rankdata(_vals, method="average") / max(len(_vals), 1) * 100
            ).round(1)
    return df_out


def bootstrap_calibration_from_db(STATE: dict, db_path: str) -> None:
    """
    Seed STATE from the calibration table on startup.  [F-10]
    Pre-warms param_registry, breadth_hist, per_stock_winrate so
    self-calibration works immediately instead of requiring 200 warm-up calls.
    """
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(
            "SELECT symbol, score, forward_ret FROM calibration ORDER BY ts ASC"
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return
    if not rows:
        return

    winrate_hits:  dict = {}
    winrate_total: dict = {}
    score_vals = []
    for sym, score, fwd_ret in rows:
        if score is not None:
            score_vals.append(score)
        if fwd_ret is not None:
            winrate_total[sym] = winrate_total.get(sym, 0) + 1
            if fwd_ret > 0:
                winrate_hits[sym] = winrate_hits.get(sym, 0) + 1

    # Seed per-stock win-rate
    wr_map = {}
    for sym in winrate_total:
        n = winrate_total[sym]; h = winrate_hits.get(sym, 0)
        if n >= 3:
            wr_map[sym] = float(np.clip(h / n, 0.35, 0.70))
    if wr_map:
        STATE.setdefault("per_stock_winrate", {}).update(wr_map)

    # Seed breadth_hist proxy from score distribution chunks
    if len(score_vals) >= 10:
        chunk = 20
        breadth_proxy = []
        for i in range(0, len(score_vals), chunk):
            sl = score_vals[i:i+chunk]
            breadth_proxy.append(sum(1 for s in sl if s > 50) / max(len(sl), 1))
        existing = STATE.get("breadth_hist", [])
        STATE["breadth_hist"] = (existing + breadth_proxy)[-200:]


# ─────────────────────────────────────────────
# BACKGROUND EXTRACTION
# ─────────────────────────────────────────────

def run_extraction(targets_dict, min_avg_vol):
    status = STATE["extraction_status"]
    status.update({"running": True, "done": 0, "total": len(targets_dict), "errors": 0, "rate_limited": 0, "log": []})
    STATE["raw_data_cache"] = {}
    STATE["score_cache"] = {}
    STATE["cs_rs_5d"] = {}; STATE["cs_rs_20d"] = {}
    STATE["cs_bb_squeeze"] = {}; STATE["cs_vol_dryup"] = {}
    STATE["cs_clv_accum"] = {}; STATE["cs_vcp"] = {}
    STATE["breadth_cache"] = None

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')

    # fetch live quotes first
    live_quotes = fetch_live_quotes(list(targets_dict.values()))
    STATE["live_quotes_cache"] = live_quotes
    STATE["last_live_refresh"] = time.time()

    sym_keys = list(targets_dict.items())
    if min_avg_vol > 0 and live_quotes:
        _filtered = []
        for _sym, _key in sym_keys:
            _lq = live_quotes.get(normalize_key(_key))
            if _lq is None or _lq.get("volume") is None:
                _filtered.append((_sym, _key))
            elif float(_lq["volume"]) >= min_avg_vol * 0.20:
                _filtered.append((_sym, _key))
        sym_keys = _filtered
    status["total"] = len(sym_keys)

    FETCH_WORKERS = 4; FETCH_DELAY = 0.15; FETCH_RETRIES = 4; FETCH_BACKOFF = 0.5

    def _fetch_one(sym_key_pair):
        sym, key = sym_key_pair
        url = (f"https://api.upstox.com/v2/historical-candle/"
               f"{urllib.parse.quote(key)}/day/{end_date}/{start_date}")
        delay = FETCH_BACKOFF
        headers = get_headers()
        for attempt in range(FETCH_RETRIES + 1):
            try:
                time.sleep(FETCH_DELAY)
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 429:
                    if attempt < FETCH_RETRIES:
                        time.sleep(delay); delay *= 2; continue
                    return sym, None, "HTTP 429"
                if r.status_code != 200:
                    return sym, None, f"HTTP {r.status_code}"
                raw = r.json().get("data", {}).get("candles", [])
                if not raw:
                    return sym, None, "empty"
                df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume","oi"])
                df = to_ascending(df)
                live_q = live_quotes.get(normalize_key(key))
                if live_q:
                    df = patch_live_bar(df, live_q)
                return sym, df, None
            except requests.exceptions.Timeout:
                if attempt < FETCH_RETRIES:
                    time.sleep(delay); delay *= 2; continue
                return sym, None, "timeout"
            except Exception as e:
                return sym, None, str(e)
        return sym, None, "max retries"

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, pair): pair for pair in sym_keys}
        for future in as_completed(futures):
            sym, df, err = future.result()
            status["done"] += 1
            if err:
                if "429" in str(err): status["rate_limited"] += 1
                elif err not in ("empty",): status["errors"] += 1
                continue
            if df is None:
                continue
            df_clean = df[df["volume"] > 0].copy() if "volume" in df.columns else df.copy()
            if len(df_clean) < 30:
                continue
            if min_avg_vol > 0 and len(df_clean) >= 5:
                if float(df_clean["volume"].tail(20).mean()) < min_avg_vol:
                    continue
            STATE["raw_data_cache"][sym] = df_clean

    # Compute cross-sectional ranks via O(n log n) CDF ranking  [F-02]
    compute_cs_ranks(STATE)

    status["running"] = False
    # Auto-save snapshot to DB
    try:
        universe = STATE.get("_last_universe", "unknown")
        rows_to_save = []
        mkt = STATE.get("mkt", {})
        for sym, df_raw in STATE["raw_data_cache"].items():
            cached = STATE["score_cache"].get(sym, {}).get("result")
            if cached:
                live = STATE["live_quotes_cache"].get(normalize_key(STATE["targets"].get(sym,"")), {})
                ltp = live.get("ltp") or float(df_raw["close"].iloc[-1])
                rows_to_save.append({"Ticker": sym, "LTP": round(ltp, 2), **cached})
        if rows_to_save:
            save_snapshot(rows_to_save, universe)
    except Exception as _e:
        pass


def refresh_live_prices_bg():
    now = time.time()
    if now - STATE["last_live_refresh"] < LIVE_REFRESH_SEC:
        return
    if not STATE["raw_data_cache"] or not STATE["targets"]:
        return
    keys = list(STATE["targets"].values())
    live = fetch_live_quotes(keys)
    if not live:
        return
    for sym, df in STATE["raw_data_cache"].items():
        key = STATE["targets"].get(sym)
        if not key: continue
        live_q = live.get(normalize_key(key))
        if not live_q: continue
        STATE["raw_data_cache"][sym] = patch_live_bar(df, live_q)
    STATE["live_quotes_cache"] = live
    STATE["last_live_refresh"] = time.time()
    # Invalidate score cache on price change
    STATE["score_cache"] = {}


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("static/login.html", encoding="utf-8") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/options", response_class=HTMLResponse)
async def options_page():
    with open("static/pages/options.html", encoding="utf-8") as f:
        return f.read()

@app.get("/fundamentals", response_class=HTMLResponse)
async def fundamentals_page():
    with open("static/pages/fundamentals.html", encoding="utf-8") as f:
        return f.read()

@app.get("/ml", response_class=HTMLResponse)
async def ml_page():
    with open("static/pages/ml.html", encoding="utf-8") as f:
        return f.read()

@app.post("/api/token")
async def set_token(body: dict):
    tok = body.get("token", "").strip()
    if not tok:
        raise HTTPException(400, "Empty token")
    STATE["token"] = tok
    # Save to file
    try:
        with open(".upstox_token", "w") as f: f.write(tok)
    except Exception:
        pass
    return {"status": "ok", "token_prefix": tok[:16]}

@app.get("/api/token/status")
async def token_status():
    # Try loading saved token
    if not STATE["token"]:
        try:
            if os.path.exists(".upstox_token"):
                t = open(".upstox_token").read().strip()
                if t: STATE["token"] = t
        except Exception:
            pass
    return {"connected": bool(STATE["token"]),
            "prefix": STATE["token"][:16] if STATE["token"] else ""}

@app.get("/api/universe")
async def get_universe(universe: str = "Nifty 50"):
    master_df = get_live_master()
    if master_df.empty:
        raise HTTPException(503, "Could not load instrument master")
    eq = master_df[(master_df['exchange'] == 'NSE') & (master_df['instrument_type'] == 'EQ')]
    if universe == "Nifty 50":
        nifty50 = get_nifty50_live()
        df = eq[eq['trading_symbol'].isin(nifty50)]
    elif universe == "F&O Stocks":
        fo_contracts = master_df[master_df['segment'].astype(str).str.contains("FO", na=False)]
        fo_underlyings = fo_contracts['underlying_symbol'].dropna().astype(str).unique()
        df = eq[eq['trading_symbol'].isin(fo_underlyings)]
    else:
        df = eq
    targets = {row['trading_symbol']: row['instrument_key'] for _, row in df.iterrows()}
    STATE["targets"] = targets
    return {"count": len(targets), "symbols": list(targets.keys())[:20]}

@app.post("/api/extract")
async def start_extraction(body: dict, background_tasks: BackgroundTasks):
    universe = body.get("universe", "Nifty 50")
    min_vol = body.get("min_avg_vol", 100000)
    rsi_p = body.get("rsi_period", 7)
    STATE["rsi_period"] = int(rsi_p)
    STATE["min_avg_vol"] = int(min_vol)
    STATE["sector_cap_enabled"] = body.get("sector_cap_enabled", False)

    if STATE["extraction_status"]["running"]:
        return {"status": "already_running"}

    # Build targets
    master_df = get_live_master()
    if master_df.empty:
        raise HTTPException(503, "Master list unavailable")
    eq = master_df[(master_df['exchange'] == 'NSE') & (master_df['instrument_type'] == 'EQ')]
    if universe == "Nifty 50":
        nifty50 = get_nifty50_live()
        df = eq[eq['trading_symbol'].isin(nifty50)]
    elif universe == "F&O Stocks":
        fo_contracts = master_df[master_df['segment'].astype(str).str.contains("FO", na=False)]
        fo_underlyings = fo_contracts['underlying_symbol'].dropna().astype(str).unique()
        df = eq[eq['trading_symbol'].isin(fo_underlyings)]
    else:
        df = eq
    targets = {row['trading_symbol']: row['instrument_key'] for _, row in df.iterrows()}
    STATE["targets"] = targets

    # Load market context in background too
    mkt = get_market_context()
    STATE["mkt"] = mkt
    STATE["sector_returns"] = mkt.get("sector_returns", {})
    STATE["sector_returns_10d"] = mkt.get("sector_returns_10d", {})
    STATE["top_sectors"] = mkt.get("top_sectors", set())

    STATE["_last_universe"] = universe
    background_tasks.add_task(run_extraction, targets, min_vol)
    return {"status": "started", "total": len(targets)}

@app.get("/api/extraction/status")
async def extraction_status():
    s = STATE["extraction_status"]
    return {**s, "cached": len(STATE["raw_data_cache"]),
            "live_quotes": len(STATE["live_quotes_cache"]),
            "last_refresh": STATE["last_live_refresh"]}

@app.get("/api/screener")
async def get_screener(sort_by: str = "CompositeRank", horizon: str = "ALL"):
    # Refresh live prices if needed
    refresh_live_prices_bg()

    mkt = STATE["mkt"] or get_market_context()
    nifty_r5 = mkt.get("nifty_r5"); nifty_r20 = mkt.get("nifty_r20")

    rows = []
    _min_vol = STATE.get("min_avg_vol", 0)
    for sym, df_raw in STATE["raw_data_cache"].items():
        try:
            if _min_vol > 0 and "volume" in df_raw.columns and len(df_raw) >= 5:
                if float(df_raw["volume"].tail(20).mean()) < _min_vol:
                    continue

            live = STATE["live_quotes_cache"].get(normalize_key(STATE["targets"].get(sym, "")), {})

            # Per-stock score cache invalidation via LTP fingerprint
            _ltp_now = live.get("ltp") or float(df_raw["close"].iloc[-1])
            _vol_now = live.get("volume") or float(df_raw["volume"].iloc[-1]) if "volume" in df_raw.columns else 0
            _cache_entry = STATE["score_cache"].get(sym)
            if (_cache_entry and abs(_cache_entry.get("ltp", 0) - _ltp_now) < 0.01 * _ltp_now
                    and _cache_entry.get("result") is not None):
                result = _cache_entry["result"]
            else:
                result = score_stock_dual(sym, df_raw, live, nifty_r5, nifty_r20)
                STATE["score_cache"][sym] = {"result": result, "ltp": _ltp_now, "vol": _vol_now}

            if result is None:
                continue
            live_ltp = float(_ltp_now)
            rows.append({
                "Ticker": sym,
                "LTP": round(live_ltp, 2),
                "DayHigh": round(float(live.get("high", df_raw["high"].iloc[-1])), 2),
                "DayLow": round(float(live.get("low", df_raw["low"].iloc[-1])), 2),
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
    df_out = _apply_coverage_score(df_out)   # adds score_percentile  [F-03]
    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by, ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", df_out.index + 1)

    # Sector cap
    if STATE.get("sector_cap_enabled") and "Sector" in df_out.columns:
        seen = set(); capped = []
        for _, row in df_out.iterrows():
            sec = str(row.get("Sector", "?"))
            if sec == "?" or sec not in seen:
                capped.append(row)
                if sec != "?": seen.add(sec)
        df_out = pd.DataFrame(capped).reset_index(drop=True)
        df_out["Rank"] = df_out.index + 1

    if horizon != "ALL":
        df_out = df_out[df_out["Horizon"] == horizon].reset_index(drop=True)
        df_out["Rank"] = df_out.index + 1

    # Replace NaN/inf
    df_out = df_out.replace({np.nan: None, np.inf: None, -np.inf: None})

    sector_ret = {}
    for sec, v in STATE["sector_returns"].items():
        sector_ret[sec] = round(v * 100, 2)

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
    return {
        "rsi_period": STATE["rsi_period"],
        "min_avg_vol": STATE["min_avg_vol"],
        "sector_cap_enabled": STATE["sector_cap_enabled"],
    }

@app.post("/api/config")
async def set_config(body: dict):
    if "rsi_period" in body: STATE["rsi_period"] = int(body["rsi_period"])
    if "min_avg_vol" in body: STATE["min_avg_vol"] = int(body["min_avg_vol"])
    if "sector_cap_enabled" in body: STATE["sector_cap_enabled"] = bool(body["sector_cap_enabled"])
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════
# DATABASE — SQLite persistence (screener snapshots + calibration)
# ══════════════════════════════════════════════════════════════════
DB_PATH = pathlib.Path("monarch_data.db")

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT    NOT NULL,
            universe  TEXT    NOT NULL,
            row_json  TEXT    NOT NULL
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
        CREATE INDEX IF NOT EXISTS idx_cal_sym ON calibration(symbol);

        CREATE TABLE IF NOT EXISTS watchlist (
            symbol  TEXT PRIMARY KEY,
            note    TEXT,
            added   TEXT
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol  TEXT NOT NULL,
            cond    TEXT NOT NULL,
            value   REAL NOT NULL,
            fired   INTEGER DEFAULT 0,
            ts      TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()
bootstrap_calibration_from_db(STATE, DB_PATH)

import json as _json
from datetime import datetime as _dt

def save_snapshot(rows: list, universe: str):
    """Persist a screener run to SQLite."""
    if not rows:
        return
    ts = _dt.now().isoformat()
    conn = get_db()
    for row in rows:
        conn.execute(
            "INSERT INTO snapshots(ts, universe, row_json) VALUES (?,?,?)",
            (ts, universe, _json.dumps(row))
        )
    conn.commit()
    conn.close()

# ── SNAPSHOT ENDPOINTS ──────────────────────────────────────────
@app.get("/api/db/snapshots")
async def list_snapshots(limit: int = 20):
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT ts, universe, COUNT(*) as n FROM snapshots GROUP BY ts ORDER BY ts DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return {"snapshots": [dict(r) for r in rows]}

@app.get("/api/db/snapshot")
async def get_snapshot(ts: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT row_json FROM snapshots WHERE ts=? ORDER BY rowid",
        (ts,)
    ).fetchall()
    conn.close()
    data = [_json.loads(r["row_json"]) for r in rows]
    return {"ts": ts, "rows": data, "count": len(data)}

@app.delete("/api/db/snapshots/old")
async def purge_old_snapshots(keep_days: int = 30):
    from datetime import timedelta
    cutoff = (_dt.now() - timedelta(days=keep_days)).isoformat()
    conn = get_db()
    n = conn.execute("DELETE FROM snapshots WHERE ts < ?", (cutoff,)).rowcount
    conn.commit()
    conn.close()
    return {"deleted": n}

# ── CALIBRATION ENDPOINTS ────────────────────────────────────────
@app.post("/api/db/calibration")
async def add_calibration(body: dict):
    """Log a signal for forward-return tracking."""
    conn = get_db()
    conn.execute(
        "INSERT INTO calibration(ts,symbol,score,forward_ret,horizon,setup,regime) VALUES(?,?,?,?,?,?,?)",
        (_dt.now().isoformat(), body.get("symbol",""), body.get("score"),
         body.get("forward_ret"), body.get("horizon",""), body.get("setup",""), body.get("regime",""))
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/api/db/calibration/stats")
async def calibration_stats():
    """Self-calibration: win rate and avg return by score quartile."""
    conn = get_db()
    rows = conn.execute(
        "SELECT symbol, score, forward_ret, horizon, setup, regime FROM calibration WHERE forward_ret IS NOT NULL"
    ).fetchall()
    conn.close()
    if not rows:
        return {"error": "no data yet — add signals and update their forward returns"}

    import numpy as np
    data = [dict(r) for r in rows]
    scores = [r["score"] for r in data if r["score"] is not None]
    rets   = [r["forward_ret"] for r in data if r["forward_ret"] is not None]

    if len(scores) < 4:
        return {"count": len(data), "note": "need at least 4 records for quartile analysis"}

    # Quartile breakdown
    q25, q50, q75 = float(np.percentile(scores, 25)), float(np.percentile(scores, 50)), float(np.percentile(scores, 75))
    quartiles = {"Q1 (0-25%)": [], "Q2 (25-50%)": [], "Q3 (50-75%)": [], "Q4 (75-100%)": []}
    for r in data:
        s, ret = r.get("score"), r.get("forward_ret")
        if s is None or ret is None: continue
        if s <= q25:   quartiles["Q1 (0-25%)"].append(ret)
        elif s <= q50: quartiles["Q2 (25-50%)"].append(ret)
        elif s <= q75: quartiles["Q3 (50-75%)"].append(ret)
        else:          quartiles["Q4 (75-100%)"].append(ret)

    result = {}
    for qname, qrets in quartiles.items():
        if qrets:
            result[qname] = {
                "n": len(qrets),
                "avg_ret": round(float(np.mean(qrets)), 3),
                "win_rate": round(float(np.mean([1 if r > 0 else 0 for r in qrets])) * 100, 1),
                "avg_score_thresh": round(q25 if "Q1" in qname else q50 if "Q2" in qname else q75 if "Q3" in qname else q75, 1)
            }

    return {
        "count": len(data),
        "overall_win_rate": round(float(np.mean([1 if r > 0 else 0 for r in rets])) * 100, 1),
        "overall_avg_ret": round(float(np.mean(rets)), 3),
        "quartiles": result,
        "score_percentiles": {"p25": round(q25,1), "p50": round(q50,1), "p75": round(q75,1)},
    }

@app.post("/api/db/calibration/update_returns")
async def update_forward_returns():
    """
    Auto-fill forward returns for logged signals using cached OHLCV data.
    Uses 5-day forward close return from the date logged.
    """
    conn = get_db()
    pending = conn.execute(
        "SELECT id, symbol, ts FROM calibration WHERE forward_ret IS NULL"
    ).fetchall()
    updated = 0
    for rec in pending:
        sym = rec["symbol"]
        log_ts = _dt.fromisoformat(rec["ts"])
        df = STATE["raw_data_cache"].get(sym)
        if df is None or "time" not in df.columns:
            continue
        df_t = df.copy()
        df_t["time"] = pd.to_datetime(df_t["time"])
        df_t = df_t.sort_values("time").reset_index(drop=True)
        # Find bar closest to log date
        idx = (df_t["time"] - log_ts).abs().idxmin()
        fwd_idx = idx + 5
        if fwd_idx < len(df_t):
            entry = float(df_t.loc[idx, "close"])
            exit_p = float(df_t.loc[fwd_idx, "close"])
            fwd_ret = round((exit_p / entry - 1) * 100, 3)
            conn.execute("UPDATE calibration SET forward_ret=? WHERE id=?", (fwd_ret, rec["id"]))
            updated += 1
    conn.commit()
    conn.close()
    return {"updated": updated, "pending": len(pending)}

# ── WATCHLIST ENDPOINTS ──────────────────────────────────────────
@app.get("/api/watchlist")
async def get_watchlist():
    conn = get_db()
    rows = conn.execute("SELECT * FROM watchlist ORDER BY added DESC").fetchall()
    conn.close()
    return {"symbols": [dict(r) for r in rows]}

@app.post("/api/watchlist")
async def add_to_watchlist(body: dict):
    sym = body.get("symbol","").upper().strip()
    if not sym:
        raise HTTPException(400, "Symbol required")
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO watchlist(symbol,note,added) VALUES(?,?,?)",
        (sym, body.get("note",""), _dt.now().isoformat())
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "symbol": sym}

@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    conn = get_db()
    conn.execute("DELETE FROM watchlist WHERE symbol=?", (symbol.upper(),))
    conn.commit()
    conn.close()
    return {"status": "ok"}

# ── NEWS FEED ────────────────────────────────────────────────────
@app.get("/api/news")
async def get_news(symbol: str = ""):
    """Fetch NSE/market news via RSS. Filters by symbol if provided."""
    import feedparser, html
    feeds = [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ]
    articles = []
    for url in feeds:
        try:
            f = feedparser.parse(url)
            for e in f.entries[:15]:
                title = html.unescape(getattr(e, "title", ""))
                summary = html.unescape(getattr(e, "summary", ""))[:200]
                link = getattr(e, "link", "#")
                pub = getattr(e, "published", "")
                if symbol and symbol.upper() not in (title + summary).upper():
                    continue
                articles.append({"title": title, "summary": summary, "link": link, "pub": pub, "source": url.split("/")[2]})
        except Exception:
            pass
    return {"articles": articles[:30], "symbol": symbol}

# ── CHART DATA ───────────────────────────────────────────────────
@app.get("/api/chart/{symbol}")
async def get_chart_data(symbol: str):
    """Return OHLCV + indicators for Bloomberg-style chart."""
    df = STATE["raw_data_cache"].get(symbol.upper())
    if df is None:
        raise HTTPException(404, f"{symbol} not in cache — run extraction first")

    df = df.copy().tail(120)  # last 120 bars
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")

    # Indicators
    c = df["close"]
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean().round(2)
    df["ema20"] = c.ewm(span=20, adjust=False).mean().round(2)
    df["ema50"] = c.ewm(span=50, adjust=False).mean().round(2)

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean().round(2)

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/STATE["rsi_period"], adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/STATE["rsi_period"], adjust=False).mean()
    rs   = gain / loss.replace(0, float("nan"))
    df["rsi"] = (100 - 100/(1+rs)).round(1)

    # Volume MA
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean().round(0)

    # Replace NaN
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

    cols = ["time","open","high","low","close","volume","ema9","ema20","ema50","atr","rsi","vol_ma20"]
    cols = [c for c in cols if c in df.columns]
    return {"symbol": symbol.upper(), "bars": df[cols].to_dict("records")}

# ── SCORE EXPLANATION ────────────────────────────────────────────
@app.get("/api/explain/{symbol}")
async def explain_score(symbol: str):
    """Plain-English explanation of every factor in the score."""
    sym = symbol.upper()
    df_raw = STATE["raw_data_cache"].get(sym)
    if df_raw is None:
        raise HTTPException(404, "Run extraction first")

    live = STATE["live_quotes_cache"].get(
        normalize_key(STATE["targets"].get(sym, "")), {}
    )
    cached = STATE["score_cache"].get(sym, {}).get("result")
    if not cached:
        raise HTTPException(404, "Score not computed yet — run screener first")

    lines = []
    def line(label, value, why, implication):
        lines.append({"label": label, "value": value, "why": why, "implication": implication})

    s = cached
    ltp = live.get("ltp") or float(df_raw["close"].iloc[-1])

    line("LTP", f"₹{ltp:.2f}", "Last traded price", "Current market price")
    line("Score", s.get("Score","—"), "Composite quality score 0–100", "Higher = better setup quality")
    line("Setup", s.get("Setup","—"), "Pattern detected on price/volume", "Defines entry strategy and target multiplier")
    line("Horizon", s.get("Horizon","—"), "Expected trade duration", "Match to your holding capacity")
    line("EMI", s.get("EMI","—"), "Score × ATR% — reward-adjusted quality", "Higher EMI = better quality per unit of volatility")
    line("RSI", s.get("RSI","—"), f"Relative Strength Index ({STATE['rsi_period']}-period)", ">70 overbought, <30 oversold, 40-60 ideal for entries")
    line("ATR%", s.get("ATR%","—"), "Average True Range as % of price (14-day)", "Measures daily volatility — sizing and stop placement")
    line("Vol Z-Score", s.get("VolZ","—"), "Today's volume vs 20-day average in σ units", ">1.5σ = unusual accumulation, <−1σ = drying up")
    line("R:R", s.get("RR","—"), "Risk:Reward ratio (Target÷Stop)", ">2.0 is acceptable, >3.0 is excellent")
    line("Entry", f"₹{s.get('Entry',0):.2f}" if s.get('Entry') else "—", "Suggested entry price", "Price at which the setup triggers")
    line("Target", f"₹{s.get('Target',0):.2f}" if s.get('Target') else "—", "Price target based on ATR multiples", "Exit here to realise the expected gain")
    line("Stop", f"₹{s.get('Stop',0):.2f}" if s.get('Stop') else "—", "Stop-loss level", "Exit if price hits this — invalidates the setup")
    line("Sector", s.get("Sector","—"), "GICS sector classification", "Check if sector is in top performers for confirmation")

    regime = STATE.get("mkt",{}).get("regime","BULL")
    vix = STATE.get("mkt",{}).get("vix_level")
    line("Market Regime", regime,
         "BULL=Nifty>50DMA+rising, BEAR=below 50DMA, CHOP=mixed",
         "BULL: take all setups. CHOP: only high-score Breakout. BEAR: avoid Breakout, focus Reversal.")
    if vix:
        line("India VIX", f"{vix:.1f}", "Fear index — higher = more uncertainty",
             "<14 = calm market, 14-20 = normal, >20 = elevated risk, >25 = avoid new entries")

    return {"symbol": sym, "explanation": lines, "score_summary": {
        "score": s.get("Score"), "setup": s.get("Setup"), "horizon": s.get("Horizon"),
        "regime": regime, "total_factors": len(lines)
    }}

# ── SELF-CALIBRATION CYCLE ───────────────────────────────────────
@app.post("/api/calibration/snapshot")
async def calibration_snapshot():
    """
    Log ALL current screener rows to calibration table for forward-return tracking.
    Call this after each extraction. After 5 trading days, call /update_returns.
    """
    if not STATE["raw_data_cache"]:
        return {"error": "no data in cache"}

    mkt = STATE.get("mkt") or {}
    regime = mkt.get("regime", "BULL")
    nifty_r5 = mkt.get("nifty_r5")
    nifty_r20 = mkt.get("nifty_r20")

    conn = get_db()
    logged = 0
    for sym, df_raw in STATE["raw_data_cache"].items():
        try:
            live = STATE["live_quotes_cache"].get(
                normalize_key(STATE["targets"].get(sym, "")), {})
            result = score_stock_dual(sym, df_raw, live, nifty_r5, nifty_r20)
            if result is None:
                continue
            conn.execute(
                "INSERT INTO calibration(ts,symbol,score,forward_ret,horizon,setup,regime) VALUES(?,?,?,?,?,?,?)",
                (_dt.now().isoformat(), sym, result.get("Score"),
                 None, result.get("Horizon",""), result.get("Setup",""), regime)
            )
            logged += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return {"logged": logged, "note": "Call /api/db/calibration/update_returns after 5 trading days"}

# ── EXPORT TO CSV / EXCEL ────────────────────────────────────────
from fastapi.responses import StreamingResponse
import io as _io

@app.get("/api/export/screener")
async def export_screener_csv():
    """Download current screener as CSV."""
    if not STATE["raw_data_cache"]:
        raise HTTPException(404, "No data — run extraction first")
    mkt = STATE.get("mkt") or {}
    rows = []
    for sym, df_raw in STATE["raw_data_cache"].items():
        cached = STATE["score_cache"].get(sym, {}).get("result")
        if cached:
            live = STATE["live_quotes_cache"].get(normalize_key(STATE["targets"].get(sym,"")), {})
            ltp = live.get("ltp") or float(df_raw["close"].iloc[-1])
            rows.append({"Ticker": sym, "LTP": round(ltp, 2), **cached})
    if not rows:
        raise HTTPException(404, "No scored rows")
    df_out = pd.DataFrame(rows)
    df_out = df_out.replace({float("nan"): "", float("inf"): "", float("-inf"): ""})
    buf = _io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=monarch_screener_{_dt.now().strftime('%Y%m%d_%H%M')}.csv"}
    )

@app.get("/api/export/snapshot/{ts}")
async def export_snapshot_csv(ts: str):
    """Download a historical snapshot as CSV."""
    conn = get_db()
    rows = conn.execute("SELECT row_json FROM snapshots WHERE ts=?", (ts,)).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(404, "Snapshot not found")
    data = [_json.loads(r["row_json"]) for r in rows]
    df_out = pd.DataFrame(data)
    buf = _io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    ts_safe = ts.replace(":", "-").replace(".", "-")[:19]
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=snapshot_{ts_safe}.csv"}
    )

# ── AUTO-SAVE SNAPSHOT ON EXTRACTION COMPLETE ────────────────────
# Patch run_extraction to save snapshot when done



# ══════════════════════════════════════════════════════════════════
# ML PREDICTION ENGINE  (scikit-learn RandomForest, trained on cache)
# ══════════════════════════════════════════════════════════════════
_ml_models: dict = {}   # {symbol: {"model":..., "trained_at":..., "accuracy":...}}
_ml_feature_cols = [
    "rsi","atr_pct","vol_ratio","ma_struct","bb_squeeze",
    "vol_dryup","clv","vcp","rs","breakout_prob","spread_comp",
    "up_vol_skew","cpr","signal_persist","proximity","vol_cont"
]

def _build_features(df: "pd.DataFrame", rsi_period: int = 7) -> "pd.DataFrame":
    """Extract ML feature matrix from OHLCV dataframe."""
    import numpy as np
    d = df.copy()
    c = d["close"]; h = d["high"]; lo = d["low"]; v = d.get("volume", pd.Series([1]*len(d)))

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    d["rsi"] = (100 - 100/(1 + gain/loss.replace(0, float("nan")))).fillna(50)

    # ATR%
    tr = pd.concat([(h-lo),(h-c.shift(1)).abs(),(lo-c.shift(1)).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=14,adjust=False).mean()
    d["atr_pct"] = (atr / c.replace(0,float("nan")) * 100).fillna(1)

    # Volume ratio
    vol_ma = v.rolling(20).mean()
    d["vol_ratio"] = (v / vol_ma.replace(0,float("nan"))).fillna(1).clip(0,10)

    # MA structure: EMA9/EMA50 ratio
    e9  = c.ewm(span=9,  adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    d["ma_struct"] = ((e9/e50.replace(0,float("nan")))-1).fillna(0).clip(-0.2,0.2)

    # BB squeeze
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_width = (bb_std * 4 / bb_mid.replace(0,float("nan"))).fillna(0.05)
    d["bb_squeeze"] = (1 - bb_width / bb_width.rolling(50).mean().replace(0,float("nan"))).fillna(0).clip(-1,1)

    # Volume dry-up
    d["vol_dryup"] = (1 - d["vol_ratio"].clip(0,2)/2).clip(0,1)

    # CLV (close location value)
    d["clv"] = ((c - lo - (h - c)) / (h - lo).replace(0,float("nan"))).fillna(0)

    # VCP proxy: range contraction
    d["vcp"] = (1 - (h - lo).rolling(5).mean() / (h - lo).rolling(20).mean().replace(0,float("nan"))).fillna(0).clip(0,1)

    # RS (5d return)
    d["rs"] = c.pct_change(5).fillna(0).clip(-0.3,0.3)

    # Breakout probability proxy: distance from 20-day high
    d["breakout_prob"] = ((h.rolling(20).max() - c) / c.replace(0,float("nan"))).fillna(0.05).clip(0,0.3)

    # Spread compression
    spread = (h - lo) / c.replace(0,float("nan"))
    d["spread_comp"] = (1 - spread / spread.rolling(20).mean().replace(0,float("nan"))).fillna(0).clip(-1,1)

    # Up-vol skew
    up_vol   = v.where(c >= c.shift(1), 0).rolling(5).sum()
    down_vol = v.where(c < c.shift(1), 0).rolling(5).sum()
    d["up_vol_skew"] = (up_vol / (up_vol + down_vol + 1e-9)).fillna(0.5) - 0.5

    # CPR (close position rank in 5-day range)
    d["cpr"] = ((c - lo.rolling(5).min()) / (h.rolling(5).max() - lo.rolling(5).min() + 1e-9)).fillna(0.5) - 0.5

    # Signal persistence: fraction of last 3 days above EMA9
    d["signal_persist"] = (c > e9).rolling(3).mean().fillna(0.5)

    # Proximity to 20-day high
    d["proximity"] = (1 - (h.rolling(20).max() - c) / (h.rolling(20).max() - lo.rolling(20).min() + 1e-9)).fillna(0.5)

    # Vol continuity (ATR5/ATR20)
    atr5  = tr.ewm(span=5,  adjust=False).mean()
    atr20 = tr.ewm(span=20, adjust=False).mean()
    d["vol_cont"] = (atr5 / atr20.replace(0,float("nan"))).fillna(1).clip(0,3)

    return d

def _train_ml(sym: str, df: "pd.DataFrame", horizon: int = 5) -> dict:
    """Train a RandomForest on the stock's own OHLCV history."""
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        import numpy as np

        feat_df = _build_features(df, STATE.get("rsi_period", 7))
        feat_df["fwd_ret"] = feat_df["close"].pct_change(horizon).shift(-horizon)
        feat_df["target"]  = (feat_df["fwd_ret"] > 0.005).astype(int)  # >0.5% = buy
        feat_df = feat_df.dropna(subset=_ml_feature_cols + ["target"])

        if len(feat_df) < 60:
            return {"error": "insufficient history (need 60+ bars)"}

        X = feat_df[_ml_feature_cols].values
        y = feat_df["target"].values

        # Time-series split for honest accuracy
        tscv = TimeSeriesSplit(n_splits=3)
        accs = []
        for tr_idx, val_idx in tscv.split(X):
            clf = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42, n_jobs=1)
            clf.fit(X[tr_idx], y[tr_idx])
            accs.append(float((clf.predict(X[val_idx]) == y[val_idx]).mean()))

        # Final model on all data
        clf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)
        clf_final.fit(X, y)

        importances = dict(zip(_ml_feature_cols, clf_final.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:6]

        # Predict on latest bar
        latest = feat_df[_ml_feature_cols].iloc[-1:].values
        prob_up   = float(clf_final.predict_proba(latest)[0][1])
        direction = "BUY" if prob_up > 0.55 else "SELL" if prob_up < 0.45 else "NEUTRAL"

        # Price targets using ATR
        ltp = float(df["close"].iloc[-1])
        atr = float(_build_features(df)["atr_pct"].iloc[-1] / 100 * ltp)
        mul = 2.0 if prob_up > 0.65 else 1.5 if prob_up > 0.55 else 1.0
        target_price = round(ltp + mul * atr, 2)
        stop_price   = round(ltp - 1.0 * atr, 2)

        result = {
            "symbol":        sym,
            "direction":     direction,
            "prob_up":       round(prob_up * 100, 1),
            "confidence":    "HIGH" if abs(prob_up-0.5)>0.15 else "MEDIUM" if abs(prob_up-0.5)>0.08 else "LOW",
            "cv_accuracy":   round(float(np.mean(accs))*100, 1),
            "horizon_days":  horizon,
            "ltp":           round(ltp, 2),
            "ml_target":     target_price,
            "ml_stop":       stop_price,
            "top_features":  top_features,
            "trained_bars":  len(feat_df),
            "model":         "RandomForest(100 trees, depth=5)",
            "validation":    "3-fold TimeSeriesSplit",
        }
        _ml_models[sym] = {"result": result}
        return result

    except ImportError:
        return {"error": "scikit-learn not installed — run: pip install scikit-learn"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/ml/{symbol}")
async def ml_predict(symbol: str, horizon: int = 5):
    """Train ML model on stock history and return prediction."""
    sym = symbol.upper()
    df = STATE["raw_data_cache"].get(sym)
    if df is None:
        raise HTTPException(404, "Not in cache — run extraction first")
    # Use cached if recent (same session)
    if sym in _ml_models:
        return _ml_models[sym]["result"]
    result = _train_ml(sym, df, horizon)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result

@app.delete("/api/ml/{symbol}/cache")
async def clear_ml_cache(symbol: str):
    _ml_models.pop(symbol.upper(), None)
    return {"status": "cleared"}

# ══════════════════════════════════════════════════════════════════
# FUNDAMENTALS  (Yahoo Finance via yfinance)
# ══════════════════════════════════════════════════════════════════
_fund_cache: dict = {}  # {symbol: {data, ts}}

@app.get("/api/fundamentals/{symbol}")
async def get_fundamentals(symbol: str):
    """Fetch key fundamental ratios from Yahoo Finance."""
    sym = symbol.upper()
    import time as _time

    # Cache for 4 hours
    cached = _fund_cache.get(sym)
    if cached and (_time.time() - cached["ts"]) < 14400:
        return cached["data"]

    # Map NSE symbol to Yahoo ticker
    yf_sym = sym + ".NS"
    try:
        import yfinance as yf
        tk = yf.Ticker(yf_sym)
        info = tk.info or {}

        def safe(key, fmt=None):
            v = info.get(key)
            if v is None or v == "N/A": return None
            try:
                v = float(v)
                return round(v, 2) if fmt == "float" else v
            except: return str(v)

        data = {
            "symbol":         sym,
            "yf_ticker":      yf_sym,
            "name":           info.get("longName") or info.get("shortName") or sym,
            "sector":         info.get("sector","—"),
            "industry":       info.get("industry","—"),
            "market_cap":     safe("marketCap"),
            "market_cap_cr":  round(safe("marketCap")/1e7, 0) if safe("marketCap") else None,
            "pe_ratio":       safe("trailingPE","float"),
            "forward_pe":     safe("forwardPE","float"),
            "pb_ratio":       safe("priceToBook","float"),
            "ps_ratio":       safe("priceToSalesTrailing12Months","float"),
            "ev_ebitda":      safe("enterpriseToEbitda","float"),
            "roe":            round(safe("returnOnEquity")*100, 1) if safe("returnOnEquity") else None,
            "roce":           None,  # not in yfinance
            "debt_equity":    safe("debtToEquity","float"),
            "current_ratio":  safe("currentRatio","float"),
            "revenue_growth": round(safe("revenueGrowth")*100,1) if safe("revenueGrowth") else None,
            "earnings_growth":round(safe("earningsGrowth")*100,1) if safe("earningsGrowth") else None,
            "profit_margin":  round(safe("profitMargins")*100,1) if safe("profitMargins") else None,
            "gross_margin":   round(safe("grossMargins")*100,1) if safe("grossMargins") else None,
            "dividend_yield": round(safe("dividendYield")*100,2) if safe("dividendYield") else None,
            "52w_high":       safe("fiftyTwoWeekHigh","float"),
            "52w_low":        safe("fiftyTwoWeekLow","float"),
            "avg_vol_30d":    safe("averageVolume"),
            "float_shares":   safe("floatShares"),
            "beta":           safe("beta","float"),
            "eps_ttm":        safe("trailingEps","float"),
            "book_value":     safe("bookValue","float"),
            "description":    (info.get("longBusinessSummary") or "")[:400],
            "analyst_target": safe("targetMeanPrice","float"),
            "analyst_low":    safe("targetLowPrice","float"),
            "analyst_high":   safe("targetHighPrice","float"),
            "analyst_count":  safe("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey","—"),
        }
        _fund_cache[sym] = {"data": data, "ts": _time.time()}
        return data
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")


if __name__ == "__main__":
    import uvicorn
    try:
        import config as _c
        _port = _c.PORT
        _host = _c.HOST
    except Exception:
        _port = 8000
        _host = "0.0.0.0"
    uvicorn.run(app, host=_host, port=_port)