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
import gzip, json as _json, math, os, pathlib, sqlite3, threading, time, urllib.parse
import io as _io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from datetime import datetime as _dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import argrelmin as _argrelmin
from scipy.stats import rankdata

# ── Sector DB (full 2500-stock NSE mapping) ───────────────────────
from sector_db import (
    get_sector_db,
    _load_cache as _load_sector_db_cache,
    get_all_mappings   as _sector_get_all,
    get_coverage_stats as _sector_get_stats,
    manual_add         as _sector_manual_add,
    reload_cache       as _sector_reload_cache,
    build              as _sector_build,
)


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
    # Rebalanced for pre-move detection: VCP/Coil/Proximity are predictive,
    # RS/Momentum are lagging confirming signals.
    W_RS:                   float = 0.12   # reduced — lagging, measures past strength
    W_RS_SECT:              float = 0.05   # reduced
    W_MOMENTUM:             float = 0.08   # reduced — EMA acceleration is lagging
    W_VOLUME:               float = 0.10   # reduced — confirmed vol = move already started
    W_COIL:                 float = 0.20   # DOUBLED — core pre-move compression signal
    W_MA:                   float = 0.10   # unchanged — structural filter
    W_PROXIMITY:            float = 0.12   # increased — WHERE price is vs trigger matters most
    W_VCP:                  float = 0.13   # increased — best pattern predictor of imminent move
    W_DARVAS:               float = 0.06   # increased — tight box = coiling energy
    W_MICROSTRUCTURE:       float = 0.04   # unchanged: CLV + BB + VC + spread

    # ── Signal coverage floor (fraction of valid signals required) ──
    COVERAGE_FLOOR:         float = 0.40

    # ── Interaction term amplifier ────────────────────────────────
    # When RS + volume + proximity all fire together the score gets a
    # multiplicative lift.  Cap prevents runaway scores.
    INTERACTION_FLOOR:      float = 0.50   # was 0.60 — lower so more setups get the boost
    INTERACTION_BOOST_MAX:  float = 0.14   # was 0.12 — slightly bigger boost for aligned signals

    # ── Calibration-to-weight adaptation ─────────────────────────
    # How aggressively learned win-rates nudge static weights.
    # 0 = disabled, 1 = full replacement.
    CALIB_ADAPT_ALPHA:      float = 0.25   # blend factor

    # ── Universe breakout saturation guard ───────────────────────
    # If this fraction of the universe is already tagged Breakout,
    # marginal breakout probability gets discounted.
    BO_SATURATION_FLOOR:    float = 0.65   # was 0.50 — only discount when 65%+ are breakouts
    BO_SATURATION_DISCOUNT: float = 0.15   # was 0.30 — gentler discount

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
    ALREADY_BO_CAP:         float = 6.0    # reduced: don't kill stocks that are actually breaking out
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
    CANDLE_HAMMER_CLOSE_RATIO: float = 0.60   # close must be in top 60% of bar range
    CANDLE_BULL_DOJI_WICK_MULT:float = 1.5    # lower wick must be > this × upper wick
    CANDLE_MSTAR_BODY_RATIO:   float = 0.50   # prev bar body / range must exceed this
    CANDLE_GAP_BODY_MULT:      float = 0.50   # gap must exceed prev_body × this

    # ── Reversal detection ──
    REVERSAL_MIN_WASHOUT_ATR:  float = 1.5    # washout_depth >= this ATR to qualify
    REVERSAL_TAIL_MIN_POS:     float = 0.40   # t1_close_pos must be >= this to qualify

    # ── Intraday breakout volume minimum ──
    INTRADAY_BO_VOL_MIN:       float = 0.15   # vol_confirm session guard (alias) — lowered to match VOL_CONFIRM_MIN_FRAC

    # ── Sweep ──
    SWEEP_WICK_MIN_ATR:        float = 0.50   # lower wick must be >= this × ATR

    # ── Calibration breadth batch / extraction prefilter ──
    CALIB_BREADTH_BATCH:       int   = 20     # batch size for breadth_hist computation
    EXTRACT_MIN_BARS_PREFILTER:int   = 30     # min bars before adding to raw_data_cache

    # ── Fetch / extraction ──
    FETCH_WORKERS:          int   = 5      # reduced: Upstox historical API silently throttles >5 concurrent
    FETCH_DELAY:            float = 0.12   # increased: breathing room between requests per worker
    FETCH_RETRIES:          int   = 5      # extra retry since empty-candle is now retried too
    FETCH_BACKOFF:          float = 1.0    # longer backoff on retries
    LIVE_QUOTE_CHUNK:       int   = 50
    LIVE_QUOTE_DELAY:       float = 0.12
    HISTORY_DAYS:           int   = 600
    CHART_BARS:             int   = 120
    MIN_BARS:               int   = 60

    # ── Cache TTLs (seconds) ──
    MASTER_TTL:             int   = 3_600
    NIFTY50_TTL:            int   = 14_400
    MKT_CONTEXT_TTL:        int   = 600    # was 900 — refresh market context every 10 min
    LIVE_REFRESH_SEC:       int   = 30     # was 60 — refresh live quotes every 30s

    # ── Calibration / DB ──
    FORWARD_RETURN_DAYS:    int   = 5
    SNAPSHOT_KEEP_DAYS:     int   = 30

    # ── Registry history ──
    REG_HISTORY:            int   = 200

    # ── RS combination blend weights (cs_rs / slope / abs_rs) ────────────
    # Sum must equal 1.0.  cs_rs is cross-sectional rank (most reliable),
    # rs_slope is momentum direction, abs_rs is vol-normalised alpha vs Nifty.
    RS_BLEND_CS:            float = 0.40
    RS_BLEND_SLOPE:         float = 0.35
    RS_BLEND_ABS:           float = 0.25

    # ── Sector RS additive adjustments ────────────────────────────────────
    # How much sector-rank and sector-acceleration can nudge rs_sect.
    RS_SECT_RANK_BONUS:     float = 0.12   # max lift from being in a top sector
    RS_SECT_ACCEL_BONUS:    float = 0.08   # max lift from sector momentum improving

    # ── Win-rate prior formula coefficients ──────────────────────────────
    # wr_prior = WR_BASE + WR_RS_COEF * rs + WR_STAB_COEF * stab
    WR_BASE:                float = 0.40
    WR_RS_COEF:             float = 0.20
    WR_STAB_COEF:           float = 0.10

    # ── Regime penalty factors (applied to coiling quality nf) ───────────
    REGIME_BEAR_FACTOR:     float = 5.0    # was 8.0 — softer: strong coils still surface in bear
    REGIME_CHOP_FACTOR:     float = 2.5    # was 4.0 — softer: chop doesn't kill pre-move setups

    # ── VIX falling extra penalty scale ──────────────────────────────────
    VIX_FALL_EXTRA:         float = 1.5

    # ── ATR expansion onset: max bars since compression to count ─────────
    # Derived from ATR_FAST window — expansion is only "onset" for the
    # first half of the fast ATR window after compression ends.
    ATR_EXP_ONSET_BARS:     int   = 3    # = ATR_FAST // 2, set at startup

    # ── Reversal detection volume threshold (P-rank of vol ratio) ────────
    # Stock must have vol in top this fraction of its own history.
    REVERSAL_VOL_PRANK:     float = 0.70   # top 30%% of historical vol ratios

    # ── Breakout entry buffer (fraction of ATR above base high) ──────────
    BO_ENTRY_BUFFER_ATR:    float = 0.08

    # ── Pullback target: fraction of 20d high (1.0 = exact high) ─────────
    # 0.995 leaves a small buffer for bid-ask spread / fill slippage.
    PB_TARGET_HIGH_FRAC:    float = 0.995

    # ── Coil bp_bonus sigmoid cap ─────────────────────────────────────────
    COIL_BP_BONUS_CAP:      float = 0.15

    # ── Proximity: over-shoot penalty asymmetry ───────────────────────────
    # When price has overshot ideal entry, how much harder to penalise.
    # 1.0 = symmetric; >1.0 = penalise overshoot more than undershoot.
    PROX_OVERSHOOT_MULT:    float = 1.3
    PROX_BO_OVERSHOOT_MULT: float = 2.5   # steeper decay when price is above base_hi for Breakout/Coiling

    # ── Darvas ideal box duration (bars) ─────────────────────────────────
    # time_score = bars_in_box / (DARVAS_IDEAL_BARS * 2).  A box that has
    # been consolidating for DARVAS_IDEAL_BARS scores 0.5; 2× scores 1.0.
    DARVAS_IDEAL_BARS:      int   = 15

    # ── Intraday velocity: minimum session fraction before using live vol ─
    # Below this fraction of the session, live vol is too noisy to trust.
    VOL_VELOCITY_MIN_FRAC:  float = 0.20   # ~first 75 min of 375-min session

    # ── Stability fallback (when no stab_pct percentile available) ───────
    STAB_ADJ_FALLBACK_HI:   float = 0.40   # award this when stability is high
    STAB_FALLBACK_THRESH:   float = 0.55   # stability >= this → "high"

    # ── Institutional feature sigmoid fallback parameters ─────────────────
    # Used when < 20 bars of rolling inst ratio history are available.
    INST_FALLBACK_CENTRE:   float = 1.15   # typical 5d/20d vol ratio
    INST_FALLBACK_SCALE:    float = 0.30   # IQR-equivalent spread

    # ── VCP detection thresholds ───────────────────────────────────────────
    VCP_DETECT_SCORE_THRESH:       float = 0.45  # was 0.55 — detect VCP earlier, before it's obvious
    VCP_DETECT_CONTRACTION_THRESH: float = 0.32  # was 0.40 — earlier detection

    # ── Candle confirmation gate ──────────────────────────────────────────
    # Candle patterns on an incomplete live bar are unreliable.  Below this
    # fraction of the session the cdl_sc is zeroed; patterns are still detected
    # and reported but do not contribute to score.
    CANDLE_CONFIRM_FRAC:    float = 0.95   # ~last 19 min of 375-min session

    # ── Volume confirmation session guard ─────────────────────────────────
    # vol_confirm and Breakout classification are suppressed before this
    # fraction of the session has elapsed — pace-adjusting early-session volume
    # reduces but does not eliminate noise at open.
    VOL_CONFIRM_MIN_FRAC:   float = 0.15   # was 0.30 — P95 cap handles noise; guard was suppressing genuine early breakouts

    # ── Score gate for entry/target/stop output ───────────────────────────
    # Below this score the Entry/Target/Stop fields are set to None and the
    # EntryNote explains the stock is watchlist-only.  Prevents false precision
    # on weak setups and keeps the screener output clean.
    MIN_SCORE_FOR_LEVELS:   float = 38.0  # was 45 — show levels for more candidates

    # ── ATR expansion onset reclassification ──────────────────────────────
    # atr_exp_feature fires at the *start* of ATR expansion (move already
    # beginning).  When combined with vol_confirm it acts as a confirmation
    # bonus rather than a pre-move coil signal.  This multiplier scales it
    # down when vol_confirm is absent.
    ATR_EXP_NOCONFIRM_SCALE: float = 0.25  # discount factor without vol confirm

    # ── Intraday market circuit breaker ──────────────────────────────────
    # If Nifty's intraday change is below this threshold, all Breakout signals
    # are downgraded to Pullback and EntryNote warns of market-wide weakness.
    NIFTY_INTRADAY_KILL:    float = -0.020  # −2.0 %

    # ── Penalty magnitude constants (previously hardcoded in function bodies) ─
    MARKET_CB_PENALTY:      float = 6.0    # flat penalty when circuit breaker fires
    BREADTH_FALLBACK_PENALTY: float = 8.0  # breadth penalty when Nifty below 20DMA
    VIX_TANH_SCALE:         float = 6.0    # tanh multiplier for VIX z-score penalty

    # ── Reversal sub-score weights (must sum to 1.0) ──────────────────────
    REV_W_RSI:   float = round(1/3,  10)   # 1/3
    REV_W_COIL:  float = round(1/4,  10)   # 1/4
    REV_W_PROX:  float = round(1/6,  10)   # 1/6
    REV_W_SPR:   float = round(1/12, 10)   # 1/12
    REV_W_VOL:   float = round(1/24, 10)   # 1/24
    REV_W_WASH:  float = round(1/12, 10)   # 1/12
    REV_W_TAIL:  float = round(1/24, 10)   # 1/24

    # ── Darvas sub-score weights ───────────────────────────────────────────
    DARVAS_W_TIGHT: float = 0.40
    DARVAS_W_POS:   float = 0.35
    DARVAS_W_TIME:  float = 0.25
    # Fallback weights (when box_high is NaN): raw_score vs position_in_box
    DARVAS_FB_W_RAW: float = 0.65
    DARVAS_FB_W_POS: float = 0.35

    # ── Weekly trend alignment ────────────────────────────────────────────
    # Breakout signals are discounted when the weekly EMA9 < weekly EMA20
    # (daily data resampled to weekly).  This flag enables the check;
    # the discount is applied as a penalty in compute_penalties.
    WEEKLY_TREND_CHECK:     bool  = True
    WEEKLY_TREND_PENALTY:   float = 6.0    # was 12 — softer counter-trend penalty

    # ── Pullback sub-type thresholds ─────────────────────────────────────
    # Derived from EMA distance in ATR units (stock's own distribution).
    # PB_EMA_ATR_BAND: price within this ATR of EMA20/EMA9 → "PB-EMA" (at support)
    # PB_DRY_VOL_FRAC: vol below this fraction of 40th-pct ratio → "PB-Dry" (vol drying)
    # These are ATR/vol multiples — not price levels, so they auto-scale per stock.
    # "PB-Deep" = real pullback but not near EMA = deeper correction, still above EMA50
    # "Base"    = no real pullback detected = just range-bound / no signal
    # The band is already derived from the stock's own P20-P80 dist in compute_signals;
    # we just rename the branches here — no new threshold needed.

    # ── VolClimax detection multipliers ──────────────────────────────────
    # Vol climax = vol_ratio exceeds P-rank threshold AND RSI exceeds its
    # own P-rank threshold AND price extended beyond ext_p90.
    # All three are derived from each stock's own distribution — no fixed levels.
    VOL_CLIMAX_PRANK:       float = 0.90   # vol must be in top 10% of its own history
    VOL_CLIMAX_RSI_PRANK:   float = 0.70   # RSI must be in top 30% of its own history
    VOL_CLIMAX_EXT_PRANK:   float = 0.90   # price extension must be in top 10% of its own history

    # ── Resistance proximity gate for target price ────────────────────────
    # Target is capped below a prior high when tgt lands within this
    # fraction of the resistance level.  Derived from the stock's own
    # ATR as a multiple (1 ATR tolerance), not a fixed % like 2%.
    TGT_RESIST_ATR_BUFFER:  float = 1.0    # if tgt is within 1 ATR of a prior high → cap it

    # ── Minimum calibration rows before Kelly is shown ───────────────────
    # Below this count the formula win-rate is made-up; suppress the output.
    KELLY_MIN_CALIB_ROWS:   int   = 10

    # ── Compression signal field label ────────────────────────────────────
    # Rename "BreakoutProb" to "CompressionScore" in the return dict so
    # it's not misread as a probability.  Both backend and frontend use this.
    # (Actual rename is in aggregate_score return dict.)

    # ── News freshness window ─────────────────────────────────────────────
    # Articles older than this many days are filtered out regardless of match.
    NEWS_MAX_AGE_DAYS:      int   = 7

    # ── Pullback entry note: max ATR distance above EMA20 before note warns ─
    # If LTP > EMA20 + this multiple × ATR, the entry note warns the user
    # they are not at support yet.  Derived from proximity lambda logic.
    PB_NOTE_EMA_ATR_WARN:   float = 1.0    # 1 ATR above EMA20 = warn

    # ── Candle pattern point values ────────────────────────────────────────
    # These sum is capped at 10 in detect_candle_patterns.
    # Ordering follows strength of signal (Engulfing > Hammer ≈ MorningStar > …)
    CANDLE_PTS_ENGULFING:   float = 3.0
    CANDLE_PTS_HAMMER:      float = 2.5
    CANDLE_PTS_MORNING_STAR:float = 2.5
    CANDLE_PTS_OUTSIDE_BAR: float = 2.0
    CANDLE_PTS_STRONG_GREEN:float = 2.0
    CANDLE_PTS_GAP_CONTINUE:float = 2.0
    CANDLE_PTS_INSIDE_BAR:  float = 1.5
    CANDLE_PTS_BULL_DOJI:   float = 1.0

    # ── Breakout volume threshold percentile ───────────────────────────────
    # Day volume must exceed this percentile of last-60-bar distribution
    # for a bar to count as a confirmed volume breakout.
    BO_VOL_PERCENTILE:      float = 0.85

    # ── RSI fallback values (used when < MIN_BARS_RSI bars available) ─────
    RSI_OB_FALLBACK:        float = 80.0   # overbought threshold fallback
    RSI_OS_FALLBACK:        float = 42.0   # oversold threshold fallback (P35 proxy)
    RSI_MID_FALLBACK:       float = 55.0   # neutral midpoint fallback (P60 proxy)
    RSI_LOW_FALLBACK:       float = 25.0   # deep oversold fallback (P10 proxy)
    MIN_BARS_RSI:           int   = 20     # min bars required for RSI quantile computation

    # ── Percentile lookback for per-stock quantile signals ─────────────────
    # Most per-stock percentile windows use PERCENTILE_WINDOW_SHORT (60).
    # A handful of structural patterns (VCP, Darvas) use their own fixed
    # windows — those are documented at the call site.
    PERCENTILE_WINDOW_PERSHORT: int = 60   # alias used for tail(60) sites

    # ── Coiling score composite ───────────────────────────────────────────
    # CoilingScore = weighted average of pure pre-move compression signals.
    # Computed independently of vol_confirm so it scores BEFORE the move starts.
    # Rebalanced: CLV (institutional buying INTO compression) and VCP (Minervini
    # pattern) are the two strongest pre-move predictors. BB and VDU confirm
    # compression is happening but alone are less predictive.
    COIL_W_BB:          float = 0.18   # was 0.25 — BB squeeze confirms compression
    COIL_W_VDU:         float = 0.15   # was 0.22 — vol dryup confirms compression
    COIL_W_CLV:         float = 0.27   # was 0.15 — INCREASED: smart money buying into base
    COIL_W_VCP:         float = 0.27   # was 0.20 — INCREASED: Minervini pattern = strongest predictor
    COIL_W_SC:          float = 0.08   # was 0.10 — spread compression + rising close
    COIL_W_VC:          float = 0.05   # was 0.08 — ATR contraction ratio

    # Minimum coiling score (0-100) to classify a stock as "Coiling" setup
    COIL_SETUP_THRESH:  float = 52.0   # was 58 — capture more candidates before they move

    # ── Signal persistence (multi-day coil streak) ────────────────────────
    # How many consecutive days BB squeeze AND vol dryup must both be in the
    # top COIL_PERSIST_PRANK fraction of the universe to earn a streak bonus.
    COIL_PERSIST_PRANK: float = 0.60   # was 0.65 — slightly more lenient to catch more candidates
    COIL_PERSIST_MIN_DAYS: int = 2     # was 3 — start rewarding after 2 days, not 3
    COIL_PERSIST_BONUS_CAP: float = 8.0  # was 6.0 — bigger reward for multi-day compression

    # ── Adaptive volume weighting ─────────────────────────────────────────
    # When vol_confirm is False (pre-move), volume weight is redistributed to
    # coil + VCP instead of penalising the stock for lacking current volume.
    # vol_weight_confirmed   = W_VOLUME (unchanged when breakout confirmed)
    # vol_weight_unconfirmed = W_VOLUME * VOL_WEIGHT_UNCONFIRMED_SCALE
    # The freed weight flows into coil and VCP proportionally.
    VOL_WEIGHT_UNCONFIRMED_SCALE: float = 0.20   # was 0.40 — pre-move vol is noise, not signal
    COIL_WEIGHT_BOOST:            float = 0.09   # freed weight → coil (bigger pre-move boost)
    VCP_WEIGHT_BOOST:             float = 0.09   # freed weight → VCP  (bigger pre-move boost)

    # ── compute_cs_ranks minimum bar guards ────────────────────────────────
    # Derived from the indicator windows each cross-sectional signal needs:
    #   bbr  (BB squeeze) : BB_WINDOW + 10  → a settled BB + room to assess
    #   vdr  (vol dryup)  : ATR_FAST + ATR_SLOW → enough rolling vol history
    #   clvr (CLV accum)  : BB_WINDOW + 5   → 20-bar rolling sum + headroom
    #   vcpr (VCP)        : MIN_BARS         → hard floor in detect_vcp
    #   breadth           : BB_WINDOW        → EMA20 must be settled
    # These are computed as properties so they update if windows change.
    @property
    def CS_MIN_BARS_BB(self)  -> int: return self.BB_WINDOW + 10
    @property
    def CS_MIN_BARS_VDR(self) -> int: return self.ATR_FAST + self.ATR_SLOW
    @property
    def CS_MIN_BARS_CLV(self) -> int: return self.BB_WINDOW + 5
    @property
    def CS_MIN_BARS_VCP(self) -> int: return self.MIN_BARS
    @property
    def CS_MIN_BARS_BRD(self) -> int: return self.BB_WINDOW


SCORE_CFG = ScoreConfig()

# Startup assertion: reversal weights must sum to 1.0 (guards against future drift).
_rev_w_sum = (SCORE_CFG.REV_W_RSI + SCORE_CFG.REV_W_COIL + SCORE_CFG.REV_W_PROX +
              SCORE_CFG.REV_W_SPR + SCORE_CFG.REV_W_VOL + SCORE_CFG.REV_W_WASH +
              SCORE_CFG.REV_W_TAIL)
assert abs(_rev_w_sum - 1.0) < 1e-9, (
    f"Reversal weights must sum to 1.0; got {_rev_w_sum:.10f}. "
    "Update ScoreConfig.REV_W_* fields to correct fractions."
)

# Startup assertion: coiling score weights must sum to 1.0.
# CoilingScore has the largest single weight (W_COIL=0.20) in the main score;
# a miscalibrated COIL_W_* set silently inflates/deflates the entire pipeline.
_coil_w_sum = (SCORE_CFG.COIL_W_BB + SCORE_CFG.COIL_W_VDU + SCORE_CFG.COIL_W_CLV +
               SCORE_CFG.COIL_W_VCP + SCORE_CFG.COIL_W_SC + SCORE_CFG.COIL_W_VC)
assert abs(_coil_w_sum - 1.0) < 1e-6, (
    f"COIL_W_* weights must sum to 1.0; got {_coil_w_sum:.8f}. "
    "Update ScoreConfig.COIL_W_* fields."
)

# Startup assertion: main score component weights must sum to 1.0.
_main_w_sum = (SCORE_CFG.W_RS + SCORE_CFG.W_RS_SECT + SCORE_CFG.W_MOMENTUM +
               SCORE_CFG.W_VOLUME + SCORE_CFG.W_COIL + SCORE_CFG.W_MA +
               SCORE_CFG.W_PROXIMITY + SCORE_CFG.W_VCP + SCORE_CFG.W_DARVAS +
               SCORE_CFG.W_MICROSTRUCTURE)
assert abs(_main_w_sum - 1.0) < 1e-6, (
    f"Main score W_* weights must sum to 1.0; got {_main_w_sum:.8f}. "
    "Update ScoreConfig.W_* fields."
)

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

# ── NSE index → yfinance ticker (for sector return computation) ──
SECTOR_TICKERS: Dict[str, str] = {
    "IT":          "^CNXIT",      "Bank":        "^NSEBANK",
    "Auto":        "^CNXAUTO",    "Pharma":      "^CNXPHARMA",
    "Metal":       "^CNXMETAL",   "Energy":      "^CNXENERGY",
    "Infra":       "^CNXINFRA",   "FMCG":        "^CNXFMCG",
    "Realty":      "^CNXREALTY",  "PSUBank":     "^CNXPSUBANK",
    "Chemicals":   "^CNXCHEMICALS","ConsumerDur": "^CNXCONSUMER",
    "Insurance":   "^CNXFINSERVICE","Telecom":   "^CNXTELECOM",
    "Retail":      "^CNXCONSUMER",             # nearest proxy
    "Logistics":   "^CNXINFRA",                # nearest proxy
}

# ── NSE equity-stockIndices index name → internal sector label ───
# These match what NSE's /api/equity-stockIndices?index=<name> returns.
_NSE_INDEX_TO_SECTOR: Dict[str, str] = {
    "NIFTY IT":                   "IT",
    "NIFTY BANK":                 "Bank",
    "NIFTY AUTO":                 "Auto",
    "NIFTY PHARMA":               "Pharma",
    "NIFTY METAL":                "Metal",
    "NIFTY ENERGY":               "Energy",
    "NIFTY INFRASTRUCTURE":       "Infra",
    "NIFTY FMCG":                 "FMCG",
    "NIFTY REALTY":               "Realty",
    "NIFTY PSU BANK":             "PSUBank",
    "NIFTY CHEMICALS":            "Chemicals",
    "NIFTY CONSUMER DURABLES":    "ConsumerDur",
    "NIFTY FINANCIAL SERVICES":   "Insurance",
    "NIFTY INDIA DIGITAL":        "Telecom",
    "NIFTY INDIA CONSUMPTION":    "Retail",
    "NIFTY MIDSMALL HEALTHCARE":  "Pharma",
    "NIFTY MIDSMALL IT & TELECOM":"IT",
}

# ── Index priority: higher = more specific = wins on conflict ─────────────────
# NIFTY IT (→IT) and NIFTY INDIA DIGITAL (→Telecom) both contain TCS/INFY/WIPRO.
# NIFTY BANK (→Bank) and NIFTY FINANCIAL SERVICES (→Insurance) both contain
# HDFCBANK/ICICIBANK/KOTAKBANK/AXISBANK/SBIN.
# Without priority, the LAST index iterated wins — which is wrong.
# Fix: specific sector indices get priority=10; broad thematic indices get priority=1.
_NSE_INDEX_PRIORITY: Dict[str, int] = {
    "NIFTY IT":                   10,
    "NIFTY BANK":                 10,
    "NIFTY AUTO":                 10,
    "NIFTY PHARMA":               10,
    "NIFTY METAL":                10,
    "NIFTY ENERGY":               10,
    "NIFTY INFRASTRUCTURE":       10,
    "NIFTY FMCG":                 10,
    "NIFTY REALTY":               10,
    "NIFTY PSU BANK":             10,
    "NIFTY CHEMICALS":            10,
    "NIFTY CONSUMER DURABLES":    10,
    "NIFTY OIL AND GAS":          10,
    "NIFTY MIDSMALL HEALTHCARE":   5,
    "NIFTY MIDSMALL IT & TELECOM": 5,
    # Broad/thematic — overlaps heavily with specific indices above
    "NIFTY FINANCIAL SERVICES":    1,  # overlaps NIFTY BANK
    "NIFTY INDIA DIGITAL":         1,  # overlaps NIFTY IT
    "NIFTY INDIA CONSUMPTION":     1,  # overlaps FMCG/Retail
}

# ── Dynamic STOCK_SECTOR_MAP ──────────────────────────────────────
# Populated at startup (and periodically refreshed) by fetching the
# constituent list of every NSE sector index via the same session-based
# API pattern used by get_nifty50_live().  A stock may appear in multiple
# indices; the LAST sector in _NSE_INDEX_TO_SECTOR iteration wins, which
# is fine because sector return is used only for relative-strength context,
# not as a strict classification.
#
# The static fallback dict below is used:
#   (a) on first startup before the async fetch completes, and
#   (b) whenever the NSE API returns fewer than 5 stocks for an index
#       (i.e. the response is clearly bad / rate-limited).
_SECTOR_MAP_LOCK = threading.Lock()

_STATIC_SECTOR_FALLBACK: Dict[str, str] = {
    # IT
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","MASTEK":"IT","HEXAWARE":"IT",
    # Bank (private)
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank","RBLBANK":"Bank","YESBANK":"Bank",
    "CSBBANK":"Bank","DCBBANK":"Bank","KARURVYSYA":"Bank",
    # PSU Bank
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "UNIONBANK":"PSUBank","BANKINDIA":"PSUBank","MAHABANK":"PSUBank",
    "INDIANB":"PSUBank","UCOBANK":"PSUBank","CENTRALBK":"PSUBank",
    # Auto
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "APOLLOTYRE":"Auto","MRF":"Auto","CEATLTD":"Auto","EXIDEIND":"Auto",
    # Pharma
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "LUPIN":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","GLENMARK":"Pharma",
    "IPCALAB":"Pharma","NATCOPHARM":"Pharma","LAURUSLABS":"Pharma",
    "FORTIS":"Pharma","METROPOLIS":"Pharma","LALPATHLAB":"Pharma",
    # Metal
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal","JINDALSTEL":"Metal",
    "APLAPOLLO":"Metal","RATNAMANI":"Metal","NATIONALUM":"Metal","MOIL":"Metal",
    # Energy
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy","HPCL":"Energy",
    "PETRONET":"Energy","OIL":"Energy","HINDPETRO":"Energy","MGL":"Energy",
    "IGL":"Energy","TATAPOWER":"Energy","ADANIGREEN":"Energy","ADANIENT":"Energy",
    # Infra
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEMENT":"Infra","ACC":"Infra","SIEMENS":"Infra","ABB":"Infra",
    "BEL":"Infra","HAL":"Infra","BHEL":"Infra","CUMMINSIND":"Infra",
    "THERMAX":"Infra","KEC":"Infra","KALPATPOWR":"Infra","VOLTAS":"Infra",
    # FMCG
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "EMAMILTD":"FMCG","COLPAL":"FMCG","TATACONSUM":"FMCG","UBL":"FMCG",
    "RADICO":"FMCG","VBL":"FMCG",
    # Realty
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "GODREJPROP":"Realty","PRESTIGE":"Realty","BRIGADE":"Realty","SOBHA":"Realty",
    # Chemicals
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ALKYLAMINE":"Chemicals",
    "FINEORG":"Chemicals","VINATIORGA":"Chemicals","BALRAMCHIN":"Chemicals",
    # Insurance / FinServices
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIPRULI":"Insurance",
    "LICIHSGFIN":"Insurance","MUTHOOTFIN":"Insurance","CHOLAFIN":"Insurance",
    "ICICIGI":"Insurance","NIACL":"Insurance","GICRE":"Insurance",
    "HDFCAMC":"Insurance","NAM-INDIA":"Insurance","ABSLAMC":"Insurance",
    # Telecom
    "BHARTIARTL":"Telecom","IDEA":"Telecom","TATACOMM":"Telecom","INDUSTOWER":"Telecom",
    # ConsumerDur
    "HAVELLS":"ConsumerDur","CROMPTON":"ConsumerDur","TITAN":"ConsumerDur",
    # Retail / Logistics
    "TRENT":"Retail","DMART":"Retail",
    "CONCOR":"Logistics","BLUEDART":"Logistics",
}

# Live map starts as a copy of the fallback; refreshed by _refresh_sector_map()
STOCK_SECTOR_MAP: Dict[str, str] = dict(_STATIC_SECTOR_FALLBACK)

_sector_map_last_refresh: float = 0.0
_SECTOR_MAP_TTL: int = 6 * 3600   # refresh every 6 hours


def _fetch_index_constituents(session: requests.Session, index_name: str) -> List[str]:
    """Return trading symbols in an NSE index, or [] on failure."""
    url = "https://www.nseindia.com/api/equity-stockIndices"
    try:
        r = session.get(url, params={"index": index_name}, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        syms = [
            d["symbol"] for d in data
            if d.get("symbol") and d["symbol"] != index_name.replace(" ", "")
        ]
        return syms if len(syms) >= 5 else []
    except Exception:
        return []


def _refresh_sector_map() -> None:
    """
    Fetch constituent lists for every NSE sector index and rebuild
    STOCK_SECTOR_MAP in-place.  Uses the same NSE session pattern as
    get_nifty50_live() so it benefits from any existing cookies.

    Thread-safe: guarded by _SECTOR_MAP_LOCK.
    Falls back gracefully: if an index fetch fails or returns <5 stocks,
    the existing mapping for those symbols is preserved.

    Priority fix: stocks in multiple indices get the sector from the
    HIGHEST-priority index (see _NSE_INDEX_PRIORITY).  This prevents
    broad thematic indices (NIFTY INDIA DIGITAL→Telecom,
    NIFTY FINANCIAL SERVICES→Insurance) from overwriting specific sector
    indices (NIFTY IT→IT, NIFTY BANK→Bank) for shared constituents.
    """
    global _sector_map_last_refresh

    session = requests.Session()
    session.headers.update({
        "User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":      "application/json",
        "Referer":     "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    })
    # Warm up the session to get NSE cookies (required for API calls)
    try:
        session.get("https://www.nseindia.com/", timeout=10)
    except Exception:
        pass   # proceed anyway; cookies may still arrive on the API call

    new_map: Dict[str, str] = dict(_STATIC_SECTOR_FALLBACK)   # start from fallback
    # Track the priority at which each symbol was assigned, so a higher-priority
    # index can overwrite a lower-priority one but not vice versa.
    map_priority: Dict[str, int] = {sym: 0 for sym in new_map}  # fallback = priority 0
    fetched_any = False

    for index_name, sector_label in _NSE_INDEX_TO_SECTOR.items():
        syms = _fetch_index_constituents(session, index_name)
        if not syms:
            continue   # keep existing / fallback entries for this sector
        fetched_any = True
        priority = _NSE_INDEX_PRIORITY.get(index_name, 1)
        for sym in syms:
            sym_upper = sym.upper()
            existing_priority = map_priority.get(sym_upper, -1)
            if priority >= existing_priority:
                new_map[sym_upper] = sector_label
                map_priority[sym_upper] = priority
        time.sleep(0.15)   # be polite to NSE API

    if fetched_any:
        with _SECTOR_MAP_LOCK:
            STOCK_SECTOR_MAP.clear()
            STOCK_SECTOR_MAP.update(new_map)
        _sector_map_last_refresh = time.time()


def _maybe_refresh_sector_map() -> None:
    """Call from startup and from get_market_context() to keep map fresh."""
    if time.time() - _sector_map_last_refresh > _SECTOR_MAP_TTL:
        threading.Thread(target=_refresh_sector_map, daemon=True,
                         name="sector-map-refresh").start()

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
    "coil_streak_days": {},   # {sym: int} — consecutive days in top-tier BB+VDU
    "breadth_cache": None,
    "breadth_hist": [],
    "rs_div_hist": {},
    "param_registry": {
        "tanh_w": [], "inst_sigma": [], "prox_lambda": [],
        "stab_adj_scale": [], "stab_adj_obs": [], "pos52w_max": [],
    },
    "per_stock_winrate": {},
    "_setup_winrate": {},
    "_setup_winrate_counts": {},   # raw count per setup type — Kelly gate uses this
    "_stock_calib_counts":   {},   # raw count per ticker  — Kelly gate uses this
    "last_live_refresh": 0,
    "extraction_status": {
        "running": False, "done": 0, "total": 0,
        "errors": 0, "rate_limited": 0, "log": []
    },
    "mkt": {},
    "sector_returns": {},
    "sector_returns_10d": {},
    "top_sectors": set(),
    # BUG 14 FIX: Default changed from 7 to 14.  RSI-7 is hypersensitive and frequently
    # flags healthy trending stocks as overbought (RSI penalty up to -15 pts), suppressing
    # valid 5%-move candidates below the levels threshold.  RSI-14 is the professional
    # standard and aligns with how most practitioners interpret overbought/oversold.
    "rsi_period": 14,
    "min_avg_vol": 100_000,
    "sector_cap_enabled": False,
    # Progressive streaming: scored rows are pushed here as extraction completes
    # each stock. The SSE stream reads and clears this list, pushing "row" events
    # to the frontend so the table populates stock-by-stock without waiting.
    "_row_stream_queue": [],
    # ── CHG% fix: immutable previous-session close, captured BEFORE patch_live_bar()
    # overwrites iloc[-1].  This is the single source of truth for DayChg_pct across
    # extraction, screener GET, SSE prices patch, and premove endpoint.
    # Key: symbol (str) → Value: float (last fully-completed session close)
    "prev_close_cache": {},
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
    """
    Lookup order:
      1. In-memory STOCK_SECTOR_MAP  (live NSE index refresh, ~600 large-caps)
      2. sector_map.db _DB_CACHE     (full ~2500-stock mapping, loaded at startup)
      3. None                        (genuinely unknown — score degrades gracefully)
    """
    return get_sector_db(ticker, STOCK_SECTOR_MAP, _SECTOR_MAP_LOCK)


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


def _compute_coiling_score(
    bb_cs: Optional[float],
    vdu_cs: Optional[float],
    clv_cs: Optional[float],
    vcp_cs: Optional[float],
    sc_feature: float,
    vc_feature: float,
) -> float:
    """Pure pre-move coiling quality score (0–100).

    Uses only compression/accumulation signals that are valid BEFORE the
    breakout volume arrives.  Completely independent of vol_confirm so stocks
    that are coiling but not yet breaking score high here, not low.

    All inputs are [0,1] percentile ranks or normalised features.
    Missing inputs (None) are replaced with 0.5 (neutral) so the score
    degrades gracefully rather than crashing.
    """
    cfg = SCORE_CFG
    bb  = bb_cs  if bb_cs  is not None else 0.5
    vdu = vdu_cs if vdu_cs is not None else 0.5
    clv = clv_cs if clv_cs is not None else 0.5
    vcp = vcp_cs if vcp_cs is not None else 0.5
    sc  = float(np.clip(sc_feature,  0.0, 1.0))
    vc  = float(np.clip(vc_feature,  0.0, 1.0))
    raw = (bb  * cfg.COIL_W_BB  +
           vdu * cfg.COIL_W_VDU +
           clv * cfg.COIL_W_CLV +
           vcp * cfg.COIL_W_VCP +
           sc  * cfg.COIL_W_SC  +
           vc  * cfg.COIL_W_VC)
    return float(np.clip(raw * 100.0, 0.0, 100.0))


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
        patterns.append("Engulfing"); pts += cfg.CANDLE_PTS_ENGULFING
    if (lower_w >= cfg.CANDLE_HAMMER_LOWER_MULT * body
            and upper_w <= cfg.CANDLE_UPPER_WICK_MAX * rng and c > o
            and (c - l) / rng >= cfg.CANDLE_HAMMER_CLOSE_RATIO):
        patterns.append("Hammer"); pts += cfg.CANDLE_PTS_HAMMER
    if h <= ph and l >= pl:
        patterns.append("InsideBar"); pts += cfg.CANDLE_PTS_INSIDE_BAR
    if h > ph and l < pl and c > o and c > (h + l) / 2:
        patterns.append("OutsideBar"); pts += cfg.CANDLE_PTS_OUTSIDE_BAR
    if (body / rng > cfg.CANDLE_STRONG_BODY_RATIO and c > o
            and (c - l) / rng > cfg.CANDLE_STRONG_CLOSE_RATIO):
        patterns.append("StrongGreen"); pts += cfg.CANDLE_PTS_STRONG_GREEN
    if body / rng < cfg.CANDLE_HAMMER_BODY_RATIO and lower_w > cfg.CANDLE_BULL_DOJI_WICK_MULT * upper_w:
        patterns.append("BullDoji"); pts += cfg.CANDLE_PTS_BULL_DOJI
    if pc < po and prev_body / prev_rng > cfg.CANDLE_MSTAR_BODY_RATIO and c > o and c > (po + pc) / 2:
        patterns.append("MorningStar"); pts += cfg.CANDLE_PTS_MORNING_STAR
    _gap_min = max(prev_body * cfg.CANDLE_GAP_BODY_MULT, prev_rng * 0.01)
    if o > pc + _gap_min and c > o:
        patterns.append("GapContinue"); pts += cfg.CANDLE_PTS_GAP_CONTINUE
    return min(pts, 10.0), patterns


# ═════════════════════════════════════════════════════════════════
# 8. STRUCTURAL PATTERN DETECTORS  (Darvas, VCP, BB, CLV, Vol Dryup)
# ═════════════════════════════════════════════════════════════════

def darvas_box_score(df: pd.DataFrame, atr_v: float) -> dict:
    cfg  = SCORE_CFG
    # FIX-6: Added position_in_box=0.5 (neutral) to null dict.  The fallback path
    # in compute_signals calls darvas_r.get("position_in_box", 0.5) but this key
    # was never present in the null dict, so it always returned 0.5 regardless of
    # setup type — making the Breakout/Pullback inversion logic a no-op.
    null = {"darvas_score": 0.0, "box_high": np.nan, "box_low": np.nan,
            "in_box": False, "bars_in_box": 0, "box_atr_ratio": np.nan,
            "position_in_box": 0.5}
    if len(df) < 20:
        return null
    hh, hl, hc = df["high"], df["low"], df["close"]
    tr = pd.concat([(hh - hl), (hh - hc.shift(1)).abs(), (hl - hc.shift(1)).abs()], axis=1).max(axis=1)
    _atr_v = float(tr.ewm(alpha=1 / cfg.ATR_PERIOD, adjust=False).mean().iloc[-1]) \
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
        _box_low = float(hl.iloc[ws:i + 1].min())  # consolidation window only, not full tail
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
        .tail(60).dropna().mean()
    ) if len(hh) >= 10 else 0.5
    if np.isnan(_coiling_frac):
        _coiling_frac = 0.5
    _typical_dur = max(int(_coiling_frac * cfg.DARVAS_IDEAL_BARS), 3)
    time_score = float(np.clip(bars_in_box / (_typical_dur * 2.0 + 1e-9), 0.0, 1.0))
    darvas_score = round((tightness * 0.40 + pos_score * 0.35 + time_score * 0.25) * 10.0, 1)
    return {"darvas_score": darvas_score, "box_high": round(_box_high, 2),
            "box_low": round(_box_low, 2), "in_box": in_box,
            "bars_in_box": bars_in_box, "box_atr_ratio": round(bar, 2),
            "position_in_box": round(float(np.clip(pos_in_box, 0.0, 1.0)), 4)}


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
        return 0.5, 0.5   # neutral: insufficient history
    v = v.replace(0, np.nan).dropna()
    if len(v) < long:
        return 0.5, 0.5   # neutral: insufficient history after NaN drop
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
    # FIX: use hc.iloc[-1] (the last bar of the pattern slice, same as the rest of
    # detect_vcp) rather than c.iloc[-1] (today's live LTP from the full series).
    # c.iloc[-1] is today's patched bar; hc = c.iloc[:-1] so the pattern boundaries
    # exclude it.  Using c.iloc[-1] could push pos_sc > 1.0 on breakout days
    # (before the clip), discarding the signal that price has already moved.
    pos_sc   = float(np.clip((float(hc.iloc[-1]) - last_sl) / cons_rng, 0.0, 1.0))
    # Geometric mean of the 4 pattern conditions (contraction, compression, dryup, tightness).
    # Geometric mean is correct here: ALL four must be present simultaneously.
    # If any one is near-zero the composite collapses, unlike arithmetic mean which masks weak legs.
    # pos_sc applied as a separate position gate (multiplicative) — VCP at range lows scores near zero.
    _sub = np.array([contraction, vc_sc, vdu_sc, tight_sc], dtype=float)
    _sub = np.clip(_sub, 1e-9, 1.0)
    vcp_raw   = float(np.exp(np.log(_sub).mean()))   # geometric mean
    vcp_score = float(np.clip(vcp_raw * pos_sc, 0.0, 1.0))
    detected  = vcp_score >= SCORE_CFG.VCP_DETECT_SCORE_THRESH and len(recent) >= 2 and contraction >= SCORE_CFG.VCP_DETECT_CONTRACTION_THRESH
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
    end_p  = float(c.iloc[-1])   # Bug 1 fix: use today's close, not yesterday's
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
    # Use rolling mean of exactly sma200_n bars (proper SMA, not just tail mean which is identical here,
    # but rolling explicitly documents the intent and avoids off-by-one if c has an extra live bar appended).
    sma200   = float(c.rolling(sma200_n).mean().iloc[-1]) if len(c) >= sma200_n else float(c.mean())
    vol_ma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())

    # ── Weekly trend (resample daily → weekly, check EMA9 > EMA20) ───────
    # Used in compute_penalties to gate Breakout signals against the higher
    # timeframe trend.  Computed here once so it is available downstream.
    weekly_trend_up = True   # default: assume aligned until proven otherwise
    if SCORE_CFG.WEEKLY_TREND_CHECK and len(c) >= 20:
        try:
            _df_w = pd.DataFrame({"close": c.values, "high": h.values,
                                   "low": l.values, "volume": v.values})
            _df_w.index = pd.RangeIndex(len(_df_w))
            # Approximate weekly bars by grouping every 5 trading days.
            # FIX-4: The original stride-5 sample (range(4, len, 5)) silently drops
            # the last 1-4 bars of any partial week.  On a Thursday after a 3-day
            # rally the weekly EMA never sees that move, firing a phantom
            # WEEKLY_TREND_PENALTY on a perfectly valid setup.
            # Fix: build the stride-5 weekly closes, then append today's close as the
            # partial-week bar if there are any leftover daily bars after the last
            # full-week boundary.
            _stride_indices = list(range(4, len(_df_w), 5))
            _wc_vals = [float(_df_w["close"].iloc[max(0, i-4):i+1].iloc[-1])
                        for i in _stride_indices]
            # Append partial-week close (today) if not already captured
            _last_full = _stride_indices[-1] if _stride_indices else -1
            if len(_df_w) - 1 > _last_full:
                _wc_vals.append(float(_df_w["close"].iloc[-1]))
            _wc = pd.Series(_wc_vals, dtype=float)
            if len(_wc) >= 10:
                _we9  = _wc.ewm(span=9,  adjust=False).mean()
                _we20 = _wc.ewm(span=20, adjust=False).mean()
                weekly_trend_up = float(_we9.iloc[-1]) >= float(_we20.iloc[-1])
        except Exception:
            weekly_trend_up = True   # fail open

    return {
        "c": c, "h": h, "l": l, "v": v,
        "e9": e9, "e20": e20, "e50": e50, "e5": e5,
        "atr": atr, "tr": tr, "atr5": atr5, "atr20": atr20,
        "rsi": rsi, "sma200": sma200, "vol_ma20": vol_ma20,
        "weekly_trend_up": weekly_trend_up,
    }


def compute_features(ind: dict, ticker: str, ltp: float, day_vol: float,
                      day_hi: float, day_lo: float, day_o: float,
                      nifty_r5: Optional[float], nifty_r20: Optional[float],
                      market_ctx: dict, cs_state: dict,
                      param_registry: dict,
                      rsi_period: int = 14) -> dict:  # FIX: accept rsi_period so VolClimax uses correct period
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
    rsi_v   = float(rsi.iloc[-1]);  rsi_prev = float(rsi.iloc[-2])
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
    rs_combined = cs_rs * cfg.RS_BLEND_CS + rs_slope * cfg.RS_BLEND_SLOPE + abs_rs * cfg.RS_BLEND_ABS

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
    # Use get_sector() which checks STOCK_SECTOR_MAP first, then falls
    # through to _DB_CACHE (full 2500-stock DB) — fixes "?" for mid/small-caps.
    sect = get_sector(ticker.upper())
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
            rs_sect = min(1.0, rs_sect + cfg.RS_SECT_RANK_BONUS * sr_rank)
        if sect_r10 is not None and len(all_sv) > 2:
            accel_z  = (sect_ret - sect_r10) / max(float(pd.Series(all_sv).std()), 1e-4)
            rs_sect  = float(np.clip(rs_sect + cfg.RS_SECT_ACCEL_BONUS * np.tanh(accel_z), 0.0, 1.0))
    else:
        # Neutral 0.5 when sector unknown or sector return unavailable.
        # Hard 0.0 penalises unmapped/mid-small-cap stocks vs large-caps
        # that happen to have a sector ticker — unfair and score-deflating.
        rs_sect = 0.5

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
    # Bug 10 fix: exclude today's bar from historical baseline (v.iloc[:-1])
    # so intraday partial volume doesn't contaminate the mean/sigma reference.
    _v_hist   = v.iloc[:-1]
    vol_mu    = float(_v_hist.tail(20).mean()) if len(_v_hist) >= 5 else vol_ma20
    vol_sigma = max(float(_v_hist.tail(20).std()) if len(_v_hist) > 1 else vol_mu * 0.3, vol_mu * 0.05)
    vol_z     = (float(v.iloc[-1]) - vol_mu) / (vol_sigma + 1e-9)
    vol_ratio = float(v.iloc[-1]) / (vol_ma20 + 1e-9)
    # Institutional ratio (5-day avg / 20-day avg) — use historical bars only for baseline
    inst_ratio = float(_v_hist.tail(5).mean()) / (vol_ma20 + 1e-9)
    _inst_hist = (_v_hist.rolling(5).mean() / (_v_hist.rolling(20).mean() + 1e-9)).dropna()
    if len(_inst_hist) >= 20:
        ic = float(_inst_hist.tail(60).median())
        is_ = max(float(_inst_hist.tail(60).std()), 0.05)
    else:
        ic, is_ = cfg.INST_FALLBACK_CENTRE, cfg.INST_FALLBACK_SCALE
    inst_feature = float(1.0 / (1.0 + np.exp(-((inst_ratio - ic) / is_))))

    # Volume trend (5-bar slope, normalised)
    # Exclude today's bar so an intraday volume surge doesn't inflate the slope
    # and trend percentile that are supposed to measure pre-move accumulation.
    if n >= 8:
        _v_trend = v.iloc[:-1]
        v5_vals  = _v_trend.tail(5).values.astype(float)
        v5_slope = float(np.polyfit(np.arange(5, dtype=float), v5_vals, 1)[0]) / (vol_mu + 1e-9)
        v_trend_pct = float((_v_trend.rolling(5).mean().dropna() <= float(v5_vals.mean())).mean())
        vol_signal = float(np.clip(0.5 + v5_slope * 2.0, 0.0, 1.0)) * 0.6 + v_trend_pct * 0.4
        # Volume spike decay: a true outlier vol spike (above stock's own P90) means the move
        # is already done — discount so Breakout doesn't reward chasing a news/gap event.
        # Only applies to the most recent completed bar (T-1), not the 5-bar slope.
        _t1_vol = float(_v_trend.iloc[-1])
        _vol_p90 = float(_v_trend.tail(60).quantile(0.90)) if len(_v_trend) >= 20 else vol_mu * 2.5
        if _t1_vol > _vol_p90:
            _spike_excess = (_t1_vol - _vol_p90) / max(_vol_p90, 1.0)
            _spike_decay  = float(np.clip(_spike_excess / 2.0, 0.0, 0.7))
            vol_signal    = float(np.clip(vol_signal * (1.0 - _spike_decay), 0.0, 1.0))
    else:
        vol_signal = 0.5

    # Up-volume skew
    # FIX-7: Original code used c.diff() and v on the full series including today,
    # causing today's bar to appear in both the current up_vol/dn_vol measurement
    # and the uv_hist reference window (self-comparison lookahead).  On a strong
    # green day this inflated uv_feature, rewarding the move AFTER it started.
    # Fix: compute everything on historical bars only (iloc[:-1]).
    uv_feature = 0.5
    if n >= 21:
        _c_h = c.iloc[:-1];  _v_h = v.iloc[:-1]
        up_mask  = _c_h.diff() > 0;  dn_mask = _c_h.diff() < 0
        up_vol   = float(_v_h[up_mask].tail(20).sum())
        dn_vol   = float(_v_h[dn_mask].tail(20).sum())
        uv_ratio = up_vol / (dn_vol + 1e-9)
        _nh = len(_c_h)
        uv_hist  = pd.Series([
            _v_h[up_mask].iloc[max(0, i - 20):i].sum() / (_v_h[dn_mask].iloc[max(0, i - 20):i].sum() + 1e-9)
            for i in range(20, min(60, _nh))
        ], dtype=float)
        uv_feature = float((uv_hist <= uv_ratio).mean()) if len(uv_hist) >= 5 else (0.7 if uv_ratio > 1.0 else 0.4)

    # Close Position Ratio (close location in range)
    cpr_feature = 0.5
    if n >= 10:
        hl_r    = (h - l).replace(0, np.nan)
        cpr_raw = ((c - l) / hl_r).dropna()
        # Bug 14 fix: exclude today from both the current mean and the reference
        # distribution so today's strong close doesn't inflate its own percentile.
        cpr_hist = cpr_raw.iloc[:-1]
        cpr10    = float(cpr_hist.tail(10).mean())
        cpr_h    = cpr_hist.rolling(10).mean().dropna()
        cpr_feature = float((cpr_h <= cpr10).mean()) if len(cpr_h) >= 10 else round(cpr10, 3)

    # Spread compression + rising close
    # Exclude today's bar from the current-bar (qa) computation — a gap-up open
    # immediately produces compression + rising close, making a post-move bar
    # look like a pre-move coil.  The historical reference (sch) also excludes
    # today so qa is not compared against a window that contains itself.
    sc_feature = 0.5
    if n >= 15:
        _h_hist = h.iloc[:-1]; _l_hist = l.iloc[:-1]; _c_hist = c.iloc[:-1]
        r5d  = float(_h_hist.tail(5).max() - _l_hist.tail(5).min())
        r10d = float(_h_hist.tail(10).max() - _l_hist.tail(10).min())
        comp = 1.0 - (r5d / (r10d + 1e-9))
        try:
            cslope = float(np.polyfit(range(5), _c_hist.tail(5).values, 1)[0]) / (atr_v + 1e-9)
        except Exception:
            cslope = 0.0
        qa = max(0.0, comp) * max(0.0, cslope)
        if n >= 20:
            x_    = np.arange(5, dtype=float)
            sx, sx2 = x_.sum(), (x_ ** 2).sum()
            dn_   = 5 * sx2 - sx ** 2
            # FIX-9: Use c.iloc[:-1].values (historical only) so the sliding window
            # reference distribution does not contain today's bar.  Previously
            # cv = c.values included today, meaning the last window in sw compared
            # qa against itself — biasing sc_feature upward on strong compression days.
            cv    = _c_hist.values.astype(float)
            hv    = _h_hist.values.astype(float)
            lv    = _l_hist.values.astype(float)
            sw    = np.lib.stride_tricks.sliding_window_view(cv, 5)
            slop_ = (5 * (sw * x_).sum(axis=1) - sx * sw.sum(axis=1)) / (dn_ + 1e-9)
            av_   = atr.iloc[:-1].values.astype(float)[4:]
            nw    = min(len(slop_), len(av_))
            slop_ = slop_[:nw]; av_ = av_[:nw]
            ra5  = np.array([hv[i:i+5].max() - lv[i:i+5].min()             for i in range(nw)])
            ra10 = np.array([hv[max(0,i-4):i+6].max() - lv[max(0,i-4):i+6].min() + 1e-9 for i in range(nw)])
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
        # Onset window = ATR_FAST // 2 — expansion is only "just started" for
        # the first half of the fast ATR window after compression ends.
        _onset_bars = max(2, cfg.ATR_FAST // 2)
        if bars_since is not None and bars_since <= _onset_bars:
            atr_exp_feature = float(np.clip(1.0 / bars_since, 0.0, 1.0))

    # ── Position in 52-week range ─────────────────────────────────
    # Exclude today's bar: on a breakout day today's high IS the new 52W high,
    # making pos52w = 1.0 — the signal is measuring the move, not predicting it.
    n250   = min(cfg.PERCENTILE_WINDOW_LONG, n)
    hi250  = float(h.iloc[:-1].tail(n250).max()); lo250 = float(l.iloc[:-1].tail(n250).min())
    pos52w = (ltp - lo250) / (hi250 - lo250 + 1e-9)
    _c_hist_52 = c.iloc[:-1]
    pos_ser = (_c_hist_52 - _c_hist_52.rolling(n250).min()) / (_c_hist_52.rolling(n250).max() - _c_hist_52.rolling(n250).min() + 1e-9)
    pos_pct = percentile_last(pos_ser, min(n250, len(pos_ser)))
    if pd.isna(pos_pct):
        pos_pct = pos52w

    # ── Stability (% of last-20 closes positive) ──────────────────
    # Exclude today's bar — a big up day would otherwise boost stability and
    # simultaneously lift the P20 threshold that judges it in penalties.
    if n >= 21:
        stability = float((c.iloc[-21:-1].pct_change().dropna() > 0).mean())
    elif n >= 11:
        stability = float((c.iloc[-11:-1].pct_change().dropna() > 0).mean())
    else:
        stability = 0.5
    if n >= 40:
        stab_ser = c.iloc[:-1].pct_change().rolling(20).apply(
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
        # bb_width_compression_score returns (pct_below, 1-pct_below).
        # pct_below = fraction of history with NARROWER bands than today → high = wide (not squeezed).
        # We want the SQUEEZE score = 1 - pct_below = index [1].  This is correct.
        _, bb_cs = bb_width_compression_score(c)
    vdu_cs = cs_state.get("cs_vol_dryup", {}).get(ticker)
    if vdu_cs is None:
        _, vdu_cs = volume_dryup_score(v)
    clv_cs = cs_state.get("cs_clv_accum", {}).get(ticker)
    if clv_cs is None:
        _, clv_cs = clv_accumulation_score(c, h, l, v)
    vcp_cs = cs_state.get("cs_vcp", {}).get(ticker)

    # ── ADR / base data ───────────────────────────────────────────
    # BUG 4 FIX: base_hi was using h.iloc[:-2].tail(20) (excluded yesterday AND today),
    # but ext_hist in compute_signals uses h.iloc[:-1].rolling(20).max().shift(1) which
    # INCLUDES yesterday's high.  The mismatch caused valid Breakout setups to appear
    # below base_hi (Coiling) because the two references disagreed by one bar.
    # Fix: use h.iloc[:-1].tail(20) for base_hi to match ext_hist's shift(1) convention.
    # Both now represent the 20-day high EXCLUDING today's bar — fully consistent.
    base_hi  = float(h.iloc[:-1].tail(20).max());  base_lo = float(l.iloc[:-1].tail(20).min())
    base_rng = base_hi - base_lo + 1e-9
    breakout_ext = (ltp - base_hi) / (atr_v + 1e-9)

    # ── Percentile ranks for VolClimax detection ──────────────────
    # All three P-ranks are derived from the stock's own rolling history,
    # so no fixed price/RSI level is baked in.
    if n >= 40:
        _vr_hist_all = (v.iloc[:-1] / (v.iloc[:-1].rolling(20).mean() + 1e-9)).dropna()
        vol_prank    = float((_vr_hist_all < vol_ratio).mean()) if len(_vr_hist_all) >= 20 else 0.5
        _rsi_hist    = _rsi_wilder(c, rsi_period).iloc[:-1].dropna()  # FIX: was hardcoded rsi_wilder(c, 7); must match scoring RSI period
        rsi_prank    = float((_rsi_hist < rsi_v).mean())        if len(_rsi_hist)    >= 14 else 0.5
        _ext_hist    = ((c.iloc[:-1] - h.iloc[:-1].rolling(20).max().shift(1)) /
                        (atr.iloc[:-2] + 1e-9)).dropna()
        ext_prank    = float((_ext_hist < breakout_ext).mean()) if len(_ext_hist)    >= 20 else 0.5
    else:
        vol_prank = rsi_prank = ext_prank = 0.5

    # ── OI buildup (F&O stocks) ───────────────────────────────────
    _oi_feature = 0.0  # placeholder: df not available here
    # NOTE: df not available here; must be passed or handled in aggregate

    return dict(
        # Meta
        regime=regime, ltp=ltp, atr_v=atr_v, atr_pct=atr_pct,
        e9_v=e9_v, e20_v=e20_v, e50_v=e50_v,
        rsi_v=rsi_v, rsi_prev=rsi_prev,
        # RS
        rs_combined=rs_combined, acc_sc=acc_sc, rs_div_pct=rs_div_pct,
        abs_rs=abs_rs, rs_sect=rs_sect, sect_name=sect_name,
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
        # VolClimax percentile ranks (all derived from stock's own history)
        vol_prank=vol_prank, rsi_prank=rsi_prank, ext_prank=ext_prank,
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
    atr = ind["atr"]
    atr5, atr20 = ind["atr5"], ind["atr20"]
    e9, e20, _e50 = ind["e9"], ind["e20"], ind["e50"]
    rsi        = ind["rsi"]

    atr_v   = feat["atr_v"]
    e9_v    = feat["e9_v"];   e20_v = feat["e20_v"];   e50_v = feat["e50_v"]
    rsi_v   = feat["rsi_v"];  _rsi_p = feat["rsi_prev"]
    _vol_mu = feat["vol_mu"]; vol_z = feat["vol_z"]
    base_hi = feat["base_hi"]; base_lo = feat["base_lo"]; base_rng = feat["base_rng"]
    breakout_ext = feat["breakout_ext"]

    n = len(c)

    # ── Determine setup type ──────────────────────────────────────
    # Volume threshold: P85 of HISTORICAL daily volume (exclude today)
    # so intraday partial volume is on the same basis as a full daily bar.
    vol_bo_thresh = float(v.iloc[:-1].tail(cfg.PERCENTILE_WINDOW_PERSHORT).quantile(cfg.BO_VOL_PERCENTILE)) \
                    if n >= 10 else float(v.mean())

    # Extension history on historical bars only (exclude today).
    ext_hist = ((c.iloc[:-1] - h.iloc[:-1].rolling(20).max().shift(1)) /
                (atr.iloc[:-2] + 1e-9)).dropna()
    ext_p10  = float(ext_hist.quantile(0.10)) if len(ext_hist) >= 20 else -1.5
    ext_p90  = float(ext_hist.quantile(0.90)) if len(ext_hist) >= 20 else  0.3
    ext_p10  = min(ext_p10, -0.3)
    ext_p90  = max(ext_p90,  0.5)

    t1_vol_ratio  = float(v.tail(3).max()) / (vol_ma20 + 1e-9)
    hi10d         = float(h.tail(10).max())
    washout_depth = (hi10d - ltp) / (atr_v + 1e-9)
    t1_bar_rng    = float(h.iloc[-1]) - float(l.iloc[-1])
    t1_close_pos  = (float(c.iloc[-1]) - float(l.iloc[-1])) / (t1_bar_rng + 1e-9)

    # ── Reversal ─────────────────────────────────────────────────
    # Reversal vol threshold: stock must be in top (1 - REVERSAL_VOL_PRANK) of
    # its own historical vol ratio, not a fixed 1.3× absolute multiple.
    _rev_vol_p = float((v.iloc[:-1] / (v.iloc[:-1].rolling(20).mean() + 1e-9))
                       .dropna().quantile(cfg.REVERSAL_VOL_PRANK))                  if n >= 20 else 1.3
    is_reversal = (rsi_v < float(rsi.iloc[:-1].tail(cfg.PERCENTILE_WINDOW_PERSHORT).quantile(0.35)) and  # Bug 8 fix: exclude today from ref distribution
                   t1_vol_ratio >= _rev_vol_p and
                   washout_depth >= cfg.REVERSAL_MIN_WASHOUT_ATR and      # use config constant, not hardcoded 1.5
                   t1_close_pos >= cfg.REVERSAL_TAIL_MIN_POS)              # tail must close in upper 40%+ of bar

    # ── Pullback geometry ────────────────────────────────────────
    # Distance from each EMA in ATR units (signed: + = above, - = below)
    dist_e9  = (ltp - e9_v)  / (atr_v + 1e-9)
    dist_e20 = (ltp - e20_v) / (atr_v + 1e-9)
    dist_e50 = (ltp - e50_v) / (atr_v + 1e-9)

    # "Near EMA20" = within this stock's own historical P20-P80 band of
    # (close - EMA20) / ATR, derived from its own price history.
    _nc = min(n - 1, len(e20) - 1, len(atr) - 2)
    if _nc >= 20:
        _d20_hist = ((c.iloc[1:_nc+1].values - e20.iloc[1:_nc+1].values) /
                     (atr.iloc[:_nc].values + 1e-9))
        _d20_hist = _d20_hist[np.isfinite(_d20_hist)]
        _near_lo = float(np.percentile(_d20_hist, 20)) if len(_d20_hist) >= 10 else -1.5
        _near_hi = float(np.percentile(_d20_hist, 80)) if len(_d20_hist) >= 10 else  1.5
    else:
        _near_lo, _near_hi = -1.5, 1.5

    near_ema20_band = _near_lo <= dist_e20 <= _near_hi
    near_ema9_band  = abs(dist_e9)  <= abs(_near_hi)
    above_e50       = dist_e50 > -1.0   # not more than 1 ATR below EMA50

    # "Real pullback" = price has retreated at least P25 of this stock's
    # own historical pullback depth distribution.
    _pb_depth = (float(h.tail(10).max()) - ltp) / (atr_v + 1e-9)
    if n >= 30:
        _pb_hist = ((h.iloc[:-1].rolling(10).max() - c.iloc[:-1]) /
                    (atr.iloc[:-2] + 1e-9)).dropna()
        _pb_hist = _pb_hist[_pb_hist > 0]
        _pb_min  = float(_pb_hist.quantile(0.25)) if len(_pb_hist) >= 10 else 0.3
    else:
        _pb_min = 0.3
    real_pb = _pb_depth >= _pb_min

    # BUG 5 FIX: pace-adjusted volume can be massively inflated by an early news/gap spike
    # (e.g. at 09:45 with 15% session elapsed, a 2× vol spike projects to 13× full-day vol).
    # Cap the pace adjustment at a maximum multiplier based on stock's own P95 historical volume
    # so a single large early trade doesn't prematurely trigger vol_confirm.
    _vol_sess_frac = max(elapsed_frac, 1e-3)   # avoid div/0 at open
    _raw_pace_vol  = day_vol / _vol_sess_frac   # uncapped pace projection
    _vol_p95_hist  = float(v.iloc[:-1].tail(60).quantile(0.95)) if n >= 20 else float(v.mean()) * 3.0
    # Allow pace-adjustment to project up to 3× the stock's own P95 historical volume.
    # This prevents a 1-minute news-driven spike from masquerading as a confirmed full-day breakout.
    day_vol_sc  = float(np.clip(_raw_pace_vol, 0, _vol_p95_hist * 3.0))
    vol_dryup   = day_vol_sc < float(v.iloc[:-1].tail(20).median() + 1e-9) if n >= 20 else True

    # ── Breakout geometry ────────────────────────────────────────
    # Price is at or just above the pivot (20-day high), in ATR terms.
    # Lower bound uses ext_p10 (data-driven) instead of a magic -0.5.
    # FIX: at_pivot was True for ANY stock within ext_p10..ext_p90, which includes
    # stocks sitting 1-2 ATR BELOW their 20d high. A genuine pivot breakout requires
    # price to be within 0.3 ATR of the base_hi (approaching or touching the pivot).
    # The ext_p90 upper cap is preserved to avoid labeling already-extended moves.
    _BO_PIVOT_ENTRY_ATR = 0.3   # max distance below base_hi to qualify as "at pivot"
    at_pivot = breakout_ext >= -_BO_PIVOT_ENTRY_ATR and breakout_ext <= ext_p90

    # ── Session guard: suppress vol_confirm before VOL_CONFIRM_MIN_FRAC ──
    # Even pace-adjusted volume in the first ~30% of the session is too noisy
    # to confirm a Breakout.  Below this threshold vol_confirm is forced False
    # and the Breakout label is suppressed; the stock is labelled "Developing BO"
    # via the horizon note instead.
    _vol_confirm_allowed = elapsed_frac >= cfg.VOL_CONFIRM_MIN_FRAC
    # FIX: vol_confirm was purely volume-based — a flat stock with slightly elevated
    # volume got vol_confirm=True. A real breakout also requires price to be at or
    # above the pivot (within BO_ENTRY_BUFFER_ATR above base_hi).
    _price_at_pivot = ltp >= base_hi - (_BO_PIVOT_ENTRY_ATR * atr_v)
    vol_confirm = (day_vol_sc >= vol_bo_thresh) and _vol_confirm_allowed and _price_at_pivot

    # ── Intraday market circuit breaker ──────────────────────────────────
    # When Nifty is down >= 2% on the day, issuing Breakout labels is
    # dangerous.  Force all Breakout candidates to Pullback.
    _nifty_intraday_chg = float(cs_state.get("nifty_intraday_chg") or 0.0)
    _market_kill = _nifty_intraday_chg <= cfg.NIFTY_INTRADAY_KILL
    # Priority:
    #   1. Reversal  — oversold panic with volume spike
    #   2. Breakout  — at pivot WITH volume confirmation AND session/market ok
    #      (MOVED before PB-EMA: a stock at EMA20 + at pivot + vol confirm = Breakout, not PB-EMA)
    #   3. PB-EMA    — retreated to EMA20/EMA9, volume quiet, above EMA50
    #   4. Coiling   — strong compression signals but vol not confirmed yet (pre-move)
    #   5. PB-Dry    — real pullback, vol drying, not yet at EMA (deeper but healthy)
    #   6. PB-Deep   — real pullback above EMA50 but not near an EMA (still correcting)
    #   7. Base      — no real pullback signal, stock range-bound / base-building
    if is_reversal:
        setup_type = "Reversal"

    elif at_pivot and vol_confirm and not _market_kill:
        # At the base pivot with volume confirmation — genuine breakout.
        # Checked BEFORE PB-EMA so a stock at EMA20 + at_pivot + vol = Breakout, not PB-EMA.
        setup_type = "Breakout"

    elif real_pb and (near_ema20_band or near_ema9_band) and above_e50:
        # Textbook pullback: retreated, price is AT the EMA support band, trend intact
        setup_type = "PB-EMA"

    elif breakout_ext > ext_p90 and vol_confirm and not _market_kill:
        # Extended with volume but price already > 1.5 ATR above base — move already done.
        # Label as PB-Deep so it doesn't appear as a fresh entry in Breakout filters.
        if feat["breakout_ext"] > 1.5:
            setup_type = "PB-Deep"
        else:
            setup_type = "Breakout"

    elif at_pivot and not vol_confirm:
        # At pivot but vol not confirmed yet — highest-value pre-move state.
        # Classify as Coiling so users can filter for it separately.
        # Will upgrade to Breakout the moment volume confirms.
        setup_type = "Coiling"

    elif real_pb and above_e50 and vol_dryup:
        # Real pullback, volume drying up, NOT yet at EMA — deeper coiling pullback
        setup_type = "PB-Dry"

    elif breakout_ext > ext_p90 and not vol_confirm:
        # Extended without volume — post-breakout consolidation, treat as deep pullback
        setup_type = "PB-Deep"

    elif real_pb and above_e50:
        # Still declining but above EMA50 — developing correction
        setup_type = "PB-Deep"

    else:
        # Default: base building, no strong directional signal
        setup_type = "Base"

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
        # Normalise high-spread by the stock's own ATR — 1 ATR of spread
        # across 8 bars is a natural reference point, not an arbitrary 1.0.
        flat  = max(0.0, 1.0 - min(hsp, 1.0))   # hsp already in ATR units; 1 ATR spread = 0 flatness
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
        bp_bonus = cfg.COIL_BP_BONUS_CAP / (1.0 + np.exp(-(1.0 / bs) * (base_pos - bc)))
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

    if setup_type in ("Breakout", "Coiling"):
        # Both Breakout and Coiling are at or near the pivot — use pivot-distance proximity.
        # Coiling = vol not confirmed yet but price is at base_hi; same ideal_d logic applies.
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
        # d_trig > 0 : price is BELOW base_hi (approaching trigger) -- good
        # d_trig = 0 : price is exactly AT base_hi -- ideal
        # d_trig < 0 : price is ABOVE base_hi (already extended) -- penalise
        #
        # Bug fix: the old code used abs(d_trig - ideal_d) then added abs(d_trig)
        # when d_trig < 0 -- but ideal_d is always positive (distance BEFORE trigger),
        # so when d_trig is negative the abs() already under-penalises extension.
        # Additionally the extra abs(d_trig) term was added AFTER the ideal_d
        # subtraction, meaning a stock 0.5 ATR above base scored the same as one
        # 0.5 ATR below -- both gave d_adj = ideal_d + 0.5.  Extension should
        # always score worse than the equivalent distance below the trigger.
        # Fix: when d_trig < 0 (extended), use a steeper penalty multiplier so
        # the proximity score decays faster above base than below it.
        if d_trig >= 0:
            # Approaching trigger: penalise deviation from ideal approach distance
            d_adj = abs(d_trig - ideal_d)
        else:
            # Already above trigger: distance above base + full ideal_d gap
            # Use 2x multiplier so extension decays score faster than approach
            d_adj = (ideal_d + abs(d_trig)) * cfg.PROX_BO_OVERSHOOT_MULT
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
        pd_sc = abs(d_fi) if d_fi >= 0 else abs(d_fi) * cfg.PROX_OVERSHOOT_MULT
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
        d_time  = float(np.clip(darvas_r.get("bars_in_box", 0) / (cfg.DARVAS_IDEAL_BARS * 2.0), 0.0, 1.0))
        darvas_sc = d_tight * cfg.DARVAS_W_TIGHT + dp_sc * cfg.DARVAS_W_POS + d_time * cfg.DARVAS_W_TIME
    else:
        # Fallback: no valid box_high — use raw darvas_score but still apply
        # setup-type inversion to position_in_box so Pullback isn't scored
        # as if it were a Breakout.
        raw_ds   = darvas_r["darvas_score"] / 10.0
        dp_raw   = darvas_r.get("position_in_box", 0.5)
        dp_sc_fb = float(np.clip(dp_raw if setup_type == "Breakout"
                                  else 1.0 - dp_raw, 0.0, 1.0))
        darvas_sc = raw_ds * cfg.DARVAS_FB_W_RAW + dp_sc_fb * cfg.DARVAS_FB_W_POS

    # ── Sweep bonus (false-break reversal) ───────────────────────
    sweep_sc = 0.0
    if n >= 5:
        prior_sup   = float(l.iloc[:-1].tail(5).min())  # exclude today so day_lo < prior_sup can fire
        lower_wick  = min(day_o, ltp) - day_lo
        # Bug 12 fix: build vol-z reference from historical bars only
        _v_h        = v.iloc[:-1]
        vz_hist     = ((_v_h - _v_h.rolling(20).mean()) / (_v_h.rolling(20).std() + 1e-9)).tail(60)
        vz_p60      = float(vz_hist.quantile(0.60)) if len(vz_hist) >= 20 else 1.0
        if (day_lo < prior_sup and ltp > day_o and
                lower_wick >= cfg.SWEEP_WICK_MIN_ATR * atr_v and vol_z >= vz_p60):
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
    # Use session fraction instead of an absolute 30-minute floor
    _vel_min_mins = int(session_mins * cfg.VOL_VELOCITY_MIN_FRAC)
    if elapsed_mins >= _vel_min_mins and elapsed_mins < session_mins:
        vr = day_vol / (feat["vol_mu"] + 1e-9)
        vol_velocity = float(np.clip(np.tanh(vr - 1.0), 0.0, 1.0))

    # ── ATR potential (inverted) ──────────────────────────────────
    atr_hist_pct = (_atr(df, cfg.ATR_PERIOD).iloc[:-1] / c.iloc[:-1] * 100).tail(cfg.PERCENTILE_WINDOW_PERSHORT).dropna()
    atr_pct_rank = float((atr_hist_pct <= feat["atr_pct"]).mean()) if len(atr_hist_pct) >= 10 else 0.5
    atp_sc       = 1.0 - atr_pct_rank

    # ── Candle patterns ───────────────────────────────────────────
    # Patterns are detected on the live bar but only contribute to score
    # once the bar is essentially closed (>= CANDLE_CONFIRM_FRAC of session).
    # Before that threshold they are reported in Patterns but cdl_sc = 0 so
    # they don't inflate the score on an incomplete candle.
    raw_cdl, cdl_names = detect_candle_patterns(
        day_o, day_hi, day_lo, ltp,
        float(df["open"].iloc[-2]), float(df["high"].iloc[-2]),
        float(df["low"].iloc[-2]),  float(df["close"].iloc[-2]),
    )
    _bar_confirmed = elapsed_frac >= cfg.CANDLE_CONFIRM_FRAC
    cdl_sc = min(raw_cdl * 0.1, 0.5) if _bar_confirmed else 0.0   # normalise to 0-0.5

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
        stab_adj = cfg.STAB_ADJ_FALLBACK_HI if feat["stability"] >= cfg.STAB_FALLBACK_THRESH else 0.0

    # ── VCVE (volume × volatility-contraction interaction) ────────
    ii = feat["inst_ratio"] * (1.0 - min(feat["vc_ratio_now"], 1.0))
    vcve_sc = float(np.clip(np.tanh(ii / max(feat["inst_ratio"] * 0.4, 0.1)), 0.0, 1.0))

    # ── Volume quietness (Pullback) ───────────────────────────────
    vr_hist = (v.iloc[:-1] / (v.iloc[:-1].rolling(20).mean() + 1e-9)).dropna()
    vr_now  = float(v.iloc[-1]) / (feat["vol_mu"] + 1e-9)
    vol_quiet_pct = float((vr_hist >= vr_now).mean()) if len(vr_hist) >= 10 else float(np.clip(1.0 - vr_now, 0.0, 1.0))

    # ── Days since last pivot break ───────────────────────────────────────────
    # BUG 3 FIX: Previous logic walked backwards looking for the first bar below
    # base_hi after being above it.  When price STAYED above base_hi continuously
    # (e.g. broke out 5 days ago and never pulled back), _was_above stayed True
    # but the elif never fired → _days_since_break_s = 0 (looked "fresh").
    # Fix: after the loop, if _was_above=True and count is still 0, it means price
    # has been above base_hi for the entire lookback window — set count to the
    # last index where we confirmed price was above (maximum staleness).
    _base_hi_s = feat.get("base_hi", ltp)
    _days_since_break_s = 0
    if len(df) >= 5 and _base_hi_s > 0:
        _closes_s = df["close"].iloc[:-1].values[::-1]  # yesterday first
        _was_above = False
        _last_above_idx = 0
        for _i_s, _cl_s in enumerate(_closes_s[:20]):
            if _cl_s >= _base_hi_s:
                _was_above = True
                _last_above_idx = _i_s   # track most-recent (closest to today) bar above pivot
            elif _was_above:
                # first bar below pivot AFTER being above = break happened _i_s bars ago
                _days_since_break_s = _i_s
                break
        # BUG 3 FIX: if _was_above but loop ended without an "elif _was_above" hit,
        # price has been above base_hi the entire window — use _last_above_idx+1
        # so a 5-day-old breakout that never pulled back correctly shows as stale.
        if _was_above and _days_since_break_s == 0:
            _days_since_break_s = _last_above_idx + 1

    return dict(
        setup_type=setup_type,
        days_since_break=_days_since_break_s,
        # Primary signals (0-1)
        coil_sc=coil_sc, prox_sc=prox_sc,
        vcp_sc=_vcp_sc, darvas_sc=darvas_sc,
        sweep_sc=sweep_sc, vwap_sc=vwap_sc,
        oi_sc=oi_sc, vol_velocity=vol_velocity, atp_sc=atp_sc,
        cdl_sc=cdl_sc, stab_adj=stab_adj, vcve_sc=vcve_sc,
        vol_quiet_pct=vol_quiet_pct,
        # Confirmation flags (used in aggregate_score / penalties)
        vol_confirm=vol_confirm, bar_confirmed=_bar_confirmed,
        market_kill=_market_kill, vol_confirm_allowed=_vol_confirm_allowed,
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
        # ── New: vol build-up progress + breakout freshness ─────────────────────
        vol_build_pct=min(day_vol_sc / (vol_bo_thresh + 1e-9), 1.5),
        breakout_mins_ago=(max(0, elapsed_mins - int(cfg.VOL_CONFIRM_MIN_FRAC * session_mins))
                           if (vol_confirm and setup_type == "Breakout") else None),
    )


def compute_penalties(feat: dict, sig: dict, market_ctx: dict,
                       ind: dict, df: pd.DataFrame,
                       ltp: float, rsi_period: int,
                       adv_threshold: float) -> dict:
    """Stage 4 — continuous penalties mapped to score-point deductions."""
    cfg = SCORE_CFG
    c, h, _l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    atr = ind["atr"]
    atr_v    = feat["atr_v"];   atr_pct  = feat["atr_pct"]
    rsi_v    = feat["rsi_v"];   vol_mu   = feat["vol_mu"]
    sma200   = ind["sma200"]
    _vol_ma20 = ind["vol_ma20"]
    n        = len(c)
    regime   = feat["regime"]

    penalties: Dict[str, float] = {}

    # ── RSI overbought (adaptive P90) ────────────────────────────
    rsi_all = _rsi_wilder(c, rsi_period)
    # Bug 5 fix: exclude today's RSI from the reference window so an extreme
    # current reading doesn't shift the threshold that judges it.
    rsi_p90 = float(rsi_all.iloc[:-1].tail(cfg.PERCENTILE_WINDOW_PERSHORT).quantile(0.90)) if len(rsi_all) >= cfg.MIN_BARS_RSI else cfg.RSI_OB_FALLBACK
    if rsi_v > rsi_p90:
        z = (rsi_v - rsi_p90) / max(float(rsi_all.iloc[:-1].tail(20).std()), 1.0)
        penalties["rsi_ob"] = float(np.clip(8.0 * np.tanh(z), 0.0, cfg.RSI_OB_CAP))

    # ── Abnormally low volume ─────────────────────────────────────
    # Bug 11 fix: exclude today's bar from the reference quantile so a partial
    # intraday volume doesn't falsely look like an "abnormally low" historical bar.
    vol_p05 = float(v.iloc[:-1].tail(60).quantile(0.05)) if len(v) >= 20 else vol_mu * 0.10
    prev_v  = float(v.iloc[-2]) if len(v) >= 2 else float(v.iloc[-1])  # Bug 2 fix: use yesterday, not today
    if prev_v < vol_p05:
        vz = (vol_p05 - prev_v) / max(float(v.tail(20).std()), 1.0)
        penalties["vol_low"] = float(np.clip(6.0 * np.tanh(vz), 0.0, cfg.VOL_LOW_CAP))

    # ── Very low ATR (dead stock) ─────────────────────────────────
    # Exclude today: a breakout widens ATR, shifting the P10 threshold upward
    # and softening the very penalty that should flag low-volatility stocks.
    atr_hist = (atr.iloc[:-1] / c.iloc[:-1]).tail(60).dropna()
    if len(atr_hist) >= 10:
        atr_p10 = float(atr_hist.quantile(0.10))
        if atr_pct / 100.0 < atr_p10:
            excess = atr_p10 - atr_pct / 100.0
            penalties["atr_low"] = float(np.clip(excess / max(atr_p10, 1e-9) * 5.0, 0.0, 5.0))

    # ── Below SMA200 ──────────────────────────────────────────────
    # Threshold: P20 of historical (close - SMA200) / ATR distribution
    # so stocks that habitually trade near their SMA200 aren't penalised
    # more than stocks that regularly pull far below it.
    # Exclude today's close: it would shift the P20 threshold that judges it.
    sma200_gap_atr = (ltp - sma200) / (atr_v + 1e-9)
    if n >= 40:
        _sma_n    = min(cfg.SMA_TREND, n)
        _sma_hist = c.iloc[:-1].rolling(_sma_n).mean()
        _gap_hist = ((c.iloc[:-1] - _sma_hist) / (atr.iloc[:-1] + 1e-9)).dropna()
        _gap_p20  = float(_gap_hist.quantile(0.20)) if len(_gap_hist) >= 20 else -0.5
    else:
        _gap_p20 = -0.5
    if sma200_gap_atr < _gap_p20:
        excess = abs(sma200_gap_atr - _gap_p20)
        penalties["sma200"] = float(np.clip(10.0 * np.tanh(excess / 2.0), 0.0, cfg.SMA_CAP))

    # ── Liquidity penalty ─────────────────────────────────────────
    liq = feat["liquidity_sc"]
    if liq < 0.5:
        penalties["liquidity"] = float(np.clip(cfg.LIQ_CAP * (1.0 - liq * 2.0), 0.0, cfg.LIQ_CAP))

    # ── Gap penalty ───────────────────────────────────────────────
    prev_c = float(c.iloc[-1]); prev_o = float(df["open"].iloc[-1]) if "open" in df.columns else prev_c
    if prev_c > 0 and atr_v > 0 and len(c) >= 2:
        gap = prev_o - float(c.iloc[-2])
        # FIX-2: g_hist was using (h.shift(1) - c.shift(1)).abs() which measures
        # the upper shadow of each bar, not the overnight gap distribution.
        # Correct reference: (open_t - close_{t-1}) for each historical bar,
        # excluding today so the current gap is not in its own reference window.
        if "open" in df.columns:
            _o_hist = df["open"].iloc[:-1]          # historical opens, exclude today
            _c_lag  = c.shift(1).iloc[:-1]           # prior-day closes aligned to _o_hist
            g_hist  = ((_o_hist - _c_lag).abs() / (atr.iloc[:-1] + 1e-9)).dropna().tail(60)
        else:
            # No open column: fall back to close-to-close move as proxy
            g_hist  = (c.diff().abs() / (atr + 1e-9)).iloc[:-1].dropna().tail(60)
        g_p90  = float(g_hist.quantile(0.90)) if len(g_hist) >= 20 else 2.0
        g_atr  = abs(gap) / (atr_v + 1e-9)
        if g_atr > g_p90:
            penalties["gap"] = float(np.clip(8.0 * np.tanh((g_atr - g_p90) / (g_p90 + 1e-9)),
                                              0.0, cfg.GAP_CAP))

    # ── Already broken out (extended AND high vol — move already done) ───
    # Only penalise when price is ABOVE the base high by more than 0.5 ATR
    # AND volume is extreme. Being AT the pivot with vol = genuine breakout to buy.
    setup = sig["setup_type"]

    # Below-SMA200 hard gate for Breakout / Coiling setups.
    # Pullback and Reversal setups below SMA200 are legitimate mean-reversion
    # plays so they are intentionally excluded from this gate.
    _sma200_v = float(ind["sma200"])
    if setup in ("Breakout", "Coiling") and ltp < _sma200_v:
        _below_atr = (_sma200_v - ltp) / (atr_v + 1e-9)
        penalties["below_sma200_bo"] = float(np.clip(5.0 + 5.0 * np.tanh(_below_atr / 2.0), 0.0, cfg.SMA_CAP))
    if setup == "Breakout" and ltp >= feat["base_hi"] + 0.5 * atr_v:
        t1v     = float(v.iloc[-1])
        vrank   = float((v.iloc[:-1] <= t1v).mean())
        if vrank >= 0.82:   # tightened: catches moves at 82nd+ vol percentile
            z = (vrank - 0.90) / 0.10
            penalties["already_bo"] = float(np.clip(6.0 * np.tanh(z * 2.0), 0.0, cfg.ALREADY_BO_CAP))

    # ── Stale breakout: move already happened N days ago ─────────────────
    # If price crossed above base_hi more than 3 bars ago, this is NOT a fresh
    # entry — it is a post-move consolidation. Apply a scaling penalty.
    _dsb = sig.get("days_since_break", 0)
    if setup in ("Breakout", "Coiling") and _dsb >= 4:
        _stale_excess = (_dsb - 3) / 5.0   # 0 at 3 days, 1.0 at 8 days
        penalties["stale_breakout"] = float(np.clip(12.0 * np.tanh(_stale_excess), 0.0, 15.0))

    # ── Overextended breakout (above ext_p90) ─────────────────────
    if setup == "Breakout" and sig["breakout_ext"] > sig["ext_p90"]:
        excess = (sig["breakout_ext"] - sig["ext_p90"]) / max(sig["ext_p90"] - sig["ext_p10"], 0.5)
        penalties["overextended"] = float(np.clip(10.0 * np.tanh(excess), 0.0, cfg.EXT_CAP))

    # ── Stability kill (very choppy) ──────────────────────────────
    # Threshold derived from the stock's own stability distribution:
    # P20 of rolling 20-day positive-close fraction over last 60 bars.
    # Exclude today's return so a big green day doesn't raise the P20 threshold
    # that decides whether today is "too choppy".
    stab = feat["stability"]
    if n >= 40:
        _stab_ser = c.iloc[:-1].pct_change().rolling(20).apply(
            lambda x: (x > 0).sum() / max(len(x.dropna()), 1), raw=False
        ).dropna()
        _stab_p20 = float(_stab_ser.quantile(0.20)) if len(_stab_ser) >= 10 else 0.35
    else:
        _stab_p20 = 0.35
    # RSI threshold: stock's own P35 (the same bar used for reversal detection)
    # Bug 6 fix: exclude today's RSI so an extreme current reading doesn't
    # raise/lower the threshold that judges it.
    _rsi_all_pen = _rsi_wilder(c, rsi_period)
    _rsi_p35     = float(_rsi_all_pen.iloc[:-1].tail(cfg.PERCENTILE_WINDOW_PERSHORT).quantile(0.35)) if len(_rsi_all_pen) >= cfg.MIN_BARS_RSI else cfg.RSI_OS_FALLBACK
    if stab < _stab_p20 and (feat["rsi_v"] < _rsi_p35 or feat["rsi_v"] <= feat["rsi_prev"]):
        _excess = (_stab_p20 - stab) / max(_stab_p20, 1e-9)
        penalties["stability"] = float(np.clip(15.0 * _excess, 0.0, cfg.STAB_CAP))

    # ── VIX penalty (continuous z-score map) ──────────────────────
    vix_v   = market_ctx.get("vix_level")
    vix_med = market_ctx.get("vix_median", 14.5)
    vix_sig = market_ctx.get("vix_sigma",  4.5)
    vix_fall = market_ctx.get("vix_falling", True)
    if vix_v is not None and vix_med is not None and vix_sig is not None:
        # Only compute z-score when we have real distribution parameters from live data.
        vix_z   = (vix_v - vix_med) / (max(vix_sig, 0.5) + 1e-9)
        # vix_adj: positive = penalty (high VIX), negative = bonus (low VIX).
        # Stored directly so total -= vix_adj is consistent with all other keys.
        vix_adj = float(np.clip(cfg.VIX_TANH_SCALE * np.tanh(vix_z), -cfg.VIX_BONUS_CAP, -cfg.VIX_PENALTY_FLOOR))
        if not vix_fall:
            # FIX-1: When VIX is rising, add an incremental penalty on top of the
            # z-score-based adjustment — but do NOT collapse a bonus to zero first.
            # The old code used np.clip(..., 0.0, ...) which simultaneously wiped
            # any low-VIX bonus AND added the extra penalty (double-hit of up to
            # VIX_BONUS_CAP + VIX_PENALTY_FLOOR pts).  Correct approach: add the
            # extra penalty independently and re-clamp within the full valid range.
            extra   = cfg.VIX_FALL_EXTRA * abs(float(np.tanh(vix_z)))
            vix_adj = float(np.clip(vix_adj + extra, -cfg.VIX_BONUS_CAP, -cfg.VIX_PENALTY_FLOOR))
        penalties["vix"] = vix_adj   # positive = penalty, negative = bonus; applied via total -= p_val
    elif vix_v is not None and not vix_fall:
        # Have VIX level but no history — only penalise if it's clearly rising
        penalties["vix"] = 3.0

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
        penalties["breadth"] = cfg.BREADTH_FALLBACK_PENALTY

    # ── Regime penalty for Breakouts and Coiling setups ──────────
    if setup in ("Breakout", "Coiling"):
        bb_cs  = feat["bb_cs"] if feat["bb_cs"] is not None else 0.5
        vdu_cs = feat["vdu_cs"] if feat["vdu_cs"] is not None else 0.5
        nf     = (bb_cs + vdu_cs) / 2.0
        if regime == "BEAR":
            penalties["regime_bo"] = (1.0 - nf) * cfg.REGIME_BEAR_FACTOR
        elif regime == "CHOP":
            penalties["regime_bo"] = (1.0 - nf) * cfg.REGIME_CHOP_FACTOR

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
    if setup in ("Breakout", "Coiling"):
        bo_frac = STATE.get("_bo_saturation_frac", 0.0)
        if bo_frac > cfg.BO_SATURATION_FLOOR:
            excess = (bo_frac - cfg.BO_SATURATION_FLOOR) / (
                1.0 - cfg.BO_SATURATION_FLOOR + 1e-9
            )
            penalties["bo_saturation"] = float(np.clip(
                cfg.BO_SATURATION_DISCOUNT * 100.0 * excess,
                0.0,
                cfg.GAP_CAP   # cap at 15 pts — same as gap/RSI caps; 30-pt cap was too large
            ))

    # ── Weekly trend alignment penalty ───────────────────────────────
    # A Breakout or Coiling setup against the weekly trend is lower probability.
    if setup in ("Breakout", "Coiling") and cfg.WEEKLY_TREND_CHECK:
        _wtu = ind.get("weekly_trend_up", True)
        if not _wtu:
            penalties["weekly_countertrend"] = cfg.WEEKLY_TREND_PENALTY

    # ── Intraday market circuit breaker note ─────────────────────────
    # If the circuit breaker fired (Nifty down ≥2%) any stock that would have
    # been Breakout is already reclassified to Pullback in compute_signals.
    # We add a small residual penalty here for stocks near the pivot so they
    # don't surface near the top of the screener on a bad market day.
    _nifty_chg = float(market_ctx.get("nifty_intraday_chg") or 0.0)
    if _nifty_chg <= cfg.NIFTY_INTRADAY_KILL:
        penalties["market_circuit_breaker"] = cfg.MARKET_CB_PENALTY

    return {"penalties": penalties,
            "total_penalty": sum(
                # vix and breadth store signed contributions: positive=penalty, negative=bonus.
                # All other keys: stored as positive penalty magnitudes (no bonus possible).
                # For total_penalty display, count only the penalty portion of each key.
                max(0.0, p)
                for k, p in penalties.items()
            )}


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
    _regime = feat["regime"]

    # ────────────────────────────────────────────────────────────
    # Signal strength × coverage  (prevents sparse-signal stocks)
    # ────────────────────────────────────────────────────────────
    c, h, l, v = ind["c"], ind["h"], ind["l"], ind["v"]
    atr = ind["atr"]
    atr_v  = feat["atr_v"];   atr_pct = feat["atr_pct"]
    _e9_v  = feat["e9_v"];    e20_v   = feat["e20_v"];   e50_v = feat["e50_v"]
    rsi_v  = feat["rsi_v"];   rsi_p   = feat["rsi_prev"]
    _vol_mu = feat["vol_mu"]; stability = feat["stability"]
    liquidity_sc = feat["liquidity_sc"]
    n = len(c)

    if setup == "Reversal":
        # Reversal uses its own sub-scoring
        rsi_all  = _rsi_wilder(c, rsi_period)
        # Bug 7 fix: exclude today's RSI from its own reference band so an
        # extreme oversold reading doesn't compress the range that grades it.
        rp90     = float(rsi_all.iloc[:-1].tail(60).quantile(0.90)) if len(rsi_all) >= 20 else 70.0
        rp10     = float(rsi_all.iloc[:-1].tail(60).quantile(0.10)) if len(rsi_all) >= 20 else 25.0
        r_range  = max(rp90 - rp10, 10.0)
        rev_rsi  = float(np.clip((rp90 - rsi_v) / r_range, 0.0, 1.0))
        rev_coil = float(np.clip(sig["coil_sc"], 0.0, 1.0))
        rev_prox = sig["prox_sc"]
        # FIX-8: vol_quiet_pct is HIGH when today's volume is LOW (quiet).
        # Reversals require a volume SURGE (panic climax), so the contribution
        # to the reversal sub-score must be inverted: high vol → high rev_spr.
        # The old code used vol_quiet_pct directly, which penalised genuine
        # panic-volume reversal days and rewarded low-volume ones.
        rev_spr  = 1.0 - sig["vol_quiet_pct"]
        rev_vol  = float((v.iloc[:-1] <= float(v.iloc[-1])).mean()) if n > 1 else 0.5
        # Washout depth: normalise by this stock's own historical washout distribution
        if n >= 20:
            _wd_hist = ((h.iloc[:-1].rolling(10).max() - c.iloc[:-1]) /
                        (atr.iloc[:-2] + 1e-9)).dropna()
            _wd_hist = _wd_hist[_wd_hist > 0]
            _wd_p25  = float(_wd_hist.quantile(0.25)) if len(_wd_hist) >= 10 else 1.0
            _wd_p75  = float(_wd_hist.quantile(0.75)) if len(_wd_hist) >= 10 else 4.0
        else:
            _wd_p25, _wd_p75 = 1.0, 4.0
        _wd_range = max(_wd_p75 - _wd_p25, 0.5)
        rev_wash = float(np.clip((sig["washout_depth"] - _wd_p25) / _wd_range, 0.0, 1.0))
        # Close tail: P20 of historical close-position-in-range as lower bound
        if n >= 20:
            _hl_r    = (h.iloc[:-1] - l.iloc[:-1]).replace(0, np.nan)
            _cp_hist = ((c.iloc[:-1] - l.iloc[:-1]) / _hl_r).dropna()
            _cp_p20  = float(_cp_hist.quantile(0.20)) if len(_cp_hist) >= 10 else 0.25
        else:
            _cp_p20 = 0.25
        rev_tail = float(np.clip((sig["t1_close_pos"] - _cp_p20) / max(1.0 - _cp_p20, 0.1), 0.0, 1.0))
        # Weights from ScoreConfig — sum = 1.0 (previously hardcoded, now configurable).
        sub_scores = [rev_rsi  * cfg.REV_W_RSI,  rev_coil * cfg.REV_W_COIL,
                      rev_prox * cfg.REV_W_PROX,  rev_spr  * cfg.REV_W_SPR,
                      rev_vol  * cfg.REV_W_VOL,   rev_wash * cfg.REV_W_WASH,
                      rev_tail * cfg.REV_W_TAIL]
        # Coverage counts how many underlying inputs were computable (non-None),
        # not whether their weighted product is > 0 (which silently undercounts
        # valid signals that happen to contribute 0 pts).
        _rev_inputs = [rev_rsi, rev_coil, rev_prox, rev_spr, rev_vol, rev_wash, rev_tail]
        raw_coverage = sum(1 for s in _rev_inputs if s is not None and pd.notna(s)) / max(len(_rev_inputs), 1)
        coverage   = raw_coverage   # alias for any downstream use
        signal_str = sum(sub_scores)
    else:
        # Breakout + Pullback + Coiling unified pipeline
        # ── Adaptive volume weighting ─────────────────────────────
        # When vol_confirm is False (pre-move / Coiling), volume hasn't fired yet.
        # Penalising the stock for this makes the screener rank pre-move setups LOW
        # and only surface them AFTER the move starts.  Instead we reduce vol weight
        # and redistribute to coil + VCP — the signals that actually predict the move.
        _vol_confirmed = sig.get("vol_confirm", False)
        if setup == "Breakout":
            _vol_w = cfg.W_VOLUME                          # full weight — vol is required
            _coil_w = cfg.W_COIL
            _vcp_w  = cfg.W_VCP
        elif setup == "Coiling":
            # Pre-move: shrink vol weight, boost coil + VCP
            _vol_w  = cfg.W_VOLUME * cfg.VOL_WEIGHT_UNCONFIRMED_SCALE
            _coil_w = cfg.W_COIL  + cfg.COIL_WEIGHT_BOOST
            _vcp_w  = cfg.W_VCP   + cfg.VCP_WEIGHT_BOOST
        else:
            # PB-EMA / PB-Dry / PB-Deep / Base / Reversal: vol_signal = dryup (inverted)
            _vol_w  = cfg.W_VOLUME
            _coil_w = cfg.W_COIL
            _vcp_w  = cfg.W_VCP

        vol_sig = (feat["vol_signal"] if setup == "Breakout"
                   else (1.0 - feat["vol_signal"]) if setup == "Coiling"  # dryup desired pre-move: low vol = coiling energy
                   else (1.0 - feat["vol_signal"]))                        # dryup desired for all pullback variants

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
            "rs":        cfg.W_RS,       "rs_sect":   cfg.W_RS_SECT,
            "momentum":  cfg.W_MOMENTUM, "volume":    _vol_w,
            "coil":      _coil_w,        "ma":        cfg.W_MA,
            "proximity": cfg.W_PROXIMITY,"vcp":       _vcp_w,
            "darvas":    cfg.W_DARVAS,   "micro":     cfg.W_MICROSTRUCTURE,
        }
        # ── OI as a weighted signal for F&O universe ──────────────────
        # For F&O stocks OI data is meaningful and available; for NSE All
        # most stocks have zero OI so the signal would be noise.
        # When active, we steal a small weight from "micro" to keep weights
        # normalised — OI weight (0.05) is taken proportionally from micro.
        _is_fno = STATE.get("_last_universe") == "F&O Stocks"
        _oi_sig = sig.get("oi_sc", 0.0)
        if _is_fno and _oi_sig > 0:
            _oi_w = 0.05
            raw_sigs["oi"] = _oi_sig
            weights["oi"]   = _oi_w
            # Reduce micro weight to keep total normalised
            weights["micro"] = max(0.0, cfg.W_MICROSTRUCTURE - _oi_w)
        # ── Signal validity gates ─────────────────────────────────
        # Each signal is set to None when it genuinely lacks enough
        # data to be meaningful, rather than silently returning a
        # fallback float.  This makes SignalPersist / SIGCOV actually
        # reflect data quality instead of always showing 100%.
        #
        # Every threshold is derived from the signal's own minimum
        # window requirement — no arbitrary constants:
        #   rs        needs cross-sectional pass (cs5 or cs20) AND ≥23 bars
        #   rs_sect   needs known sector + live sector return data
        #   momentum  needs ≥PERCENTILE_WINDOW_SHORT bars of acc history
        #   volume    needs ≥8 bars for 5-bar slope to be meaningful
        #   coil      needs ≥40 bars for adaptive peak-gap detection
        #   ma        needs ≥EMA_SLOW bars for EMA50 to be settled
        #   proximity needs ≥30 bars for distance distribution
        #   vcp       needs ≥60 bars (hard floor in detect_vcp)
        #   darvas    needs ≥20 bars (hard floor in darvas_box_score)
        #   micro     needs at least one cross-sectional rank AND ≥25 bars
        _cs5_ok  = cs_state.get("cs_rs_5d",  {}).get(ticker) is not None
        _cs20_ok = cs_state.get("cs_rs_20d", {}).get(ticker) is not None
        # _sect_ok no longer gates rs_sect: compute_features returns neutral 0.5 when
        # sector/return data is unavailable, which is the correct contribution.
        # Nulling it here would zero-out the weight for unmapped stocks, deflating scores.
        _sect_ok = True
        _micro_cs_ok = any(
            cs_state.get(k, {}).get(ticker) is not None
            for k in ("cs_bb_squeeze", "cs_vol_dryup", "cs_clv_accum")
        )
        if not (_cs5_ok or _cs20_ok) or n < 23:           raw_sigs["rs"]        = None
        # rs_sect always included (neutral 0.5 when data missing — see compute_features)
        if n < cfg.PERCENTILE_WINDOW_SHORT:                raw_sigs["momentum"]  = None
        if n < 8:                                          raw_sigs["volume"]    = None
        if n < 40:                                         raw_sigs["coil"]      = None
        if n < cfg.EMA_SLOW:                               raw_sigs["ma"]        = None
        if n < 30:                                         raw_sigs["proximity"] = None
        if n < 60:                                         raw_sigs["vcp"]       = None
        if n < 20:                                         raw_sigs["darvas"]    = None
        if not _micro_cs_ok or n < 25:                     raw_sigs["micro"]     = None

        valid_sigs   = {k: v for k, v in raw_sigs.items() if v is not None and pd.notna(v)}
        coverage     = len(valid_sigs) / max(len(raw_sigs), 1)
        wt_sum       = sum(weights[k] for k in valid_sigs)
        signal_str   = sum(raw_sigs[k] * weights[k] for k in valid_sigs) / (wt_sum + 1e-9)

        # ── Interaction term: RS × Volume × Proximity compounding ──────
        # A simple weighted sum treats these as independent.  When all three
        # fire strongly together the actual edge is multiplicative, not additive.
        # Only activates when all three exceed INTERACTION_FLOOR (default 0.60)
        # to avoid rewarding marginal co-occurrence.
        _fl  = cfg.INTERACTION_FLOOR
        # FIX-5: Use explicit None checks instead of `or 0.0`.
        # `raw_sigs.get("rs") or 0.0` coerces both None (missing signal) and a
        # genuine 0.0 float to the same 0.0, permanently disabling the interaction
        # boost for new/short-history stocks whose rs=None was treated identically
        # to rs=0.0.  The weighted-sum path above correctly excludes None via
        # valid_sigs; the interaction guard must match that same contract.
        _rs_raw  = raw_sigs.get("rs")
        _vol_raw = raw_sigs.get("volume")
        _prx_raw = raw_sigs.get("proximity")
        if (_rs_raw  is not None and pd.notna(_rs_raw)  and
            _vol_raw is not None and pd.notna(_vol_raw) and
            _prx_raw is not None and pd.notna(_prx_raw) and
            _rs_raw > _fl and _vol_raw > _fl and _prx_raw > _fl):
            _norm = (1.0 - _fl) ** 3
            _interaction = ((_rs_raw - _fl) * (_vol_raw - _fl) * (_prx_raw - _fl)) / (_norm + 1e-9)
            signal_str *= (1.0 + float(np.clip(_interaction, 0.0, cfg.INTERACTION_BOOST_MAX)))

    # FIX-3: Keep raw_coverage for the SignalPersist output field so traders can
    # actually see when a stock has sparse signal coverage (e.g. 2/10 valid signals
    # shows as 0.20, not artificially inflated to 0.40 by the floor).
    # The coverage floor is still respected internally for score computation — but
    # signal_str is already a coverage-weighted average so it naturally degrades
    # with fewer valid signals; the floor here was only affecting the display field.
    raw_coverage = coverage
    coverage_floored = max(coverage, cfg.COVERAGE_FLOOR)  # kept for any future internal use
    base_score = signal_str * 100.0

    # ── CoilingScore — pure pre-move compression quality (0–100) ─────────
    # Computed from BB squeeze, vol dryup, CLV, VCP, spread compression, VC.
    # Independent of vol_confirm: scores HIGH before the move, not after.
    # Used as a standalone sort/filter in the screener "Coiling" view.
    coiling_score = _compute_coiling_score(
        feat["bb_cs"], feat["vdu_cs"], feat["clv_cs"],
        feat.get("vcp_cs"), feat["sc_feature"], feat["vc_feature"],
    )

    # ── Multi-day streak persistence bonus ───────────────────────────────
    # If this stock has been in the top COIL_PERSIST_PRANK tier of BB squeeze
    # AND vol dryup for COIL_PERSIST_MIN_DAYS+ consecutive days, reward it.
    # This gives the screener "memory" — a stock coiling for a week should rank
    # higher than one that just entered compression today.
    _streak = cs_state.get("coil_streak_days", {}).get(ticker, 0)

    _days_since_break = sig.get("days_since_break", 0)   # computed in compute_signals
    streak_bonus = 0.0
    if _streak >= cfg.COIL_PERSIST_MIN_DAYS:
        # Bonus grows with streak length, capped at COIL_PERSIST_BONUS_CAP
        streak_bonus = float(np.clip(
            cfg.COIL_PERSIST_BONUS_CAP * np.tanh((_streak - cfg.COIL_PERSIST_MIN_DAYS + 1) / 4.0),
            0.0, cfg.COIL_PERSIST_BONUS_CAP
        ))
    # ── Bonuses ───────────────────────────────────────────────────
    # ATR expansion onset fires at the START of ATR expansion — the move has
    # already begun.  Scale it down heavily when vol_confirm is absent so it
    # only contributes meaningfully as a confirmation signal, not a pre-move
    # coil signal.  When vol is confirmed it acts as a momentum accelerator.
    _atr_exp_scale = (1.0 if sig.get("vol_confirm") else cfg.ATR_EXP_NOCONFIRM_SCALE)
    _atr_exp_bonus = feat["atr_exp_feature"] * cfg.ATR_EXP_CAP * _atr_exp_scale

    bonus_raw = (
        sig["sweep_sc"]    * cfg.SWEEP_CAP    +
        sig["oi_sc"]       * cfg.OI_CAP       +
        feat["uv_feature"] * cfg.UV_CAP       +
        feat["cpr_feature"]* cfg.CPR_CAP      +
        feat["sc_feature"] * cfg.SC_CAP       +
        _atr_exp_bonus                         +
        sig["vcve_sc"]     * cfg.VCVE_CAP     +
        streak_bonus                            # multi-day coil persistence reward
    )
    # Persistence multiplier (compression in last 3 bars)
    # Persistence multiplier: fraction of last 3 bars compressed below P40,
    # smoothed via tanh rather than a rigid 0.25-step function.
    persist = 1.0
    vc_ser  = (ind["atr5"] / ind["atr20"].replace(0, np.nan)).dropna()
    if n >= 6 and len(vc_ser) >= 10:
        vc3    = vc_ser.iloc[-4:-1].dropna()
        vc_p40 = float(vc_ser.quantile(0.40))
        _vc_p20 = float(vc_ser.quantile(0.20))
        _comp_frac = float((vc3 < vc_p40).mean())   # 0, 0.33, 0.67, or 1.0
        # tanh maps 0→0.5, full-compression→~1.0; scaled to [0.5, 1.0]
        persist = float(np.clip(0.5 + 0.5 * np.tanh(_comp_frac * 2.0), 0.5, 1.0))
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
    # compression_score is the arithmetic mean of five correlated percentile-rank
    # signals (BB squeeze, vol dryup, CLV, VCP, VC ratio).  It is *not* a
    # calibrated probability — the signals share common drivers (vol contraction,
    # price compression) so they are far from independent.  Treat values above 0.7
    # as qualitative alignment indicators only.  To convert this into a genuine
    # forward-return probability, fit a logistic regression or isotonic calibration
    # against actual FORWARD_RETURN_DAYS outcomes once ≥100 calibration rows exist.
    compression_score = float(np.mean([bb_n, vdu_n, clv_n, vcp_n, vc_n]))

    # ── VolClimax flag ────────────────────────────────────────────
    # All three thresholds are derived from each stock's own distribution
    # via percentile ranks already computed in compute_features:
    #   vol_ratio P-rank vs VOL_CLIMAX_PRANK  (e.g. top 10% of own history)
    #   RSI P-rank vs VOL_CLIMAX_RSI_PRANK    (e.g. top 30% of own history)
    #   price extended beyond its own ext_p90 percentile
    # No fixed price/RSI levels — purely stock-relative.
    _vol_prank   = feat.get("vol_prank",  0.0)   # percentile rank of today's vol_ratio
    _rsi_prank   = feat.get("rsi_prank",  0.0)   # percentile rank of today's RSI value
    _ext_prank   = feat.get("ext_prank",  0.0)   # percentile rank of price extension
    vol_climax   = bool(
        _vol_prank >= cfg.VOL_CLIMAX_PRANK and
        _rsi_prank >= cfg.VOL_CLIMAX_RSI_PRANK and
        _ext_prank >= cfg.VOL_CLIMAX_EXT_PRANK
    )
    # composite_rank uses emi_pct — EMI expressed as a percentile across the
    # scored universe — instead of raw EMI.  Raw EMI (score × ATR%) means the
    # most volatile stock always ranks first regardless of setup quality, because
    # ATR% is unbounded.  emi_pct maps EMI onto [0, 1] within the current batch,
    # making the rank driven by relative quality rather than absolute volatility.
    # _emi_universe is injected by get_screener() after it collects all rows;
    # during single-stock calls (explain endpoint) it falls back to raw EMI / 10.
    _emi_universe = cs_state.get("_emi_universe") or {}
    if _emi_universe:
        _all_emis = sorted(_emi_universe.values())
        emi_pct   = float(sum(1 for e in _all_emis if e <= emi) / max(len(_all_emis), 1))
    else:
        emi_pct   = float(np.clip(emi / 10.0, 0.0, 1.0))   # graceful fallback
    composite_rank = round(emi_pct * 0.70 + liquidity_sc * 0.20 + min(stability, 1.0) * 0.10, 4)

    # ── Horizon classification ────────────────────────────────────
    _atr_cv   = float(atr.iloc[-20:].std() / (atr.iloc[-20:].mean() + 1e-9)) if n >= 20 else 0.3
    _cv_scale = float(np.clip(1.0 + _atr_cv, 0.7, 1.5))
    # BUG 1 FIX: Swing 2-5D multiplier raised from 1.8→2.8 so that at typical ATR%
    # of 1.8-2.0%, the target reaches ~5%.  Previous 1.8× only hit 5% when ATR%≥2.78%,
    # systematically under-targeting mid-cap stocks.
    _tgt_mult = {
        "Imminent BO": round(0.6 * _cv_scale, 2),
        "Intraday":    round(0.6 * _cv_scale, 2),
        "Swing 2-5D":  round(2.8 * _cv_scale, 2),   # was 1.8 — now reliably reaches 5% at ATR% ~1.8%
        "Mid 5-14D":   round(3.2 * _cv_scale, 2),
        "Long 14-30D": round(4.5 * _cv_scale, 2),
    }

    rsi_all = _rsi_wilder(c, rsi_period)
    if n >= 40:
        # Bug 13 fix: exclude today's bar from the reference percentile distributions
        # so today's price action doesn't shift the breakout/pullback distance bands
        # that classify how far away the trigger is.
        bo_dh = ((h.iloc[:-1].rolling(20).max() - c.iloc[:-1]) / (atr.iloc[:-2] + 1e-9)).dropna().tail(60)
        p20_bo = float(np.percentile(bo_dh, 20)) if len(bo_dh) >= 10 else 0.25
        p50_bo = float(np.percentile(bo_dh, 50)) if len(bo_dh) >= 10 else 1.0
        p80_bo = float(np.percentile(bo_dh, 80)) if len(bo_dh) >= 10 else 3.0
        pb_dh  = ((ind["e20"].iloc[:-1] - c.iloc[:-1]) / (atr.iloc[:-2] + 1e-9)).clip(0).dropna().tail(60)
        p20_pb = float(np.percentile(pb_dh, 20)) if len(pb_dh) >= 10 else 0.3
        p50_pb = float(np.percentile(pb_dh, 50)) if len(pb_dh) >= 10 else 1.0
        p80_pb = float(np.percentile(pb_dh, 80)) if len(pb_dh) >= 10 else 2.5
    else:
        p20_bo, p50_bo, p80_bo = 0.25, 1.0, 3.0
        p20_pb, p50_pb, p80_pb = 0.30, 1.0, 2.5

    base_hi  = feat["base_hi"];   base_lo = feat["base_lo"]
    vol_ratio = feat["vol_ratio"]; t1_vr  = sig["t1_vol_ratio"]
    t1_cp    = sig["t1_close_pos"]; cdl_n  = sig["cdl_names"]
    _washout = sig["washout_depth"]

    vol_bo_t = sig["vol_bo_thresh"]
    rsi_p60  = float(rsi_all.iloc[:-1].tail(60).quantile(0.60)) if len(rsi_all) >= 20 else 55.0  # Bug 4 fix: exclude today's RSI from its own reference percentile
    rsi_p90  = float(rsi_all.iloc[:-1].tail(60).quantile(0.90)) if len(rsi_all) >= 20 else cfg.RSI_OB_FALLBACK  # BUG 12 FIX: used in Intraday horizon gate

    if setup == "Breakout":
        d_bo = (base_hi - ltp) / (atr_v + 1e-9)
        if d_bo <= p20_bo and day_vol >= vol_bo_t:
            horizon = "Imminent BO"
            hz_note = f"AT TRIGGER — vol {vol_ratio:.1f}× avg. Enter now or market open."
        elif d_bo <= 0.0 and vol_ratio >= 1.5 and rsi_v < rsi_p90:
            # BUG 12 FIX: Was rsi_v < rsi_p60 — a stock actively breaking out with 1.5×
            # volume almost always has RSI > P60, so this branch almost never fired.
            # Changed to rsi_v < rsi_p90 (not yet severely overbought) to correctly
            # classify fresh intraday breakouts.
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
    elif setup == "Coiling":
        # Pre-move: price is at the pivot but volume hasn't confirmed yet.
        # Horizon tells the trader exactly what to watch for.
        d_bo = (base_hi - ltp) / (atr_v + 1e-9)
        _streak_days = cs_state.get("coil_streak_days", {}).get(ticker, 0)
        _streak_str  = f" ({_streak_days}d streak)" if _streak_days >= cfg.COIL_PERSIST_MIN_DAYS else ""
        # Escalate to "Imminent BO" only when BOTH the score is high AND the streak
        # confirms multiple days of compression — single-bar high scores are noise.
        _coil_escalate = (coiling_score >= 75 and _streak_days >= cfg.COIL_PERSIST_MIN_DAYS)
        if d_bo <= p20_bo or _coil_escalate:
            horizon = "Imminent BO"
            hz_note = (f"AT PIVOT — tight base{_streak_str}. CompressionScore {coiling_score:.0f}. "
                       f"Set alert above {base_hi:.2f}. Entry on vol surge >{round(sig['vol_bo_thresh'],0):.0f} shares.")
        elif d_bo <= p50_bo:
            horizon = "Swing 2-5D"
            hz_note = (f"{d_bo:.2f} ATR below trigger{_streak_str}. "
                       f"Base tightening — watch for vol expansion day. CompressionScore {coiling_score:.0f}.")
        else:
            horizon = "Mid 5-14D"
            hz_note = (f"Compression building{_streak_str}. "
                       f"CompressionScore {coiling_score:.0f} — add to watchlist, do not enter yet.")
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
        # PB-EMA / PB-Dry / PB-Deep / Base — all share the same pullback horizon logic
        rsi_turn  = rsi_v > rsi_p
        pb_d_atr  = (e20_v - ltp) / (atr_v + 1e-9)
        # Volume dryup threshold: P40 of this stock's own vol-ratio history
        _vr_hist  = (v.iloc[:-1] / (v.iloc[:-1].rolling(20).mean() + 1e-9)).dropna()
        _vr_p40   = float(_vr_hist.quantile(0.40)) if len(_vr_hist) >= 20 else 0.8
        if pb_d_atr <= p20_pb and rsi_turn and vol_ratio <= _vr_p40:
            horizon = "Intraday"
            hz_note = f"EMA20 support + RSI turning ({rsi_v:.0f}↑). Vol dry. Buy near {e20_v:.1f}."
        elif pb_d_atr <= p20_pb and rsi_turn and sig["raw_cdl"] >= 2:
            horizon = "Imminent BO"
            hz_note = f"Reversal candle at EMA. RSI {rsi_v:.0f}↑, pattern: {', '.join(cdl_n) or 'none'}."
        elif pb_d_atr <= p50_pb and rsi_v >= float(rsi_all.iloc[:-1].tail(60).quantile(0.35)):  # Bug 9 fix: exclude today from RSI reference
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
    # Dynamic minimum risk: stock's own 20th-percentile daily range
    # normalised by ATR — derived entirely from this stock's behaviour,
    # no external constants.  Guards against inverted / degenerate stops.
    _hl_atr_ratio = ((ind["h"] - ind["l"]) / (atr.replace(0, np.nan))).dropna()
    _min_risk_atr = float(_hl_atr_ratio.tail(60).quantile(0.20)) \
                    if len(_hl_atr_ratio) >= 20 else 0.5
    _min_risk     = _min_risk_atr * atr_v   # absolute minimum distance entry→stop

    # Dynamic rolling support: lowest low over a window scaled to this
    # stock's own ATR coefficient-of-variation (more volatile → wider window).
    _atr_cv   = float(atr.tail(20).std() / (atr.tail(20).mean() + 1e-9)) \
                if n >= 20 else 0.3
    _sup_win  = int(np.clip(round(20 * (1.0 + _atr_cv)), 10, 60))
    _roll_sup = float(l.iloc[:-1].tail(_sup_win).min())  # empirical support from price history

    # Nearest swing low within 2 ATR below ltp (used as tighter stop anchor).
    # Falls back to base_lo / rolling support when no valid swing low found.
    _swing_low = None
    if n >= 10:
        try:
            _ll = l.values.astype(float)
            _sl_idx = _argrelmin(_ll, order=3)[0]
            # Keep only swing lows within the last 15 bars and within 2 ATR of ltp
            _sl_candidates = [
                _ll[i] for i in _sl_idx
                if i >= n - 15 and ltp - _ll[i] <= 2.0 * atr_v and _ll[i] < ltp
            ]
            if _sl_candidates:
                _swing_low = max(_sl_candidates)  # nearest (highest) swing low below ltp
        except Exception:
            _swing_low = None

    vc_ratio = feat["vc_ratio_now"]

    # Score-gate: below MIN_SCORE_FOR_LEVELS don't emit actionable levels —
    # they would be misleading precision on a weak setup.
    # BUG 10 FIX: Pre-move Coiling and PB setups legitimately score lower because
    # vol_confirm=False reduces the volume weight contribution.  A stock with
    # CoilingScore>=65 (strong compression) is actionable even at Score 30-38.
    # Add a CoilingScore override so the Coiling ★ view is actually useful.
    _coiling_override = (
        setup in ("Coiling", "PB-EMA", "PB-Dry") and
        coiling_score >= 65 and
        total >= 28   # hard floor — never emit levels for truly weak setups
    )
    _levels_ok = total >= cfg.MIN_SCORE_FOR_LEVELS or _coiling_override

    if not _levels_ok:
        entry    = None
        tgt      = None
        stp      = None
        en_note  = f"Score {total:.0f} < {cfg.MIN_SCORE_FOR_LEVELS:.0f} — watchlist only, no entry yet"
        risk_raw = reward_raw = rr = move_pct = kelly = 0.0
    else:
        if setup in ("Breakout", "Coiling"):
            _buf   = atr_v * cfg.BO_ENTRY_BUFFER_ATR * max(0.5, vc_ratio)
            entry  = round(base_hi + _buf, 2)
            if setup == "Coiling":
                en_note = (f"Set limit-buy alert above {entry:.2f}. "
                           f"Do NOT enter until vol exceeds {round(sig['vol_bo_thresh'],0):.0f} shares.")
            else:
                en_note = (f"Buy above {entry:.2f}" if ltp < base_hi
                           else f"Breaking now — buy on close above {base_hi:.2f}")
            if sig.get("market_kill"):
                en_note += " | ⚠ Market down — confirm before entry"
            tgt   = round(entry + tgt_mult * atr_v, 2)
            # ── Resistance cap: if tgt lands within TGT_RESIST_ATR_BUFFER ATR of a
            # prior high (excluding the current base_hi itself), cap it below that high.
            # Resistance is derived from the stock's own 3-month price history —
            # no fixed % threshold.
            _hist_hi = float(h.iloc[:-1].tail(min(66, n - 1)).max()) if n > 1 else base_hi
            if (_hist_hi > base_hi and
                    tgt >= _hist_hi - cfg.TGT_RESIST_ATR_BUFFER * atr_v and
                    tgt <= _hist_hi + cfg.TGT_RESIST_ATR_BUFFER * atr_v):
                tgt = round(_hist_hi - cfg.TGT_RESIST_ATR_BUFFER * atr_v * 0.5, 2)
                en_note += f" | Resistance ~₹{_hist_hi:.0f}"
            # Prefer nearest swing low as stop anchor; fall back to box low / rolling support.
            if _swing_low is not None and (entry - _swing_low) <= 2.0 * atr_v:
                _stp_anchor = _swing_low
            else:
                _box_valid  = (base_hi - base_lo) >= atr_v
                _stp_anchor = base_lo if _box_valid else _roll_sup
            _stp_buf = atr_v * float(np.clip(vc_ratio, 0.3, 0.7))
            stp = round(_stp_anchor - _stp_buf, 2)

        elif setup == "Reversal":
            entry  = round(ltp, 2)
            en_note = f"Buy at open — reversal. RSI {rsi_v:.0f}. Stop below {float(l.iloc[-1]):.2f}"
            _stp_buf = atr_v * float(np.clip(vc_ratio, 0.15, 0.35))
            # Prefer swing low as stop anchor for tighter RR
            if _swing_low is not None:
                stp = round(_swing_low - _stp_buf, 2)
            else:
                stp = round(float(l.iloc[-1]) - _stp_buf, 2)
            tgt   = max(round(e20_v, 2), round(entry + tgt_mult * atr_v, 2))

        else:
            # PB-EMA / PB-Dry / PB-Deep / Base
            entry  = round(ltp, 2)
            # Warn when LTP is materially above EMA20 — user is not at support yet.
            # Threshold derived from TGT_RESIST_ATR_BUFFER (same ATR scale).
            _dist_above_ema = (ltp - e20_v) / (atr_v + 1e-9)
            if _dist_above_ema > cfg.PB_NOTE_EMA_ATR_WARN:
                en_note = (f"LTP ₹{ltp:.0f} is {_dist_above_ema:.1f} ATR above EMA20 "
                           f"({e20_v:.2f}) — not at support yet. Wait for pullback.")
            else:
                en_note = f"Buy near EMA20 ({e20_v:.2f}) on reversal candle"
            tgt_s  = round(float(h.tail(20).max()) * cfg.PB_TARGET_HIGH_FRAC, 2)
            tgt    = max(tgt_s, round(entry + tgt_mult * atr_v, 2))
            # Resistance cap for pullback targets too
            _hist_hi_pb = float(h.iloc[:-1].tail(min(66, n - 1)).max()) if n > 1 else tgt
            if (tgt >= _hist_hi_pb - cfg.TGT_RESIST_ATR_BUFFER * atr_v and
                    tgt <= _hist_hi_pb + cfg.TGT_RESIST_ATR_BUFFER * atr_v and
                    _hist_hi_pb > entry):
                tgt = round(_hist_hi_pb - cfg.TGT_RESIST_ATR_BUFFER * atr_v * 0.5, 2)
                en_note += f" | Resistance ~₹{_hist_hi_pb:.0f}"
            _stp_buf = atr_v * float(np.clip(vc_ratio, 0.3, 0.7))
            if _swing_low is not None and _swing_low < entry:
                stp = round(_swing_low - _stp_buf, 2)
            elif e50_v < entry:
                stp = round(e50_v - _stp_buf, 2)
            else:
                stp = round(_roll_sup - _stp_buf, 2)

        # ── Universal stop sanity guard ───────────────────────────
        if stp >= entry:
            stp = round(entry - _min_risk, 2)

        risk_raw   = max(entry - stp, _min_risk)
        reward_raw = max(tgt - entry, _min_risk)
        rr         = round(reward_raw / risk_raw, 2)
        move_pct   = round((tgt - entry) / entry * 100, 1) if entry != 0 else 0.0

    # ── Win-rate prior + Kelly (only meaningful when levels are valid) ──
    _wr_stock   = STATE["per_stock_winrate"].get(ticker)
    _wr_setup   = STATE.get("_setup_winrate", {}).get(setup)
    # Count how many calibration rows back this setup type (need KELLY_MIN_CALIB_ROWS
    # before the formula-derived win-rate is better than noise).
    _sw_counts  = STATE.get("_setup_winrate_counts", {})
    _calib_n    = _sw_counts.get(setup, 0)
    _stock_calib_n = STATE.get("_stock_calib_counts", {}).get(ticker, 0)
    _has_real_calib = (_stock_calib_n >= cfg.KELLY_MIN_CALIB_ROWS or
                       _calib_n >= cfg.KELLY_MIN_CALIB_ROWS)

    if _wr_stock is not None and _stock_calib_n >= cfg.KELLY_MIN_CALIB_ROWS:
        wr_prior = float(_wr_stock)
    elif _wr_setup is not None and _calib_n >= cfg.KELLY_MIN_CALIB_ROWS:
        _formula = float(np.clip(cfg.WR_BASE + cfg.WR_RS_COEF * feat["rs_combined"] + cfg.WR_STAB_COEF * stability, 0.30, 0.70))
        wr_prior = float(cfg.CALIB_ADAPT_ALPHA * _wr_setup + (1.0 - cfg.CALIB_ADAPT_ALPHA) * _formula)
    else:
        # Not enough calibration data — fall back to formula but mark it uncalibrated
        wr_prior = float(np.clip(cfg.WR_BASE + cfg.WR_RS_COEF * feat["rs_combined"] + cfg.WR_STAB_COEF * stability, 0.30, 0.70))

    if _levels_ok and _has_real_calib:
        kelly = round(float(np.clip(
            0.5 * (wr_prior * max(rr, 0.5) - (1.0 - wr_prior)) / (max(rr, 0.5) + 1e-9), 0.0, 0.25)), 3)
    else:
        # None signals "not enough calibration data" — frontend shows "—" instead of 0
        kelly = None

    # ── Reconstruct per-component scores (pts for explain endpoint) ─
    rs_pts      = round(feat["rs_combined"] * cfg.W_RS * 100, 1)
    rs_sect_pts = round(feat["rs_sect"] * cfg.W_RS_SECT * 100, 1)
    if setup == "Coiling":
        # Coiling: accumulation slope (no inversion), reduced weight matches scoring pipeline
        vol_pts = round(feat["vol_signal"] * cfg.W_VOLUME * cfg.VOL_WEIGHT_UNCONFIRMED_SCALE * 100, 1)
    elif setup == "Breakout" or setup == "Reversal":
        # vol surge desired, full weight
        vol_pts = round(feat["vol_signal"] * cfg.W_VOLUME * 100, 1)
    else:
        # PB-EMA / PB-Dry / PB-Deep / Base: vol dryup desired (inverted), full weight
        vol_pts = round((1.0 - feat["vol_signal"]) * cfg.W_VOLUME * 100, 1)
    # coil_pts is the DISPLAY column (0-10 scale, matching barFmt max=10).
    # The weight boost already affects score computation via signal_str above;
    # here we just normalise to 0-10 so the column never overflows its denominator.
    coil_pts    = round(sig["coil_sc"] * 10.0, 1)
    ma_pts      = round(feat["ma_feature"] * 10, 1)
    prox_pts    = round(sig["prox_sc"] * 10, 1)
    # vcp_pts is the DISPLAY column (0-10 scale, matching barFmt max=10).
    vcp_pts     = round(sig["vcp_sc"] * 10.0, 1)
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

    # SoftPenalty: sum of all deductions (positive p_val values) only.
    # vix and breadth can be negative (bonus) — exclude those from the display.
    total_pen = sum(p for p in pen["penalties"].values() if p > 0)

    # ── Action Signal (BUY / SELL / HOLD) ─────────────────────────────────
    # Based purely on scoring signals: CoilingScore, VCP, volume, RS,
    # proximity, setup type, and composite score. No penalty heuristics.
    #
    # BUY  = strong setup with compression + volume/RS alignment
    # SELL = exhaustion: extended + vol climax, or overextended + overbought RS
    # HOLD = building / watchlist — signals present but not aligned yet

    _cs   = coiling_score                          # 0-100 compression quality
    _vcp  = sig.get("vcp_sc", 0.0)                # 0-1 VCP pattern score
    _prox = sig.get("prox_sc", 0.0)               # 0-1 proximity to trigger
    _vol  = feat.get("vol_signal", 0.5)            # 0-1 volume signal
    _rs   = feat.get("rs_combined", 0.5)           # 0-1 cross-sectional RS
    _vc   = sig.get("vol_confirm", False)          # confirmed breakout volume
    _rsi  = feat["rsi_v"]
    _bext = feat["breakout_ext"]                   # ATR units above base high

    # ── SELL: exhaustion / distribution ───────────────────────────────────
    # Either VolClimax (RSI+vol+extension all in top decile of own history)
    # OR: RSI in top 15% of own history AND price overextended AND vol surge
    _rsi_p85 = float(_rsi_wilder(c, rsi_period).iloc[:-1].tail(60).quantile(0.85)) \
               if len(c) >= 20 else 72.0
    _is_sell = (
        vol_climax or
        (_rsi >= _rsi_p85 and _bext > sig.get("ext_p90", 0.5) and _vol > 0.65)
    )

    # ── BUY: compression + alignment ──────────────────────────────────────
    # Core idea: coiling energy built up + price near trigger + RS strong
    # Four pathways — all require genuine signal quality, not just marginal coil:
    #   A. Breakout confirmed: vol confirmed + near pivot + RS strong + score ok
    #   B. Pre-move coiling:  CoilingScore HIGH + VCP strong + vol genuinely DRY + RS + BB squeeze
    #      REQUIRES vdu_cs ≥ 0.55 so we never BUY into a vol expansion that already started.
    #   C. Pullback at EMA:   PB-EMA/PB-Dry + proximity tight + vol genuinely dry + RS + score
    #   D. Reversal:          washout + vol surge CONFIRMED + RS alive + score
    _vdu = feat.get("vdu_cs")           # cross-sectional vol-dryup rank (0-1, higher = drier)
    _vdu_v = float(_vdu) if _vdu is not None else 0.5
    _bb_cs_v = feat.get("bb_cs")
    _bb_v  = float(_bb_cs_v) if _bb_cs_v is not None else 0.5

    _is_buy = False
    if not _is_sell:
        if setup == "Breakout" and _vc and _prox >= 0.55 and _rs >= 0.50 and total >= 50:
            # A. Volume-confirmed breakout: proximity tight + RS above median + solid score
            _is_buy = True
        elif (setup == "Breakout" and _vc and
              _cs >= 55 and _vcp >= 0.45 and _prox >= 0.45 and _rs >= 0.45 and total >= 45):
            # A2. Confirmed breakout from a coiling base — slightly softer path but still filtered
            _is_buy = True
        elif (setup in ("Coiling", "Breakout") and
              _cs >= 65 and                   # strong compression quality (top ~35% of universe)
              _vcp >= 0.55 and                # VCP score well above median (5.5/10 in display pts)
              _prox >= 0.50 and               # price within typical approach distance of pivot
              _rs >= 0.50 and                 # RS above median vs universe
              _vdu_v >= 0.55 and              # volume is genuinely drying up (above median dryup rank)
              _bb_v >= 0.55 and               # Bollinger Bands are genuinely squeezed
              not _vc):                       # vol NOT yet confirmed — pure pre-move watchlist BUY
            # B. Pre-move coil/VCP: genuine compression + vol dryup + RS + near pivot
            _is_buy = True
        elif (setup in ("PB-EMA", "PB-Dry") and
              _prox >= 0.55 and               # price tight against EMA support
              _rs >= 0.45 and                 # RS holding up
              _vdu_v >= 0.55 and              # volume genuinely drying on the pullback
              _vol <= 0.45 and                # raw vol_signal low (truly dry, not just below average)
              total >= 48):                   # setup score must be solid
            # C. Pullback at support: tight price + genuinely dry vol + RS decent + score ok
            _is_buy = True
        elif (setup == "Reversal" and _rs >= 0.40 and total >= 45 and
              _vol > 0.65 and _vc):           # vol surge must be CONFIRMED — genuine capitulation
            # D. Reversal: washout + confirmed vol surge + RS alive + score
            _is_buy = True

    if _is_sell:
        _action = "SELL"
        _rsi_p85_str = f"{_rsi_p85:.0f}"
        _action_reason = (
            "VolClimax — exhaustion top" if vol_climax
            else f"RSI {_rsi:.0f} ≥ P85({_rsi_p85_str}) + Extended + VolSurge"
        )
    elif _is_buy:
        _action = "BUY"
        _vol_note = (f"VolConf {feat.get('vol_ratio', 0):.1f}×" if _vc
                     else f"VolDry {_vdu_v:.2f}")
        _action_reason = (
            f"Score {total:.0f} | {setup} | Coil {_cs:.0f} | VCP {_vcp:.2f} | RS {_rs:.2f} | {_vol_note}"
        )
    else:
        _action = "HOLD"
        _action_reason = (
            f"Score {total:.0f} | {setup} — signals building, not aligned yet"
        )


    return {
        # ── Core ──────────────────────────────────────────────────
        "SetupType":   setup, "Score": round(total, 1),
        "EMI": emi, "CompositeRank": composite_rank,
        "Horizon": horizon, "HorizonNote": hz_note,
        "Entry": entry, "Target": tgt, "Stop": stp,
        "Risk": round(risk_raw, 1), "Reward": round(reward_raw, 1),
        "RR": rr, "KellyFrac": kelly, "Move%": move_pct, "EntryNote": en_note,
        # KellyFrac is None when calibration rows < KELLY_MIN_CALIB_ROWS (show '—' in UI)
        # ── Pre-move coiling quality (independent of vol_confirm) ─
        "CoilingScore": round(coiling_score, 1),
        "CoilStreakDays": _streak,
        "DaysSinceBreak": int(_days_since_break),   # 0=fresh breakout, >3=already moved
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
        "CompressionScore": round(compression_score, 3),  # renamed from BreakoutProb — not a calibrated probability
        "BreakoutProb": round(compression_score, 3),       # kept for backward-compat with older frontend builds
        "VolClimax":    vol_climax,                         # True = exhaustion warning, not a buy signal
        "SignalPersist": round(raw_coverage, 2),   # FIX-3: raw, un-floored coverage
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
        # Scale to 0-3 range using VCVE_CAP as the reference (same bonus bucket)
        "VolVelocity":   round(sig.get("vol_velocity", 0) * cfg.VCVE_CAP, 1),
        "RSDivergence":  round(feat["rs_div_pct"] * cfg.OI_CAP, 1),
        "CSRank5d":      round(cs_state.get("cs_rs_5d", {}).get(ticker, feat["rs_combined"]), 3),
        "AbsRS":         round(feat["abs_rs"], 3),
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
        "AboveSMA200":    ltp > ind["sma200"],
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
        # ── Improvement flags (visible in explain + debug) ─────────
        "BarConfirmed":      sig.get("bar_confirmed", True),
        "VolConfirmAllowed": sig.get("vol_confirm_allowed", True),
        "WeeklyTrendUp":     ind.get("weekly_trend_up", True),
        "MarketKill":        sig.get("market_kill", False),
        "LevelsValid":       _levels_ok,
        # ── Freshness / build-up (Sub-Problems 1 & 2) ──────────────────
        "VolBuildPct":       round(sig.get("vol_build_pct", 0.0), 3),
        "BreakoutMinsAgo":   sig.get("breakout_mins_ago"),   # None for non-Breakout rows
        # ── Action signal ──────────────────────────────────────────
        "Action":       _action,        # "BUY" | "SELL" | "HOLD"
        "ActionReason": _action_reason, # brief human-readable explanation
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

    rsi_period  = int(STATE.get("rsi_period", 14))   # FIX: was 7; STATE default is 14, fallback must match
    mkt         = STATE.get("mkt", {})
    cs_state    = {
        "cs_rs_5d":       STATE.get("cs_rs_5d",        {}),
        "cs_rs_20d":      STATE.get("cs_rs_20d",        {}),
        "cs_bb_squeeze":  STATE.get("cs_bb_squeeze",    {}),
        "cs_vol_dryup":   STATE.get("cs_vol_dryup",     {}),
        "cs_clv_accum":   STATE.get("cs_clv_accum",     {}),
        "cs_vcp":         STATE.get("cs_vcp",           {}),
        "coil_streak_days": STATE.get("coil_streak_days", {}),
        "rs_div_hist":    STATE.get("rs_div_hist",      {}),
        "breadth_cache":  STATE.get("breadth_cache"),
        "breadth_hist":   STATE.get("breadth_hist",     []),
        "_emi_universe":  STATE.get("_emi_universe",    {}),
    }
    # Add market context into cs_state for convenience
    cs_state.update({
        "sector_returns":      STATE.get("sector_returns", {}),
        "sector_returns_10d":  STATE.get("sector_returns_10d", {}),
        "nifty_intraday_chg":  STATE.get("mkt", {}).get("nifty_intraday_chg", 0.0),
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
    # Intraday Nifty change for circuit-breaker logic in compute_penalties
    market_ctx["nifty_intraday_chg"] = STATE.get("mkt", {}).get("nifty_intraday_chg", 0.0)

    vol_ma20 = float(df["volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df["volume"].mean())

    try:
        # Stage 1
        ind = compute_indicators(df, rsi_period)
        # Stage 2
        feat = compute_features(
            ind, ticker, ltp, day_vol_sc,
            day_hi, day_lo, day_o,
            nifty_r5, nifty_r20,
            market_ctx, cs_state, param_reg,
            rsi_period   # FIX: pass rsi_period so VolClimax RSI prank uses correct period
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

    except Exception as _score_exc:
        # Surface scoring errors into extraction log so they're visible in the UI
        # rather than silently producing missing rows.
        try:
            with STATE_LOCK:
                STATE["extraction_status"]["log"].append(
                    f"score_err {ticker}: {type(_score_exc).__name__}: {_score_exc}"
                )
        except Exception:
            pass
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
                    "ltp":        float(ltp),
                    "open":       float(ohlc.get("open",  ltp)),
                    "high":       float(ohlc.get("high",  ltp)),
                    "low":        float(ohlc.get("low",   ltp)),
                    # ohlc.close from Upstox is the previous session's close price.
                    # Store it so we can seed prev_close_cache without relying on
                    # iloc[-1] from historical data (which may not yet have today's bar).
                    "prev_close": float(ohlc.get("close", ltp)) if ohlc.get("close") else None,
                    "volume":     float(v["volume"]) if v.get("volume") else None,
                    "oi":         float(v["oi"])     if v.get("oi")     else None,
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
            with gzip.GzipFile(fileobj=_io.BytesIO(r.content)) as gz:
                data = _json.load(gz)
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
    # Kick off a sector-map refresh in parallel (no-op if recently done)
    _maybe_refresh_sector_map()
    # vix_median / vix_sigma start as None and are filled from live data below.
    # They are only used inside compute_penalties which guards on vix_v is not None,
    # so the fallback values here are never actually used in scoring — they only
    # show up in the market-context API response.  Setting them to None makes that
    # explicit rather than silently using made-up distribution parameters.
    out = dict(nifty_r5=None, nifty_r20=None, nifty_above_20dma=True, nifty_above_50dma=True,
               regime="BULL", vix_level=None, vix_falling=True, vix_median=None, vix_sigma=None,
               nifty_intraday_chg=0.0,
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
            # Intraday change: prefer live Nifty LTP from live_quotes_cache so the
            # circuit-breaker fires on today's actual move, not yesterday's daily return.
            # Falls back to day-over-day daily close when live quote is unavailable.
            _nifty_lq = STATE.get("live_quotes_cache", {}).get("NSE_INDEX|Nifty 50", {})
            _nifty_live_ltp = _nifty_lq.get("ltp") if _nifty_lq else None
            if _nifty_live_ltp is not None and len(c) >= 2:
                out["nifty_intraday_chg"] = float(_nifty_live_ltp / float(c.iloc[-2]) - 1)
            elif len(c) >= 2:
                out["nifty_intraday_chg"] = float(c.iloc[-1] / c.iloc[-2] - 1)
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
    _sect_errors: list = []

    def _fetch_sec(nt):
        nm, tk = nt
        try:
            s = yf.download(tk, period="60d", interval="1d", progress=False)
            if not s.empty:
                sc = s["Close"].squeeze()
                r5  = float(sc.iloc[-1] / sc.iloc[-6]  - 1) if len(sc) >= 6  else None
                r10 = float(sc.iloc[-1] / sc.iloc[-11] - 1) if len(sc) >= 11 else None
                return nm, r5, r10
            return nm, None, None  # empty response — no exception but no data
        except Exception as e:
            return nm, None, str(e)  # propagate error string for logging

    with ThreadPoolExecutor(max_workers=8) as ex:
        for result in ex.map(_fetch_sec, SECTOR_TICKERS.items()):
            nm, r5, r10_or_err = result
            if isinstance(r10_or_err, str):
                # r10 slot used to carry error string when r5 is None
                _sect_errors.append(f"{nm}: {r10_or_err}")
                continue
            if r5  is not None: sr5[nm]  = r5
            if r10_or_err is not None: sr10[nm] = r10_or_err

    if _sect_errors:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "sector_returns: %d/%d sector fetches failed — rs_sect will use neutral 0.5 "
            "for stocks in missing sectors. Failed: %s",
            len(_sect_errors), len(SECTOR_TICKERS),
            ", ".join(_sect_errors[:5]) + (" ..." if len(_sect_errors) > 5 else "")
        )
    if not sr5:
        import logging as _logging
        _logging.getLogger(__name__).error(
            "sector_returns is EMPTY — all sector index fetches failed. "
            "rs_sect will be 0.5 (neutral) for all stocks. "
            "Check Yahoo Finance connectivity for ^CNXIT, ^NSEBANK, etc."
        )
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
    """Compute all cross-sectional ranks from a snapshot of raw_data_cache.

    All heavy pandas work is done on a local snapshot (no lock held during
    computation).  Results are written back to STATE atomically under
    STATE_LOCK at the end, preventing races with the SSE stream and the
    screener route that read these dicts concurrently.
    """
    cfg = SCORE_CFG
    # Take a stable snapshot so we don't hold the lock during computation.
    with STATE_LOCK:
        cache = dict(st.get("raw_data_cache", {}))
    if len(cache) < 3:
        return

    r5r: dict = {}; r20r: dict = {}
    # Min-bar requirements are derived from the window definitions in ScoreConfig:
    #   5d RS  needs iloc[-6]  → requires RS_WINDOW_5D  + 1 = 6 bars
    #   20d RS needs iloc[-21] → requires RS_WINDOW_20D + 1 = 21 bars
    # Stocks with fewer bars get a NaN return and would either distort or
    # (via rankdata) unfairly rank near the middle of the universe.  We
    # exclude them from the rank computation entirely.
    # Fix: use iloc[-2] (yesterday close) not iloc[-1] (today) as the return base.
    # Each stock was being ranked against a universe that included its own today
    # return, inflating RS for any stock having a big green day.  Yesterday's
    # close measures pre-move relative strength cleanly.
    _MIN_5D  = SCORE_CFG.RS_WINDOW_5D  + 2   # 7  (iloc[-2] base + 5d back = iloc[-7])
    _MIN_20D = SCORE_CFG.RS_WINDOW_20D + 2   # 22 (iloc[-2] base + 20d back = iloc[-22])
    for sym, df in cache.items():
        c = df["close"]
        if len(c) >= _MIN_5D:  r5r[sym]  = float(c.iloc[-2] / c.iloc[-7]  - 1)
        if len(c) >= _MIN_20D: r20r[sym] = float(c.iloc[-2] / c.iloc[-22] - 1)
    cs_rs_5d  = _cdf_rank_dict(r5r)
    cs_rs_20d = _cdf_rank_dict(r20r)

    bbr: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < cfg.CS_MIN_BARS_BB: continue
            bw = (2.0 * c.rolling(20).std() / c.rolling(20).mean().replace(0, np.nan)).dropna()
            if len(bw) < 10: continue
            # Compare today's BB width against the historical distribution (excluding today)
            # so a breakout-day gap-up doesn't inflate the current BB width and
            # falsely rank the stock as "wide" when it is actually expanding.
            # FIX: We want a HIGH rank when bands are TIGHT (squeezed), so we measure
            # what fraction of historical widths are >= today's width.
            # Previously this used (bw.iloc[:-1] <= bw.iloc[-1]).mean() which gave a
            # HIGH rank for WIDE bands — the exact opposite of a squeeze signal.
            bbr[sym] = float((bw.iloc[:-1] >= float(bw.iloc[-1])).mean())
        except Exception:
            pass
    cs_bb_squeeze = _cdf_rank_dict(bbr)

    vdr: dict = {}
    for sym, df in cache.items():
        try:
            v = df["volume"].replace(0, np.nan).dropna()
            if len(v) < cfg.CS_MIN_BARS_VDR: continue
            ratio = float(v.tail(5).mean()) / (float(v.tail(20).mean()) + 1e-9)
            h_r   = (v.rolling(5).mean() / (v.rolling(20).mean() + 1e-9)).dropna()
            if len(h_r) >= 5:
                # Fraction of historical 5d/20d vol ratios that are >= today's ratio.
                # High rank = today's ratio is LOW relative to history = vol drying up = coiling.
                vdr[sym] = float((h_r >= ratio).mean())
            else:
                # FIX: Fallback must also express "low vol ratio = good squeeze".
                # clip(ratio, 0, 2) so 0 = max dryup, 2 = 2× avg vol.
                # Invert so low ratio → high score.  Clamp to [0, 1].
                vdr[sym] = float(np.clip(1.0 - ratio / 2.0, 0.0, 1.0))
        except Exception:
            pass
    cs_vol_dryup = _cdf_rank_dict(vdr)

    clvr: dict = {}
    for sym, df in cache.items():
        try:
            c = df["close"]; hh = df["high"]; ll = df["low"]; vv = df["volume"]
            if len(c) < cfg.CS_MIN_BARS_CLV: continue
            hl  = (hh - ll).replace(0, np.nan)
            clv = ((c - ll) - (hh - c)) / hl
            mf  = clv.fillna(0) * vv
            mfn = mf / vv.rolling(20).mean().replace(0, np.nan)
            rmf = mfn.rolling(20).sum().dropna()
            if len(rmf) < 5: continue
            clvr[sym] = float((rmf.iloc[:-1] <= float(rmf.iloc[-1])).mean())
        except Exception:
            pass
    cs_clv_accum = _cdf_rank_dict(clvr)

    vcpr: dict = {}
    for sym, df in cache.items():
        try:
            if len(df) < cfg.CS_MIN_BARS_VCP: continue
            c = df["close"]; hh = df["high"]; ll = df["low"]; vv = df["volume"]
            tr  = pd.concat([hh - ll, (hh - c.shift(1)).abs(), (ll - c.shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1 / SCORE_CFG.ATR_PERIOD, adjust=False).mean()
            vcpr[sym] = float(detect_vcp(c, hh, ll, vv, atr).get("vcp_score", 0.0))
        except Exception:
            pass
    cs_vcp = _cdf_rank_dict(vcpr)

    _ab = 0; _tot = 0
    new_breadth_cache = None; new_breadth_hist = None
    for sym, df in cache.items():
        try:
            c = df["close"]
            if len(c) < cfg.CS_MIN_BARS_BRD: continue
            e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            _tot += 1
            if float(c.iloc[-1]) > e20: _ab += 1
        except Exception:
            pass
    if _tot >= 10:
        br = _ab / _tot
        new_breadth_cache = br
        with STATE_LOCK:
            new_breadth_hist = (st.get("breadth_hist", []) + [br])[-200:]

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

    # ── Coil streak: count consecutive days each stock has been in top-tier ──
    # A stock in the top COIL_PERSIST_PRANK of BOTH BB squeeze AND vol dryup
    # increments its streak by 1.  Others reset to 0.
    # Read the previous streak from STATE under the lock, then update.
    with STATE_LOCK:
        prev_streaks = dict(st.get("coil_streak_days", {}))
    new_streaks: dict = {}
    _bb_thresh  = cfg.COIL_PERSIST_PRANK
    _vdu_thresh = cfg.COIL_PERSIST_PRANK
    _clv_thresh = 0.50   # CLV must be above median (any institutional accumulation)
    for sym in cache:
        bb_rank  = cs_bb_squeeze.get(sym, 0.0)
        vdu_rank = cs_vol_dryup.get(sym, 0.0)
        clv_rank = cs_clv_accum.get(sym, 0.0)
        # Streak requires: tight bands + vol drying up + at least median CLV accumulation
        in_top   = (bb_rank >= _bb_thresh) and (vdu_rank >= _vdu_thresh) and (clv_rank >= _clv_thresh)
        new_streaks[sym] = (prev_streaks.get(sym, 0) + 1) if in_top else 0

    # ── Persist streaks to SQLite so server restarts don't reset them ────────
    # Stored as a single JSON blob under a dedicated kv_store table.
    # Non-fatal: if DB write fails the in-memory streaks still work for this session.
    try:
        _db_conn = get_db()
        _db_conn.execute(
            "INSERT OR REPLACE INTO kv_store(key, value) VALUES ('coil_streak_days', ?)",
            (_json.dumps(new_streaks),)
        )
        _db_conn.commit()
    except Exception:
        pass

    # ── Atomic write: commit all computed results under STATE_LOCK ─────────
    with STATE_LOCK:
        st["cs_rs_5d"]        = cs_rs_5d
        st["cs_rs_20d"]       = cs_rs_20d
        st["cs_bb_squeeze"]   = cs_bb_squeeze
        st["cs_vol_dryup"]    = cs_vol_dryup
        st["cs_clv_accum"]    = cs_clv_accum
        st["cs_vcp"]          = cs_vcp
        st["coil_streak_days"]= new_streaks
        if new_breadth_cache is not None:
            st["breadth_cache"] = new_breadth_cache
        if new_breadth_hist is not None:
            st["breadth_hist"] = new_breadth_hist
        # BUG 11 FIX: Previously the stock-level average (r5a/r10a) always overwrote
        # the accurate Yahoo Finance sector index returns from mkt.  Reversed merge order
        # so Yahoo Finance values (authoritative NSE sector index returns) win.
        # Stock-level averages are kept as fallback for sectors not covered by Yahoo tickers.
        _stock_sr5  = {s: float(np.mean(v)) for s, v in r5a.items()}
        _stock_sr10 = {s: float(np.mean(v)) for s, v in r10a.items()}
        st["sector_returns"]     = {**_stock_sr5,  **st.get("mkt", {}).get("sector_returns",     {})}
        st["sector_returns_10d"] = {**_stock_sr10, **st.get("mkt", {}).get("sector_returns_10d", {})}


def _apply_coverage_score(df_out: pd.DataFrame) -> pd.DataFrame:
    if df_out.empty or "Score" not in df_out.columns:
        return df_out
    scores = df_out["Score"].values.astype(float)
    df_out["score_percentile"] = (rankdata(scores, method="average") / max(len(scores), 1) * 100).round(1)

    # ── PreMoveRank: composite rank for pre-move detection ──────────────
    # Combines CoilingScore (compression quality), VCP (pattern maturity),
    # CLVAccum (institutional buying), and CoilStreakDays (persistence).
    # This is the column to sort by when hunting for tomorrow's breakout today.
    if all(c in df_out.columns for c in ["CoilingScore", "VCP", "CLVAccum"]):
        cs  = df_out["CoilingScore"].fillna(0).clip(0, 100) / 100.0
        vcp = df_out["VCP"].fillna(0).clip(0, 10) / 10.0
        clv = df_out["CLVAccum"].fillna(0).clip(0, 8) / 8.0
        # BUG 8 FIX: Previously normalized by 20, but stocks rarely coil >7 days
        # without breaking out.  A 5-day streak with /20 only contributed 0.0125
        # to PreMoveRank.  Normalizing by 7 makes streaks meaningfully discriminating.
        streak = df_out["CoilStreakDays"].fillna(0).clip(0, 7) / 7.0 if "CoilStreakDays" in df_out.columns else 0.0
        prox = df_out["Proximity"].fillna(0).clip(0, 10) / 10.0 if "Proximity" in df_out.columns else 0.5
        # Weighted composite: coiling quality 35%, VCP 25%, CLV 20%, proximity 15%, streak 5%
        df_out["PreMoveRank"] = (
            cs     * 0.35 +
            vcp    * 0.25 +
            clv    * 0.20 +
            prox   * 0.15 +
            streak * 0.05
        ).round(4) * 100
    return df_out


# ═════════════════════════════════════════════════════════════════
# 16.  BACKGROUND EXTRACTION
# ═════════════════════════════════════════════════════════════════

def bootstrap_calibration_from_db(st: dict, db_path: pathlib.Path) -> None:
    """Load per-stock and per-setup win rates from calibration DB into STATE on startup.
    Also restores coil_streak_days persisted from the last server session so streaks
    survive restarts and are not reset to 0 every morning."""
    cfg = SCORE_CFG
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.cursor().execute(
            "SELECT symbol, score, forward_ret, setup FROM calibration ORDER BY ts ASC"
        ).fetchall()
        # ── Restore persisted coil streaks ────────────────────────────────────
        try:
            kv = con.cursor().execute(
                "SELECT value FROM kv_store WHERE key='coil_streak_days'"
            ).fetchone()
            if kv:
                loaded_streaks = _json.loads(kv[0])
                if isinstance(loaded_streaks, dict):
                    st["coil_streak_days"] = {str(k): int(v) for k, v in loaded_streaks.items()}
        except Exception:
            pass   # kv_store table may not exist on first run — that's fine
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
    st["_setup_winrate"] = sw   # e.g. {"Breakout": 0.54, "PB-EMA": 0.48}
    # Calibration counts — used by Kelly gate (suppress Kelly below KELLY_MIN_CALIB_ROWS)
    st["_setup_winrate_counts"] = dict(sw_t)
    st["_stock_calib_counts"]   = dict(wr_t)

    if len(sv) >= 10:
        bx = [sum(1 for s in sv[i:i+cfg.CALIB_BREADTH_BATCH] if s > 50) / max(len(sv[i:i+cfg.CALIB_BREADTH_BATCH]), 1)
              for i in range(0, len(sv), cfg.CALIB_BREADTH_BATCH)]
        st["breadth_hist"] = (st.get("breadth_hist", []) + bx)[-200:]


def run_extraction(targets_dict: dict, min_avg_vol: int) -> None:
    cfg = SCORE_CFG
    with STATE_LOCK:
        s = STATE["extraction_status"]
        s.update({"running": True, "done": 0, "total": len(targets_dict),
                  "errors": 0, "rate_limited": 0, "empty": 0, "log": []})
        STATE["raw_data_cache"] = {}; STATE["score_cache"] = {}
        STATE["_row_stream_queue"] = []   # clear any leftover rows from previous run
        STATE["prev_close_cache"] = {}    # reset CHG% baseline for new extraction run
        for k in ("cs_rs_5d","cs_rs_20d","cs_bb_squeeze","cs_vol_dryup","cs_clv_accum","cs_vcp"):
            STATE[k] = {}
        # NOTE: coil_streak_days is intentionally NOT reset here.
        # Streaks must persist across extraction runs so that a stock coiling for
        # 3 consecutive days accumulates streak=3, not streak=1 on every run.
        # compute_cs_ranks() reads prev_streaks from STATE at the end of each run
        # and correctly increments / resets each symbol.  Wiping here meant every
        # run started from 0, so the streak could never exceed 1.
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
                    # Upstox silently returns HTTP 200 + empty candles[] when
                    # rate-limited on the historical endpoint (instead of 429).
                    # Treat it like a soft throttle and retry with backoff.
                    if attempt < cfg.FETCH_RETRIES:
                        time.sleep(delay); delay *= 2; continue
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
                elif err == "empty":  status["empty"] = status.get("empty", 0) + 1
                else:                 status["errors"] += 1
                continue
            if df is None: continue
            df_c = df[df["volume"] > 0].copy() if "volume" in df.columns else df.copy()
            if len(df_c) < cfg.EXTRACT_MIN_BARS_PREFILTER: continue
            if min_avg_vol > 0 and len(df_c) >= 5:
                if float(df_c["volume"].tail(20).mean()) < min_avg_vol:
                    continue
            # ── CHG% FIX: capture the true previous-session close BEFORE patching.
            # patch_live_bar() overwrites iloc[-1] with today's intraday LTP.
            # After patching, iloc[-2] may be 2 sessions ago on stocks where Upstox
            # does not yet have a today-row (partial day), inflating CHG% wildly.
            # Priority: (1) Upstox ohlc.close (authoritative broker prev-session close)
            #           (2) df_c["close"].iloc[-1] before any patch (last completed candle)
            lq_for_pc = live_q.get(normalize_key(targets_dict.get(sym, "")), {})
            _prev_close_for_chg = (
                lq_for_pc.get("prev_close")                                  # preferred: broker field
                or (float(df_c["close"].iloc[-1]) if len(df_c) >= 1 else None)  # fallback: last bar
            )
            with STATE_LOCK:
                STATE["raw_data_cache"][sym] = df_c
                if _prev_close_for_chg is not None:
                    STATE["prev_close_cache"][sym] = float(_prev_close_for_chg)

            # ── Progressive score: score this stock immediately with whatever
            # CS ranks are available so far. The result is provisional — RS
            # percentile ranks will shift as more stocks arrive — but it lets
            # the frontend show a live-updating table rather than a blank screen.
            # A final re-score pass runs after compute_cs_ranks() completes.
            #
            # BUG 7 FIX: coil_streak_days was cleared to {} at extraction start
            # (all other cs-rank dicts are reset), meaning every provisional score
            # showed CoilStreakDays=0 and the streak bonus never fired.
            # Fix: restore DB-persisted streaks into STATE before the first
            # progressive score so accumulated multi-day streaks are reflected
            # even in the provisional pass.  compute_cs_ranks() will then
            # correctly increment/reset them at the end of extraction.
            try:
                with STATE_LOCK:
                    if not STATE.get("coil_streak_days"):
                        try:
                            _db_conn_streak = get_db()
                            _kv_row = _db_conn_streak.execute(
                                "SELECT value FROM kv_store WHERE key='coil_streak_days'"
                            ).fetchone()
                            if _kv_row:
                                _loaded = _json.loads(_kv_row[0])
                                if isinstance(_loaded, dict):
                                    STATE["coil_streak_days"] = {str(k): int(v) for k, v in _loaded.items()}
                        except Exception:
                            pass   # kv_store may not exist — that's fine
            except Exception:
                pass
            try:
                mkt      = STATE.get("mkt") or {}
                live_lq  = live_q.get(normalize_key(targets_dict.get(sym, "")), {})
                result   = score_stock_dual(sym, df_c, live_lq,
                                            mkt.get("nifty_r5"), mkt.get("nifty_r20"))
                if result is not None:
                    _ltp_lq_raw = live_lq.get("ltp")
                    ltp_now = float(_ltp_lq_raw) if _ltp_lq_raw is not None else float(df_c["close"].iloc[-1])
                    vol_now = live_lq.get("volume") or (float(df_c["volume"].iloc[-1]) if "volume" in df_c.columns else 0)
                    # Use immutable prev_close_cache captured before patch_live_bar().
                    # Use explicit None check — stored value could legitimately be 0.0 (halted stock).
                    _pc_val = STATE.get("prev_close_cache", {}).get(sym)
                    prev_close = _pc_val if _pc_val is not None else \
                                 (float(df_c["close"].iloc[-2]) if len(df_c) >= 2 else ltp_now)
                    day_chg = round((ltp_now - prev_close) / (prev_close + 1e-9) * 100, 2)
                    row = {
                        "Ticker": sym, "LTP": round(float(ltp_now), 2),
                        "DayChg_pct": day_chg,
                        "DayHigh": round(float(live_lq.get("high", df_c["high"].iloc[-1])), 2),
                        "DayLow":  round(float(live_lq.get("low",  df_c["low"].iloc[-1])),  2),
                        "LiveVol": int(live_lq["volume"]) if live_lq.get("volume") else None,
                        **result,
                    }
                    # ── Compute PreMoveRank inline for this SSE row so the PREMOVE★ column
                    # is populated immediately — not just after full rescore.
                    # Mirrors _apply_coverage_score() formula exactly.
                    try:
                        _pmr_cs  = float(result.get("CoilingScore") or 0) / 100.0
                        _pmr_vcp = float(result.get("VCP")          or 0) / 10.0
                        _pmr_clv = float(result.get("CLVAccum")     or 0) / 8.0
                        _pmr_prx = float(result.get("Proximity")    or 0) / 10.0
                        _pmr_str = min(float(result.get("CoilStreakDays") or 0), 7.0) / 7.0
                        row["PreMoveRank"] = round(
                            (_pmr_cs * 0.35 + _pmr_vcp * 0.25 + _pmr_clv * 0.20 +
                             _pmr_prx * 0.15 + _pmr_str * 0.05) * 100, 4
                        )
                    except Exception:
                        row["PreMoveRank"] = None
                    # Sanitize NaN/inf for JSON serialisation
                    row = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                           for k, v in row.items()}
                    with STATE_LOCK:
                        STATE["score_cache"][sym] = {
                            "result": result, "ltp": ltp_now, "vol": vol_now,
                            "rsi_period": STATE.get("rsi_period", 14),
                            "version": SCORE_RESULT_VERSION,
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
                                     "LTP": round(float(live.get("ltp")) if live.get("ltp") is not None else float(df_raw["close"].iloc[-1]), 2),
                                     **cached})
        if rows_to_save:
            save_snapshot(rows_to_save, universe)
    except Exception:
        pass
    # Save full machine state so restarts restore immediately without re-extraction.
    threading.Thread(target=save_state_to_db, daemon=True).start()


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
            # ── CHG% FIX: if broker supplies prev_close (ohlc.close), update our
            # cache.  This handles the edge case where prev_close_cache was not
            # populated from the initial extraction (e.g. restart without re-extract).
            if lq.get("prev_close") and sym not in STATE.get("prev_close_cache", {}):
                STATE.setdefault("prev_close_cache", {})[sym] = float(lq["prev_close"])
            STATE["raw_data_cache"][sym] = patch_live_bar(df, lq)
        STATE["live_quotes_cache"] = live
        STATE["last_live_refresh"]  = time.time()
        # ── Partial cross-sectional RS recompute ───────────────────────────
        # VCP, Darvas, and BB squeeze are expensive structural patterns that
        # don't change meaningfully on a 1-minute price tick; we leave them
        # in place.  5d/20d relative-strength ranks, however, shift with every
        # price move and are cheap to recalculate (just returns + rankdata).
        _r5r: dict = {}; _r20r: dict = {}
        for _sym, _df in STATE["raw_data_cache"].items():
            _c = _df["close"]
            # Use iloc[-2] (yesterday) consistent with compute_cs_ranks fix
            if len(_c) >= 7:  _r5r[_sym]  = float(_c.iloc[-2] / _c.iloc[-7]  - 1)
            if len(_c) >= 22: _r20r[_sym] = float(_c.iloc[-2] / _c.iloc[-22] - 1)
        if _r5r:  STATE["cs_rs_5d"]  = _cdf_rank_dict(_r5r)
        if _r20r: STATE["cs_rs_20d"] = _cdf_rank_dict(_r20r)
        # Selective cache invalidation: evict entries where LTP moved >1% OR rsi_period changed.
        # The old approach wiped the entire score_cache, forcing a full VCP/Darvas/RS rescore
        # for all stocks on every price tick.  Indicators that don't depend on LTP (MA structure, VCP,
        # Darvas, RS percentile ranks) are still valid after a small price move.
        current_rsi = STATE.get("rsi_period", 14)
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
    # ── Check alerts against live prices ───────────────────────────────────────
    fired = _check_alerts(live, dict(STATE["targets"]))
    if fired:
        with STATE_LOCK:
            STATE.setdefault("_alert_queue", [])
            STATE["_alert_queue"].extend(fired)
    # Persist updated scores so live-price-driven changes survive restart too.
    threading.Thread(target=save_state_to_db, daemon=True).start()


def _check_alerts(live_q: dict, targets: dict) -> list:
    """Evaluate unfired alerts against current live prices / setup cache.
    Returns list of newly-fired alert dicts.  Called from refresh_live_prices_bg every 30s."""
    try:
        conn    = get_db()
        unfired = conn.execute("SELECT * FROM alerts WHERE fired=0").fetchall()
        if not unfired:
            return []
        fired_now = []
        for alert in unfired:
            sym  = alert["symbol"]
            cond = alert["cond"]
            val  = float(alert["value"])
            key  = normalize_key(targets.get(sym, ""))
            lq   = live_q.get(key, {})
            ltp  = lq.get("ltp")
            triggered = False
            if cond == "price_above" and ltp is not None and float(ltp) >= val:
                triggered = True
            elif cond == "price_below" and ltp is not None and float(ltp) <= val:
                triggered = True
            elif cond == "vol_confirm":
                cached = STATE["score_cache"].get(sym, {}).get("result", {})
                triggered = cached.get("SetupType") == "Breakout"
            elif cond == "setup_change":
                cached = STATE["score_cache"].get(sym, {}).get("result", {})
                triggered = cached.get("SetupType") not in (None, "", alert.get("prev_setup"))
            if triggered:
                conn.execute("UPDATE alerts SET fired=1 WHERE id=?", (alert["id"],))
                fired_now.append(dict(alert))
        if fired_now:
            conn.commit()
        return fired_now
    except Exception:
        return []


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


# Increment this whenever new fields are added to the score result dict.
# The screener will reject cached results with a lower version and re-score.
SCORE_RESULT_VERSION = 3   # bumped: BUY gates tightened, vol_sig Coiling fixed, reversal tail gate added
SCORE_CACHE_KV_KEY   = "score_cache_v3"   # kv_store key — bumped with version to purge stale v2 cache
STATE_KV_KEY         = "state_v2"          # kv_store key for market context + cs_ranks + targets

def save_state_to_db() -> None:
    """Persist the full dynamic machine state to kv_store.

    Saves three things so every restart restores a complete working system:
      1. score_cache  — all scored results (DaysSinceBreak, CoilStreakDays etc)
      2. state_blob   — market context, cs_ranks, targets, rs_div_hist,
                        sector_returns, breadth, param_registry
    The snapshots table already stores row_json per stock per run, but querying
    it on startup requires a full DataFrame rebuild.  kv_store is faster and
    stores exactly the in-memory dicts we need.
    """
    try:
        with STATE_LOCK:
            sc = dict(STATE["score_cache"])
            state_blob = {
                "targets":           dict(STATE.get("targets", {})),
                "cs_rs_5d":          dict(STATE.get("cs_rs_5d", {})),
                "cs_rs_20d":         dict(STATE.get("cs_rs_20d", {})),
                "cs_bb_squeeze":     dict(STATE.get("cs_bb_squeeze", {})),
                "cs_vol_dryup":      dict(STATE.get("cs_vol_dryup", {})),
                "cs_clv_accum":      dict(STATE.get("cs_clv_accum", {})),
                "cs_vcp":            dict(STATE.get("cs_vcp", {})),
                "sector_returns":    dict(STATE.get("sector_returns", {})),
                "sector_returns_10d":dict(STATE.get("sector_returns_10d", {})),
                "breadth_cache":     STATE.get("breadth_cache"),
                "breadth_hist":      list(STATE.get("breadth_hist", [])),
                "rs_div_hist":       {k: list(v) for k, v in STATE.get("rs_div_hist", {}).items()},
                "mkt":               {k: v for k, v in (STATE.get("mkt") or {}).items()
                                      if not isinstance(v, (dict, list)) or k in
                                      ("regime","vix_level","market_ok","nifty_r5","nifty_r20",
                                       "nifty_above_50dma","vix_falling","market_notes",
                                       "nifty_intraday_chg")},
                "saved_at":          time.time(),
            }
        # Score cache — only versioned entries
        to_save = {sym: entry for sym, entry in sc.items()
                   if entry.get("version") == SCORE_RESULT_VERSION
                   and entry.get("result") is not None}
        db = get_db()
        if to_save:
            db.execute("INSERT OR REPLACE INTO kv_store(key,value) VALUES(?,?)",
                       (SCORE_CACHE_KV_KEY, _json.dumps(to_save, default=str)))
        db.execute("INSERT OR REPLACE INTO kv_store(key,value) VALUES(?,?)",
                   (STATE_KV_KEY, _json.dumps(state_blob, default=str)))
        db.commit()
    except Exception:
        pass


def restore_state_from_db(st: dict) -> None:
    """Restore full dynamic machine state from kv_store on startup.

    Restores score_cache, market context, cross-sectional ranks, targets,
    sector returns, breadth history, rs_div_hist — everything needed so the
    screener serves correct data immediately after a restart without waiting
    for a full re-extraction.

    Only score_cache entries matching SCORE_RESULT_VERSION are loaded.
    Stale entries are silently skipped — the screener re-scores them lazily.
    """
    db = get_db()

    # ── 1. Score cache ────────────────────────────────────────────────────────
    try:
        row = db.execute("SELECT value FROM kv_store WHERE key=?",
                         (SCORE_CACHE_KV_KEY,)).fetchone()
        if row:
            loaded = _json.loads(row[0])
            if isinstance(loaded, dict):
                for sym, entry in loaded.items():
                    if (entry.get("version") == SCORE_RESULT_VERSION
                            and entry.get("result") is not None):
                        st["score_cache"][sym] = entry
    except Exception:
        pass

    # ── 2. State blob: cs_ranks, targets, market context, etc ────────────────
    try:
        row = db.execute("SELECT value FROM kv_store WHERE key=?",
                         (STATE_KV_KEY,)).fetchone()
        if not row:
            return
        blob = _json.loads(row[0])
        if not isinstance(blob, dict):
            return
        # Reject if saved more than 26 hours ago (stale market data)
        if time.time() - blob.get("saved_at", 0) > 26 * 3600:
            return
        for key in ("targets", "cs_rs_5d", "cs_rs_20d", "cs_bb_squeeze",
                    "cs_vol_dryup", "cs_clv_accum", "cs_vcp",
                    "sector_returns", "sector_returns_10d"):
            if key in blob and isinstance(blob[key], dict):
                st[key] = blob[key]
        if "breadth_cache" in blob:
            st["breadth_cache"] = blob["breadth_cache"]
        if "breadth_hist" in blob and isinstance(blob["breadth_hist"], list):
            st["breadth_hist"] = blob["breadth_hist"]
        if "rs_div_hist" in blob and isinstance(blob["rs_div_hist"], dict):
            st["rs_div_hist"] = blob["rs_div_hist"]
        if "mkt" in blob and isinstance(blob["mkt"], dict):
            st["mkt"] = blob["mkt"]

        # ── BO saturation FIX: seed _bo_saturation_frac from the restored score
        # cache so the saturation guard fires correctly on first screener GET
        # after a restart (instead of defaulting to 0 and never discounting).
        _cached_results_restore = [v.get("result") for v in st.get("score_cache", {}).values()
                                    if v.get("result")]
        if _cached_results_restore:
            _bo_cnt  = sum(1 for r in _cached_results_restore
                           if r.get("SetupType") in ("Breakout", "Coiling"))
            _tot_cnt = max(len(_cached_results_restore), 1)
            st["_bo_saturation_frac"] = _bo_cnt / _tot_cnt
    except Exception:
        pass


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

        CREATE TABLE IF NOT EXISTS kv_store (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()


init_db()
bootstrap_calibration_from_db(STATE, DB_PATH)
restore_state_from_db(STATE)   # restore full dynamic machine state from DB

# Load full 2500-stock sector DB into memory (sector_map.db built by sector_db.py).
# Must run before any extraction so get_sector() has full coverage immediately.
_load_sector_db_cache()

# Refresh live sector map from NSE indices in background — overwrites DB cache
# entries for large-caps with the freshest classification.
_maybe_refresh_sector_map()


# ═════════════════════════════════════════════════════════════════
# 17b. BACKGROUND LIVE-PRICE DAEMON
# ═════════════════════════════════════════════════════════════════
# Runs forever in a daemon thread started at process boot.
# Sleeps for LIVE_REFRESH_SEC between iterations so the cadence is
# controlled entirely by the existing ScoreConfig constant — no new
# magic numbers.  refresh_live_prices_bg() already has its own
# time-guard so double-calls from the screener route are harmless.
# When it updates live_quotes_cache it bumps last_live_refresh, which
# the SSE generator detects and immediately pushes a 'prices' event
# to every connected browser — causing the blink without any manual
# refresh.

def _live_price_daemon() -> None:
    """Daemon: refreshes live quotes (SSE blink) and fills forward returns."""
    _fill_cycle = 0
    while True:
        try:
            time.sleep(SCORE_CFG.LIVE_REFRESH_SEC)
            refresh_live_prices_bg()
            # Fill forward returns every 10th cycle (≈ every 10 minutes at default cadence)
            # so we don't hammer the DB on every price tick.
            _fill_cycle += 1
            if _fill_cycle % 10 == 0:
                _fill_forward_returns_bg()
        except Exception:
            pass   # never let the daemon die


_daemon_thread = threading.Thread(target=_live_price_daemon, daemon=True, name="live-price-daemon")
_daemon_thread.start()


def save_snapshot(rows: list, universe: str) -> None:
    if not rows: return
    ts = _dt.now().isoformat(); conn = get_db()
    conn.executemany(
        "INSERT INTO snapshots(ts,universe,row_json) VALUES(?,?,?)",
        [(ts, universe, _json.dumps(row)) for row in rows]
    )
    conn.commit()


# ═════════════════════════════════════════════════════════════════
# 17c.  SECTOR DB API ROUTES
# ═════════════════════════════════════════════════════════════════

@app.get("/api/sector/coverage")
async def sector_coverage():
    """Coverage stats: total mapped, breakdown by sector and source."""
    return _sector_get_stats()


@app.get("/api/sector/all")
async def sector_all():
    """All symbol→sector mappings as a JSON list."""
    return _sector_get_all()


@app.get("/api/sector/lookup/{symbol}")
async def sector_lookup(symbol: str):
    """Look up the sector for a single symbol."""
    sec = get_sector(symbol.upper())
    return {"symbol": symbol.upper(), "sector": sec}


@app.post("/api/sector/manual")
async def sector_manual(body: dict):
    """
    Manually pin a symbol to a sector.  Manual pins are never overwritten
    by auto-refresh or --build.

    Body: {"symbol": "XYZ", "sector": "IT"}
    """
    sym    = (body.get("symbol") or "").strip().upper()
    sector = (body.get("sector") or "").strip()
    if not sym or not sector:
        raise HTTPException(400, "symbol and sector are required")
    _sector_manual_add(sym, sector)
    _sector_reload_cache()    # refresh in-memory cache immediately
    return {"symbol": sym, "sector": sector, "source": "manual"}


@app.post("/api/sector/rebuild")
async def sector_rebuild(background_tasks: BackgroundTasks):
    """
    Trigger a full sector DB rebuild in the background (fetches NSE index
    constituents + master CSV).  Reloads the in-memory cache when done.
    Returns immediately.
    """
    def _do_rebuild():
        try:
            _sector_build()
            _sector_reload_cache()
        except Exception as e:
            print(f"[sector_rebuild] error: {e}")
    background_tasks.add_task(_do_rebuild)
    return {"status": "rebuild started"}


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

# Alias used by login.html manual token paste form
@app.post("/auth/token")
async def set_token_alias(body: dict):
    return await set_token(body)

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

# Alias used by login.html status check
@app.get("/auth/status")
async def auth_status_alias():
    return await token_status()


def _get_nse_equity_df(master_df) -> "pd.DataFrame":
    """
    Return all tradeable NSE equity rows from the Upstox master instrument list.

    The Upstox NSE master uses `instrument_type` as the NSE *series code*, NOT a
    generic asset-class flag.  Filtering only on "EQ" captures the regular
    large/mid-cap series (~488 stocks) but misses:

        BE  — Trade-for-Trade / Z-category equities
        BZ  — Z-series equities (low-compliance companies)
        SM  — SME (NSE Emerge) equities
        ST  — Odd-lot / suspended equities
        IL  — Institutional placement series

    All of these are plain NSE-listed equities with daily OHLCV bars available
    from the Upstox historical-candle API.  Including them brings the universe
    from ~488 up to the full ~2,500 NSE equity stocks.
    """
    # BUG 13 FIX: Removed "ST" (NSE odd-lot / suspended series) — these securities
    # have erratic prices and negligible volume.  They still passed min_avg_vol
    # sometimes and consumed API fetch quota for zero actionable signals.
    _EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "IL"}
    _NON_EQUITY    = {"INDEX", "MF", "UNITMF", "GB", "GS", "TB",
                      "FUTIDX", "FUTSTK", "OPTIDX", "OPTSTK"}

    nse = master_df[master_df["exchange"] == "NSE"].copy()

    mask_include = nse["instrument_type"].isin(_EQUITY_SERIES)
    mask_exclude = nse["instrument_type"].isin(_NON_EQUITY)

    seg_str = nse["segment"].astype(str).str.upper()
    mask_bad_seg = (
        seg_str.str.contains("INDICES", na=False) |
        seg_str.str.contains("FUTIDX|FUTSTK|OPTIDX|OPTSTK", na=False, regex=True)
    )

    eq = nse[mask_include & ~mask_exclude & ~mask_bad_seg].copy()

    # Deduplicate: if the same trading_symbol appears in multiple series
    # (e.g. both EQ and BE), prefer EQ > BE > others.
    _SERIES_PREF = {"EQ": 0, "BE": 1, "BZ": 2, "SM": 3, "ST": 4, "IL": 5}
    eq["_series_order"] = eq["instrument_type"].map(_SERIES_PREF).fillna(99).astype(int)
    eq = (eq.sort_values("_series_order")
            .drop_duplicates(subset=["trading_symbol"], keep="first")
            .drop(columns=["_series_order"]))
    return eq

# ── Universe / Extract ────────────────────────────────────────────
@app.get("/api/universe")
async def get_universe(universe: str = "Nifty 50"):
    master_df = get_live_master()
    if master_df.empty: raise HTTPException(503, "Could not load instrument master")
    eq = _get_nse_equity_df(master_df)
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
    rsi_p    = body.get("rsi_period",  14)   # BUG 14 FIX: default 7→14
    STATE["rsi_period"]          = int(rsi_p)
    STATE["min_avg_vol"]         = int(min_vol)
    STATE["sector_cap_enabled"]  = body.get("sector_cap_enabled", False)
    if STATE["extraction_status"]["running"]:
        return {"status": "already_running"}
    master_df = get_live_master()
    if master_df.empty: raise HTTPException(503, "Master list unavailable")
    eq = _get_nse_equity_df(master_df)
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
    cached = len(STATE["raw_data_cache"])
    return {**s,
            "cached":       cached,
            "empty":        s.get("empty", 0),
            "no_data":      s.get("done", 0) - cached - s.get("errors", 0) - s.get("rate_limited", 0),
            "live_quotes":  len(STATE["live_quotes_cache"]),
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
                    yield f"event: status\ndata: {_json.dumps(payload)}\n\n"

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
                        yield "event: rescore_complete\ndata: {}\n\n"
                    else:
                        yield f"event: row\ndata: {_json.dumps(row, default=str)}\n\n"

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
                        # Use immutable prev_close_cache (captured before patching).
                        # This is the same source used by the screener GET so the
                        # blink-update and initial load are always consistent.
                        _pc = STATE.get("prev_close_cache", {}).get(sym)
                        prev_close = _pc if _pc is not None else \
                                     (float(df_raw["close"].iloc[-2])
                                      if len(df_raw) >= 2 else float(ltp_now))
                        chg = round((float(ltp_now) - prev_close)
                                    / (prev_close + 1e-9) * 100, 2)
                        prices[sym] = {
                            "ltp": round(float(ltp_now), 2),
                            "chg": chg,
                            "vol": int(live["volume"]) if live.get("volume") else None,
                        }
                    if prices:
                        yield f"event: prices\ndata: {_json.dumps(prices)}\n\n"

                # ── alert events: drain _alert_queue and push each as SSE ──
                with STATE_LOCK:
                    pending_alerts = STATE.get("_alert_queue", [])[:]
                    STATE["_alert_queue"] = []
                for _a in pending_alerts:
                    yield f"event: alert\ndata: {_json.dumps(_a)}\n\n"

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
async def get_screener(sort_by: str = "CoilingScore", horizon: str = "ALL"):
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
    # BUG 15 FIX: Previously computed _bo_saturation_frac from the stale score_cache
    # snapshot (always one pass behind).  On first extraction (empty cache) the fraction
    # was always 0 and the saturation guard never fired.
    # Fix: compute from the current snapshot for the first pass (cache-based), but also
    # seed STATE["_bo_saturation_frac"] from the restored DB score cache at startup
    # so restarts don't reset it to 0.
    _cached_results = [v.get("result") for v in sc_snap.values() if v.get("result")]
    _bo_count  = sum(1 for r in _cached_results if r.get("SetupType") in ("Breakout", "Coiling"))
    _tot_cache = max(len(_cached_results), 1)
    # Use a rolling blend: 80% new measurement, 20% previous value for stability
    _prev_sat = STATE.get("_bo_saturation_frac", _bo_count / _tot_cache)
    STATE["_bo_saturation_frac"] = 0.8 * (_bo_count / _tot_cache) + 0.2 * _prev_sat

    # Build a universe-wide EMI map from the previous score_cache pass so that
    # aggregate_score can percentile-rank each stock's raw EMI and avoid
    # systematically favouring the most volatile names.
    _emi_universe: dict = {}
    for _sym, _sc in sc_snap.items():
        _res = _sc.get("result")
        if _res and _res.get("EMI") is not None:
            _emi_universe[_sym] = float(_res["EMI"])
    # Inject into STATE so score_stock_dual → aggregate_score can read it via cs_state
    STATE["_emi_universe"] = _emi_universe

    for sym, df_raw in raw_snap.items():
        try:
            if _min_vol > 0 and "volume" in df_raw.columns and len(df_raw) >= 5:
                if float(df_raw["volume"].tail(20).mean()) < _min_vol: continue
            live     = lq_snap.get(normalize_key(tgt_snap.get(sym, "")), {})
            _ltp_raw = live.get("ltp")
            ltp_now  = float(_ltp_raw) if _ltp_raw is not None else float(df_raw["close"].iloc[-1])
            vol_now  = live.get("volume") or (float(df_raw["volume"].iloc[-1]) if "volume" in df_raw.columns else 0)
            cached_e = sc_snap.get(sym)
            # Cache hit: LTP within 1% AND rsi_period unchanged
            rsi_match = cached_e and cached_e.get("rsi_period") == STATE.get("rsi_period", 14)
            ltp_match = cached_e and abs(cached_e.get("ltp", 0) - ltp_now) < 0.01 * ltp_now
            ver_match = cached_e and cached_e.get("version") == SCORE_RESULT_VERSION
            if rsi_match and ltp_match and ver_match and cached_e.get("result") is not None:
                result = cached_e["result"]
            else:
                result = score_stock_dual(sym, df_raw, live, nifty_r5, nifty_r20)
                with STATE_LOCK:
                    STATE["score_cache"][sym] = {
                        "result": result, "ltp": ltp_now, "vol": vol_now,
                        "rsi_period": STATE.get("rsi_period", 14),
                        "version": SCORE_RESULT_VERSION,
                    }
            if result is None: continue
            # Use immutable prev_close_cache captured before patch_live_bar().
            # Fallback to iloc[-2] only if cache entry is missing.
            _pc = STATE.get("prev_close_cache", {}).get(sym)
            prev_close = _pc if _pc is not None else \
                         (float(df_raw["close"].iloc[-2]) if len(df_raw) >= 2 else ltp_now)
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
        seen = set(); capped = []; unknown_count = 0
        for _, row in df_out.iterrows():
            sec = str(row.get("Sector", "?"))
            if sec == "?":
                # Allow up to 3 unknown-sector stocks through so genuine signals
                # aren't completely suppressed by a missing sector mapping.
                if unknown_count < 3:
                    capped.append(row); unknown_count += 1
            elif sec not in seen:
                capped.append(row); seen.add(sec)
        df_out = pd.DataFrame(capped).reset_index(drop=True); df_out["Rank"] = df_out.index + 1
    if horizon != "ALL":
        df_out = df_out[df_out["Horizon"] == horizon].reset_index(drop=True); df_out["Rank"] = df_out.index + 1
    df_out = df_out.replace({np.nan: None, np.inf: None, -np.inf: None})
    # Persist full machine state after every screener build so restarts restore instantly.
    threading.Thread(target=save_state_to_db, daemon=True).start()
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


@app.get("/api/premove")
async def get_premove_candidates(min_coil: float = 55.0, min_vcp: float = 4.0,
                                  min_streak: int = 0, limit: int = 30):
    """
    Pre-move watchlist: stocks in tight compression BEFORE the breakout.
    Sorted by PreMoveRank (CoilingScore × VCP × CLV × Proximity composite).

    Filters:
      min_coil   — minimum CoilingScore (0-100). Default 55.
      min_vcp    — minimum VCP score points (0-10). Default 4.
      min_streak — minimum consecutive days in top-tier compression. Default 0.
      limit      — max rows to return. Default 30.
    """
    mkt = STATE["mkt"] or get_market_context()
    nifty_r5 = mkt.get("nifty_r5"); nifty_r20 = mkt.get("nifty_r20")
    raw_snap = dict(STATE["raw_data_cache"])
    lq_snap  = dict(STATE["live_quotes_cache"])
    tgt_snap = dict(STATE["targets"])
    sc_snap  = dict(STATE["score_cache"])
    rows = []
    for sym, df_raw in raw_snap.items():
        try:
            live     = lq_snap.get(normalize_key(tgt_snap.get(sym, "")), {})
            _ltp_raw_pm = live.get("ltp")
            ltp_now  = float(_ltp_raw_pm) if _ltp_raw_pm is not None else float(df_raw["close"].iloc[-1])
            cached_e = sc_snap.get(sym)
            rsi_match = cached_e and cached_e.get("rsi_period") == STATE.get("rsi_period", 14)
            ltp_match = cached_e and abs(cached_e.get("ltp", 0) - ltp_now) < 0.01 * ltp_now
            ver_match = cached_e and cached_e.get("version") == SCORE_RESULT_VERSION
            if rsi_match and ltp_match and ver_match and cached_e.get("result") is not None:
                result = cached_e["result"]
            else:
                result = score_stock_dual(sym, df_raw, live, nifty_r5, nifty_r20)
                if result:
                    with STATE_LOCK:
                        STATE["score_cache"][sym] = {
                            "result": result, "ltp": ltp_now, "vol": 0,
                            "rsi_period": STATE.get("rsi_period", 14),
                            "version": SCORE_RESULT_VERSION,
                        }
            if result is None: continue
            # Pre-move filter: must be in compression or pullback phase, not already broken out
            if result.get("SetupType") not in ("Coiling", "PB-EMA", "PB-Dry", "PB-Deep", "Base"): continue
            if (result.get("CoilingScore") or 0) < min_coil: continue
            if (result.get("VCP") or 0) < min_vcp: continue
            if (result.get("CoilStreakDays") or 0) < min_streak: continue
            _pc = STATE.get("prev_close_cache", {}).get(sym)
            prev_close = _pc if _pc is not None else \
                         (float(df_raw["close"].iloc[-2]) if len(df_raw) >= 2 else ltp_now)
            day_chg = round((ltp_now - prev_close) / (prev_close + 1e-9) * 100, 2)
            rows.append({
                "Ticker": sym,
                "LTP": round(float(ltp_now), 2),
                "DayChg_pct": day_chg,
                "LiveVol": int(live["volume"]) if live.get("volume") else None,
                **result,
            })
        except Exception:
            continue
    if not rows:
        return {"rows": [], "count": 0, "filters": {"min_coil": min_coil, "min_vcp": min_vcp, "min_streak": min_streak}}
    df_out = pd.DataFrame(rows)
    df_out = _apply_coverage_score(df_out)
    # Sort by PreMoveRank if available, else CoilingScore
    sort_col = "PreMoveRank" if "PreMoveRank" in df_out.columns else "CoilingScore"
    df_out = df_out.sort_values(sort_col, ascending=False).head(limit).reset_index(drop=True)
    df_out.insert(0, "Rank", df_out.index + 1)
    df_out = df_out.replace({np.nan: None, np.inf: None, -np.inf: None})
    return {
        "rows": df_out.to_dict("records"),
        "count": len(df_out),
        "filters": {"min_coil": min_coil, "min_vcp": min_vcp, "min_streak": min_streak},
        "regime": mkt.get("regime", "BULL"),
    }

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
                         (round(exit_p / entry - 1, 6), rec["id"]))
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

# ── Alerts ───────────────────────────────────────────────────────

@app.post("/api/alerts")
async def create_alert(body: dict):
    """Create a new price or vol_confirm alert for a symbol.

    body: {symbol, cond: 'price_above'|'price_below'|'vol_confirm'|'setup_change', value}
    """
    sym   = str(body.get("symbol", "")).upper()
    cond  = str(body.get("cond", "price_above"))
    value = float(body.get("value", 0.0))
    if not sym or cond not in ("price_above", "price_below", "vol_confirm", "setup_change"):
        raise HTTPException(400, "Invalid alert body")
    conn = get_db()
    conn.execute(
        "INSERT INTO alerts(symbol,cond,value,fired,ts) VALUES(?,?,?,0,?)",
        (sym, cond, value, _dt.now().isoformat())
    )
    conn.commit()
    return {"status": "ok", "symbol": sym, "cond": cond, "value": value}


@app.get("/api/alerts")
async def get_alerts():
    """Return all alerts (unfired and fired) ordered by ts desc."""
    rows = get_db().execute("SELECT * FROM alerts ORDER BY ts DESC").fetchall()
    return {"alerts": [dict(r) for r in rows]}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete an alert by id."""
    conn = get_db()
    conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit()
    return {"status": "ok"}


# ── News ──────────────────────────────────────────────────────────
@app.get("/api/news")
async def get_news(symbol: str = ""):
    try:
        import feedparser, html as _html
        from email.utils import parsedate_to_datetime
    except ImportError:
        return {"articles": [], "symbol": symbol, "error": "feedparser not installed"}

    feeds = [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ]

    # Build a match set for the symbol: the ticker itself plus any known
    # long-form name aliases stored in the master instrument list.
    # All matching is case-insensitive whole-word so "INFY" never matches "INFINITY".
    import re as _re
    _sym_upper = symbol.strip().upper()
    _aliases: list[str] = [_sym_upper] if _sym_upper else []
    if _sym_upper and _sym_upper in STATE.get("targets", {}):
        # targets maps sym → upstox key; try to find long name in master cache
        _master = STATE.get("master_cache", pd.DataFrame())
        if not _master.empty and "trading_symbol" in _master.columns and "name" in _master.columns:
            _rows = _master[_master["trading_symbol"] == _sym_upper]
            if not _rows.empty:
                _name = str(_rows.iloc[0].get("name", "")).strip()
                if _name:
                    # Add first word of name (e.g. "Infosys" from "Infosys Ltd")
                    _aliases.append(_name.split()[0].upper())
                    _aliases.append(_name.upper())

    def _matches(text: str) -> bool:
        if not _sym_upper:
            return True   # no filter — return everything
        t = text.upper()
        return any(
            _re.search(r'\b' + _re.escape(a) + r'\b', t)
            for a in _aliases
        )

    # Articles older than NEWS_MAX_AGE_DAYS are discarded — derived from ScoreConfig.
    _cutoff = datetime.now(timezone.utc) - timedelta(days=SCORE_CFG.NEWS_MAX_AGE_DAYS)

    def _parse_pub(pub_str: str) -> Optional[datetime]:
        if not pub_str:
            return None
        try:
            dt = parsedate_to_datetime(pub_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            try:
                return datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except Exception:
                return None

    arts = []
    for url in feeds:
        try:
            for e in feedparser.parse(url).entries[:25]:
                title = _html.unescape(getattr(e, "title",   ""))
                summ  = _html.unescape(getattr(e, "summary", ""))[:300]
                link  = getattr(e, "link",      "#")
                pub   = getattr(e, "published", "")
                # Date gate: skip articles outside the freshness window
                _pub_dt = _parse_pub(pub)
                if _pub_dt is not None and _pub_dt < _cutoff:
                    continue
                # Symbol match gate
                if not _matches(title + " " + summ):
                    continue
                arts.append({
                    "title":   title,
                    "summary": summ,
                    "link":    link,
                    "pub":     pub,
                    "pub_iso": _pub_dt.isoformat() if _pub_dt else None,
                    "source":  url.split("/")[2],
                })
        except Exception:
            pass

    # Sort newest-first using parsed ISO date when available
    arts.sort(key=lambda a: a.get("pub_iso") or "", reverse=True)
    return {"articles": arts[:30], "symbol": symbol}

# ── Chart ─────────────────────────────────────────────────────────
@app.get("/api/chart/{symbol}")
async def get_chart_data(symbol: str, bars: int = 0, tf: str = "1d"):
    """Return OHLCV + indicators for charting.
    bars=0  → return ALL available history (default).
    bars=N  → return last N bars.
    tf      → timeframe hint stored in response for client use.
    """
    df = STATE["raw_data_cache"].get(symbol.upper())
    if df is None: raise HTTPException(404, f"{symbol} not in cache — run extraction first")
    df = df.copy()
    if bars and bars > 0:
        df = df.tail(bars)
    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    c = df["close"]
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean().round(2)
    df["ema20"] = c.ewm(span=20, adjust=False).mean().round(2)
    df["ema50"] = c.ewm(span=50, adjust=False).mean().round(2)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"]  - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"]    = tr.ewm(span=14, adjust=False).mean().round(2)
    rsi_p        = STATE.get("rsi_period", 14)
    delta        = c.diff()
    gain         = delta.clip(lower=0).ewm(alpha=1 / rsi_p, adjust=False).mean()
    loss         = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_p, adjust=False).mean()
    df["rsi"]    = (100 - 100 / (1 + gain / loss.replace(0, float("nan")))).round(1)
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean().round(0)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
    cols = ["time","open","high","low","close","volume","ema9","ema20","ema50","atr","rsi","vol_ma20"]
    return {"symbol": symbol.upper(), "tf": tf, "bars": df[[c for c in cols if c in df.columns]].to_dict("records")}

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
                _ltp_raw_es = live.get("ltp")
                ltp_now = float(_ltp_raw_es) if _ltp_raw_es is not None else float(df_raw["close"].iloc[-1])
                with STATE_LOCK:
                    STATE["score_cache"][sym] = {
                        "result": cached, "ltp": ltp_now, "vol": 0,
                        "rsi_period": STATE.get("rsi_period", 14),
                    }
        except Exception as e:
            raise HTTPException(500, f"Score computation failed: {e}")
    if not cached: raise HTTPException(404, "Score unavailable — insufficient data (need 60+ bars)")
    s   = cached
    _ltp_raw_ex = live.get("ltp")
    ltp = float(_ltp_raw_ex) if _ltp_raw_ex is not None else float(df_raw["close"].iloc[-1])

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
    _kelly = s.get('KellyFrac')
    _kelly_str = f"{round(_kelly * 100, 1)}%" if _kelly is not None else "— (insufficient calibration data)"
    line("Kelly Size", _kelly_str, "Kelly criterion optimal position size", "% of capital to risk; halve this for safety")
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

# ── Forward return fill ──────────────────────────────────────────
def _fill_forward_returns_bg() -> int:
    """Fill forward_ret for calibration rows that are FORWARD_RETURN_DAYS old.
    Uses the live_quotes_cache for speed; falls back to the raw_data_cache close.
    Called by the daemon and by the calibration snapshot route.

    BUG 2 FIX: Previously used iloc[-(FORWARD_RETURN_DAYS+1)] as entry price,
    which always pointed to a fixed bar offset regardless of when the snapshot
    was taken. Now we verify the cache has advanced by >= FORWARD_RETURN_DAYS
    trading bars since the snapshot, then use the latest available close as exit
    and the bar at snapshot-time as entry — giving a true N-bar forward return.
    """
    cfg  = SCORE_CFG
    conn = get_db()
    # Use 7 calendar days as cutoff (covers weekends + 1-day buffer for holidays)
    # to ensure at least FORWARD_RETURN_DAYS trading days have elapsed.
    calendar_days_buffer = max(cfg.FORWARD_RETURN_DAYS + 2, 7)
    cutoff_ts = (_dt.now() - timedelta(days=calendar_days_buffer)).isoformat()
    pending = conn.execute(
        "SELECT id, symbol, score, ts FROM calibration WHERE forward_ret IS NULL AND ts <= ?",
        (cutoff_ts,)
    ).fetchall()
    if not pending:
        return 0
    updated = 0
    lq_snap  = dict(STATE.get("live_quotes_cache", {}))
    raw_snap = dict(STATE.get("raw_data_cache",    {}))
    tgt_snap = dict(STATE.get("targets",           {}))
    for row in pending:
        rid, sym, _entry_score, snap_ts = row["id"], row["symbol"], row["score"], row["ts"]
        try:
            df_r = raw_snap.get(sym)
            if df_r is None or len(df_r) < cfg.FORWARD_RETURN_DAYS + 2:
                continue

            # BUG 2 FIX: Align snapshot timestamp to the closest bar in the cache
            # so entry_price is the actual close at signal time, not a fixed offset.
            if "time" in df_r.columns:
                _times = pd.to_datetime(df_r["time"])
                _snap_dt = pd.to_datetime(snap_ts)
                _bar_idx = (_times - _snap_dt).abs().idxmin()
                _bar_pos = df_r.index.get_loc(_bar_idx)
            else:
                # No time column — fall back to FORWARD_RETURN_DAYS bars from end
                _bar_pos = max(0, len(df_r) - cfg.FORWARD_RETURN_DAYS - 1)

            # Require at least FORWARD_RETURN_DAYS bars have been added since snapshot
            _bars_since = len(df_r) - 1 - _bar_pos
            if _bars_since < cfg.FORWARD_RETURN_DAYS:
                continue   # not enough bars yet — skip, will be filled later

            entry_close = float(df_r["close"].iloc[_bar_pos])
            if entry_close <= 0:
                continue

            # Exit: use live LTP if available and fresh, else last bar close
            lq  = lq_snap.get(normalize_key(tgt_snap.get(sym, "")), {})
            ltp = lq.get("ltp")
            exit_price = float(ltp) if ltp else float(df_r["close"].iloc[-1])

            fwd = (exit_price - entry_close) / (entry_close + 1e-9)
            conn.execute(
                "UPDATE calibration SET forward_ret=? WHERE id=?",
                (round(fwd, 6), rid)
            )
            updated += 1
        except Exception:
            continue
    conn.commit()
    return updated


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
    import asyncio
    loop = asyncio.get_running_loop()
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
            _ltp_raw_exp = live.get("ltp")
            ltp  = float(_ltp_raw_exp) if _ltp_raw_exp is not None else float(df_raw["close"].iloc[-1])
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
# 20.  BACKTEST ENDPOINT
# ═════════════════════════════════════════════════════════════════

@app.get("/api/backtest")
async def backtest(
    min_score: float = 60.0,
    setup: str = "",
    regime: str = "",
    min_rows: int = 10,
):
    """Replay calibration DB rows with non-null forward_ret and return
    hit-rate, avg forward return, Sharpe, and max drawdown for the given
    score / setup / regime filter.  This turns the calibration DB into
    actionable validation of score thresholds.

    Query params:
      min_score  – only rows with score >= this value  (default 60)
      setup      – filter by SetupType e.g. 'Breakout' (default: all)
      regime     – filter by regime e.g. 'BULL'        (default: all)
      min_rows   – refuse analysis if fewer rows than this (default 10)
    """
    conn = get_db()
    q    = "SELECT score, forward_ret, setup, regime FROM calibration WHERE forward_ret IS NOT NULL AND score >= ?"
    args: list = [min_score]
    if setup:
        q += " AND setup = ?";  args.append(setup)
    if regime:
        q += " AND regime = ?"; args.append(regime)
    rows = conn.execute(q, args).fetchall()

    if len(rows) < min_rows:
        return {"error": f"Only {len(rows)} rows match — need at least {min_rows} for meaningful analysis",
                "rows_found": len(rows), "filter": {"min_score": min_score, "setup": setup, "regime": regime}}

    fwd   = np.array([float(r["forward_ret"]) for r in rows])
    scores = np.array([float(r["score"])       for r in rows])

    hit_rate    = float((fwd > 0).mean())
    avg_ret     = float(fwd.mean())
    std_ret     = float(fwd.std()) if len(fwd) > 1 else 0.0
    # NOTE: This is a cross-sectional information ratio, NOT a time-series Sharpe.
    # Each row is an independent snapshot (not a sequential equity curve), so the
    # standard annualisation factor sqrt(252/N) over-counts periods and inflates
    # the value significantly for short holding periods (e.g. N=5 → factor ~7).
    # The correct interpretation: mean/std of the forward-return distribution
    # (i.e. how consistently positive the distribution is).  We preserve the
    # sqrt-scaled version for backward compatibility but label it clearly.
    information_ratio = float(avg_ret / (std_ret + 1e-9) * np.sqrt(252 / SCORE_CFG.FORWARD_RETURN_DAYS))
    # Simple (unscaled) information ratio for cross-sectional snapshot analysis:
    raw_ir = float(avg_ret / (std_ret + 1e-9))

    # Max drawdown — sort descending (winners first) for a conservative
    # worst-case path estimate.  True path-dependent DD requires a time-ordered
    # portfolio equity curve which is not available from cross-sectional snapshots.
    fwd_sorted = np.sort(fwd)[::-1]
    cum   = np.cumprod(1 + fwd_sorted)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    # Score decile breakdown
    decile_stats = []
    for d in range(1, 11):
        lo = float(np.percentile(scores, (d - 1) * 10))
        hi = float(np.percentile(scores, d * 10))
        mask = (scores >= lo) & (scores <= hi)
        if mask.sum() >= 3:
            decile_stats.append({
                "decile": d, "score_range": [round(lo, 1), round(hi, 1)],
                "n": int(mask.sum()),
                "hit_rate": round(float((fwd[mask] > 0).mean()), 3),
                "avg_fwd_ret": round(float(fwd[mask].mean()), 4),
            })

    return {
        "filter":   {"min_score": min_score, "setup": setup or "all", "regime": regime or "all"},
        "n_rows":   len(rows),
        "hit_rate": round(hit_rate, 3),
        "avg_fwd_ret_pct": round(avg_ret * 100, 2),
        "information_ratio": round(information_ratio, 3),
        "raw_ir": round(raw_ir, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "forward_return_days": SCORE_CFG.FORWARD_RETURN_DAYS,
        "decile_breakdown": decile_stats,
        "note": ("information_ratio is annualised mean/std and is inflated for short N-day "
                 "holding periods due to cross-sectional (not time-series) data. "
                 "Use hit_rate and avg_fwd_ret_pct per decile to validate score thresholds. "
                 "BreakoutProb is NOT a calibrated probability."),
    }


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
