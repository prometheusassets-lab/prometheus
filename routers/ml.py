"""
routers/ml.py — MONARCH PRO ML Predictor Engine
FastAPI router ported from 4_ML_Predictor.py.
Stacked ensemble: RF + GBM → meta-learner, walk-forward CV.
"""

import time, math
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

router = APIRouter(prefix="/api/ml", tags=["ml"])

_STATE: dict = {}
_ml_models: dict = {}   # {symbol: result}

def init_state(state_dict: dict):
    global _STATE
    _STATE = state_dict

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
ML_FEATURE_COLS = [
    "rsi7", "rsi14", "macd_hist", "bb_pct", "atr_pct",
    "vol_ratio", "ema9_dist", "ema20_dist", "ema50_dist",
    "ret1d", "ret5d", "ret20d",
    "high52w_dist", "low52w_dist",
    "vol_cont", "clv",
]

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _build_features(df: pd.DataFrame, rsi_period: int = 7) -> pd.DataFrame:
    c  = df["close"].astype(float)
    h  = df["high"].astype(float)
    l  = df["low"].astype(float)
    v  = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    d = pd.DataFrame(index=df.index)

    # RSI
    d["rsi7"]  = _rsi(c, 7)
    d["rsi14"] = _rsi(c, 14)

    # MACD histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = (macd - sig) / c.replace(0, np.nan) * 100

    # Bollinger %B
    sma20  = c.rolling(20).mean()
    std20  = c.rolling(20).std()
    bb_up  = sma20 + 2 * std20
    bb_lo  = sma20 - 2 * std20
    d["bb_pct"] = (c - bb_lo) / (bb_up - bb_lo).replace(0, np.nan)

    # ATR %
    atr14_s  = _atr(df, 14)
    d["atr_pct"] = atr14_s / c.replace(0, np.nan) * 100

    # Volume ratio (5-day avg vs 20-day avg)
    v_avg5  = v.rolling(5).mean()
    v_avg20 = v.rolling(20).mean()
    d["vol_ratio"] = v_avg5 / v_avg20.replace(0, np.nan)

    # EMA distance %
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    d["ema9_dist"]  = (c - ema9)  / c.replace(0, np.nan) * 100
    d["ema20_dist"] = (c - ema20) / c.replace(0, np.nan) * 100
    d["ema50_dist"] = (c - ema50) / c.replace(0, np.nan) * 100

    # Returns
    d["ret1d"]  = c.pct_change(1)
    d["ret5d"]  = c.pct_change(5)
    d["ret20d"] = c.pct_change(20)

    # 52-week position
    hi52 = c.rolling(252, min_periods=50).max()
    lo52 = c.rolling(252, min_periods=50).min()
    d["high52w_dist"] = (hi52 - c) / c.replace(0, np.nan) * 100
    d["low52w_dist"]  = (c - lo52) / c.replace(0, np.nan) * 100

    # Vol continuity
    atr5  = _atr(df, 5)
    atr20 = _atr(df, 20)
    d["vol_cont"] = (atr5 / atr20.replace(0, np.nan)).clip(0, 3)

    # CLV (close location value)
    denom = (h - l).replace(0, np.nan)
    d["clv"] = ((c - l) - (h - c)) / denom

    d["close"] = c
    return d


def _train_ensemble(sym: str, df: pd.DataFrame, horizon: int = 5) -> dict:
    """Stacked ensemble: RF + GBM base → Logistic meta-learner, walk-forward CV."""
    try:
        from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                      StackingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                                     precision_score, recall_score, f1_score)

        feat_df = _build_features(df, _STATE.get("rsi_period", 7))
        feat_df["fwd_ret"] = feat_df["close"].pct_change(horizon).shift(-horizon)
        feat_df["target"]  = (feat_df["fwd_ret"] > 0.005).astype(int)
        feat_df = feat_df.dropna(subset=ML_FEATURE_COLS + ["target"])

        if len(feat_df) < 80:
            return {"error": f"Insufficient history ({len(feat_df)} bars, need 80+)"}

        X = feat_df[ML_FEATURE_COLS].values
        y = feat_df["target"].values

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=4)
        fold_metrics = []
        for tr_idx, val_idx in tscv.split(X_sc):
            base = [
                ("rf",  RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42, n_jobs=1)),
                ("gbm", GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)),
            ]
            stk = StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(C=1.0, random_state=42),
                cv=3, passthrough=False,
            )
            stk.fit(X_sc[tr_idx], y[tr_idx])
            preds = stk.predict(X_sc[val_idx])
            proba = stk.predict_proba(X_sc[val_idx])[:, 1]
            fold_metrics.append({
                "acc":       float(accuracy_score(y[val_idx], preds)),
                "auc":       float(roc_auc_score(y[val_idx], proba)) if len(set(y[val_idx])) > 1 else 0.5,
                "logloss":   float(log_loss(y[val_idx], proba)),
                "precision": float(precision_score(y[val_idx], preds, zero_division=0)),
                "recall":    float(recall_score(y[val_idx], preds, zero_division=0)),
                "f1":        float(f1_score(y[val_idx], preds, zero_division=0)),
            })

        # Final model on all data
        base_final = [
            ("rf",  RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)),
            ("gbm", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ]
        final = StackingClassifier(
            estimators=base_final,
            final_estimator=CalibratedClassifierCV(LogisticRegression(C=1.0, random_state=42), cv=3),
            cv=5, passthrough=False,
        )
        final.fit(X_sc, y)

        # Feature importances from RF base
        rf_model = final.estimators_[0]
        importances = dict(zip(ML_FEATURE_COLS, rf_model.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: -x[1])[:8]

        # Predict on latest bar
        latest_feat = feat_df[ML_FEATURE_COLS].iloc[-1:].values
        latest_sc   = scaler.transform(latest_feat)
        prob_up     = float(final.predict_proba(latest_sc)[0][1])
        direction   = "BUY" if prob_up > 0.55 else ("SELL" if prob_up < 0.45 else "NEUTRAL")
        confidence  = "HIGH" if abs(prob_up - 0.5) > 0.15 else ("MEDIUM" if abs(prob_up - 0.5) > 0.08 else "LOW")

        # Price targets via ATR
        ltp = float(df["close"].iloc[-1])
        atr = float(_atr(df, 14).iloc[-1])
        mul = 2.0 if prob_up > 0.65 else (1.5 if prob_up > 0.55 else 1.0)
        target_price = round(ltp + mul * atr, 2)
        stop_price   = round(ltp - 1.0 * atr, 2)

        # Avg fold metrics
        avg = lambda k: round(float(np.mean([m[k] for m in fold_metrics])) * 100, 1)

        result = {
            "symbol":       sym,
            "direction":    direction,
            "prob_up":      round(prob_up * 100, 1),
            "confidence":   confidence,
            "horizon_days": horizon,
            "ltp":          round(ltp, 2),
            "ml_target":    target_price,
            "ml_stop":      stop_price,
            "top_features": top_features,
            "trained_bars": len(feat_df),
            "model":        "StackedEnsemble(RF+GBM → LogisticMeta, calibrated)",
            "validation":   "4-fold WalkForward TimeSeriesSplit",
            "cv_accuracy":  avg("acc"),
            "cv_auc":       avg("auc"),
            "cv_logloss":   round(float(np.mean([m["logloss"] for m in fold_metrics])), 4),
            "cv_precision": avg("precision"),
            "cv_recall":    avg("recall"),
            "cv_f1":        avg("f1"),
            "fold_metrics": fold_metrics,
            "feature_importances": importances,
            "trained_at":   time.time(),
        }
        _ml_models[sym] = result
        return result

    except ImportError:
        return {"error": "scikit-learn not installed — run: pip install scikit-learn"}
    except Exception as e:
        return {"error": str(e)}


def _fetch_yf_ohlcv(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance as fallback when Upstox cache is empty."""
    try:
        import yfinance as yf
        yf_sym = symbol + ".NS"
        df = yf.download(yf_sym, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        return df
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@router.get("/{symbol}")
async def ml_predict(symbol: str, horizon: int = Query(5, ge=1, le=30), refresh: bool = Query(False)):
    """Train stacked ML ensemble and return prediction for a symbol."""
    sym = symbol.upper()

    # Use cached if recent (< 4 hours old and same horizon)
    cached = _ml_models.get(sym)
    if not refresh and cached and cached.get("horizon_days") == horizon:
        age = time.time() - cached.get("trained_at", 0)
        if age < 14400:
            return cached

    # Get OHLCV data — prefer Upstox cache, fall back to yfinance
    df = _STATE.get("raw_data_cache", {}).get(sym)
    if df is None or (hasattr(df, "__len__") and len(df) < 80):
        df = _fetch_yf_ohlcv(sym)
    if df is None:
        raise HTTPException(404, f"No OHLCV data for {sym}. Run extraction first or check the ticker.")

    result = _train_ensemble(sym, df, horizon)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@router.delete("/{symbol}/cache")
async def clear_ml_cache(symbol: str):
    _ml_models.pop(symbol.upper(), None)
    return {"status": "cleared", "symbol": symbol.upper()}


@router.get("/batch/predict")
async def batch_predict(symbols: str = Query(..., description="Comma-separated symbols"),
                        horizon: int = Query(5, ge=1, le=30)):
    """Predict for multiple symbols. Returns summary list."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(syms) > 20:
        raise HTTPException(400, "Max 20 symbols per batch")
    results = []
    for sym in syms:
        df = _STATE.get("raw_data_cache", {}).get(sym) or _fetch_yf_ohlcv(sym)
        if df is None or len(df) < 80:
            results.append({"symbol": sym, "error": "no data"})
            continue
        cached = _ml_models.get(sym)
        if cached and cached.get("horizon_days") == horizon and (time.time() - cached.get("trained_at", 0)) < 14400:
            results.append({k: cached[k] for k in ["symbol","direction","prob_up","confidence","ltp","ml_target","ml_stop","cv_accuracy"]})
            continue
        r = _train_ensemble(sym, df, horizon)
        if "error" not in r:
            results.append({k: r[k] for k in ["symbol","direction","prob_up","confidence","ltp","ml_target","ml_stop","cv_accuracy"]})
        else:
            results.append({"symbol": sym, "error": r["error"]})
    return {"results": results, "horizon": horizon}
