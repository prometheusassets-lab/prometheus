# MONARCH PRO — Full Stack Equity Terminal

Bloomberg terminal UI · FastAPI backend · Upstox OTP login · Options · Fundamentals · ML Predictor

---

## Pages

| URL | Page |
|-----|------|
| `/` | Main Screener Terminal |
| `/login` | Upstox OTP Login |
| `/options` | Options Intelligence (live chain, greeks, OI, strategies) |
| `/fundamentals` | Fundamental Research (valuation, profitability, growth, health) |
| `/ml` | ML Predictor (stacked ensemble, walk-forward CV, batch scan) |

---

## Quick Start

### Step 1 — Register Upstox app (one-time)

1. Go to https://account.upstox.com/developer/apps → Create App
2. Set **Redirect URI** to: `http://localhost:8001/auth/callback`
3. Copy your **API Key** and **Secret Key**

### Step 2 — Configure credentials

Open `config.py` and paste your keys:

```python
CLIENT_ID     = "your_api_key_here"
CLIENT_SECRET = "your_secret_key_here"
```

### Step 3 — Launch

**Windows:** Double-click `START_MONARCH.bat`

**Manual (any OS):**
```bash
pip install -r requirements.txt
python main.py
# Open: http://localhost:8001/login
```

### Step 4 — Login daily

1. Browser opens → `http://localhost:8001/login`
2. Click **LOGIN WITH UPSTOX OTP**
3. Enter OTP on Upstox page → redirected to terminal
4. Token saved to `.upstox_token`

---

## Architecture

```
monarch-pro/
├── main.py                    ← FastAPI app (scoring engine, screener API)
├── config.py                  ← credentials + port
├── upstox_auth.py             ← Upstox OAuth router (/auth/*)
├── requirements.txt
├── START_MONARCH.bat          ← Windows launcher
│
├── routers/                   ← Feature modules (each fully self-contained)
│   ├── options.py             ← /api/options/* — chain, greeks, OI, bias, strategies
│   ├── fundamentals.py        ← /api/fundamentals/* — yfinance, peer compare, scorecard
│   └── ml.py                  ← /api/ml/* — stacked ensemble, batch predict
│
└── static/
    ├── index.html             ← Screener terminal
    ├── login.html             ← OTP login page
    ├── monarch-nav.js         ← Shared navigation bar (auto-injected)
    └── pages/
        ├── options.html       ← Options Intelligence page
        ├── fundamentals.html  ← Fundamental Research page
        └── ml.html            ← ML Predictor page
```

---

## API Reference

### Screener
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/extract` | Start bulk data extraction |
| GET | `/api/screener` | Get scored stocks |
| GET | `/api/extraction/status` | Extraction progress |
| POST | `/api/refresh_live` | Force live price update |

### Options
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/options/expiries/{symbol}` | Available expiry dates |
| GET | `/api/options/chain/{symbol}?expiry=…` | Live chain with greeks |
| GET | `/api/options/analysis/{symbol}?expiry=…` | Full analysis (chain + bias + strategies) |
| GET | `/api/options/greeks/{symbol}?strike=…` | Greeks for specific strike |

### Fundamentals
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fundamentals/{symbol}` | Full fundamental data |
| GET | `/api/fundamentals/{symbol}/peers` | Sector peer comparison |
| GET | `/api/fundamentals/{symbol}/scorecard` | Graded scorecard |

### ML Predictor
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ml/{symbol}?horizon=5` | Train model & predict |
| GET | `/api/ml/batch/predict?symbols=…` | Batch predictions |
| DELETE | `/api/ml/{symbol}/cache` | Clear model cache |

### Auth
| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/login` | Redirect to Upstox OAuth |
| GET | `/auth/callback` | OAuth callback |
| GET | `/auth/status` | Connection status |
| POST | `/auth/logout` | Clear token |

---

## Notes

- Token expires daily at midnight IST — run `START_MONARCH.bat` each morning
- `.upstox_token` file is created in the same folder — do not commit to git
- Options & Fundamentals pages work with just the Upstox token (no extraction needed)
- ML Predictor works with either Upstox data cache OR Yahoo Finance fallback
- `scikit-learn` is required for ML predictions: `pip install scikit-learn`
