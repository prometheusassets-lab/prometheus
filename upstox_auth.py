"""
upstox_auth.py — Upstox OAuth 2.0 flow for MONARCH PRO
Mirrors the Home.py login flow from the Streamlit version.

Flow:
  1.  GET  /auth/login         → redirects user to Upstox OAuth consent page
  2.  GET  /auth/callback      → Upstox redirects here with ?code=XXX
                                  exchanges code for access_token, saves it
  3.  GET  /auth/status        → JSON { connected, prefix }
  4.  POST /auth/logout        → clears token
  5.  POST /auth/token         → manual paste (fallback, same as /api/token)
"""

import os, requests
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

router = APIRouter()

# ── Upstox OAuth credentials ─────────────────────────────────────────────────
# Priority: config.py values > environment variables > placeholder defaults
# Edit config.py (same folder) to set your credentials.
try:
    import config as _cfg
    CLIENT_ID     = _cfg.CLIENT_ID
    CLIENT_SECRET = _cfg.CLIENT_SECRET
    REDIRECT_URI  = _cfg.REDIRECT_URI
    TOKEN_FILE    = _cfg.TOKEN_FILE
except ImportError:
    CLIENT_ID     = os.environ.get("UPSTOX_CLIENT_ID",     "YOUR_CLIENT_ID_HERE")
    CLIENT_SECRET = os.environ.get("UPSTOX_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
    REDIRECT_URI  = os.environ.get("UPSTOX_REDIRECT_URI",  "http://localhost:8000/auth/callback")
    TOKEN_FILE    = ".upstox_token"

# Shared state reference — injected by main.py after import
_STATE: dict = {}

def init_state(state_dict: dict):
    """Called by main.py to share the global STATE dict."""
    global _STATE
    _STATE = state_dict
    # Auto-load saved token on startup
    if not _STATE.get("token"):
        try:
            if os.path.exists(TOKEN_FILE):
                t = open(TOKEN_FILE).read().strip()
                if t:
                    _STATE["token"] = t
        except Exception:
            pass


def _save_token(tok: str):
    _STATE["token"] = tok
    try:
        with open(TOKEN_FILE, "w") as f:
            f.write(tok)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/auth/login")
async def upstox_login():
    """Redirect user to Upstox OAuth page."""
    auth_url = (
        "https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code"
        f"&client_id={CLIENT_ID}"
        f"&redirect_uri={requests.utils.quote(REDIRECT_URI, safe='')}"
    )
    return RedirectResponse(auth_url)


@router.get("/auth/callback", response_class=HTMLResponse)
async def upstox_callback(request: Request):
    """
    Upstox redirects here after the user logs in via OTP.
    Exchange the auth code for an access token, then redirect to /.
    """
    code  = request.query_params.get("code", "")
    error = request.query_params.get("error", "")

    if error or not code:
        return HTMLResponse(_error_page(
            f"Upstox login failed: {error or 'no code returned'}. "
            "Close this tab and try again."
        ), status_code=400)

    # Exchange code → access token
    try:
        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            data={
                "code":          code,
                "client_id":     CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uri":  REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
        data = resp.json()
    except Exception as e:
        return HTMLResponse(_error_page(f"Token exchange failed: {e}"), status_code=500)

    token = data.get("access_token", "")
    if not token:
        msg = data.get("message") or data.get("errors") or str(data)
        return HTMLResponse(_error_page(f"No access token in response: {msg}"), status_code=500)

    _save_token(token)

    # Redirect back to terminal with a success flash
    return HTMLResponse(_success_page())


@router.get("/auth/status")
async def auth_status():
    tok = _STATE.get("token", "")
    return {"connected": bool(tok), "prefix": tok[:16] if tok else ""}


@router.post("/auth/logout")
async def auth_logout():
    _STATE["token"] = ""
    try:
        os.remove(TOKEN_FILE)
    except Exception:
        pass
    return {"status": "logged_out"}


@router.get("/auth/config-status")
async def config_status():
    """Tell the frontend whether CLIENT_ID has been configured."""
    return {"configured": CLIENT_ID not in ("", "YOUR_CLIENT_ID_HERE")}


@router.post("/auth/token")
async def manual_token(body: dict):
    """Fallback: paste token manually (identical to /api/token)."""
    tok = body.get("token", "").strip()
    if not tok:
        from fastapi import HTTPException
        raise HTTPException(400, "Empty token")
    _save_token(tok)
    return {"status": "ok", "token_prefix": tok[:16]}


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_style():
    return """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0a; color: #e8e8e8;
    font-family: 'IBM Plex Mono', monospace;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #111; border: 1px solid #2a2a2a;
    border-top: 3px solid #ff8c00;
    padding: 36px 40px; max-width: 420px; width: 90%;
    text-align: center;
  }
  .logo { color: #ff8c00; font-size: 0.72rem; font-weight: 700;
          letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 6px; }
  h1   { color: #ff8c00; font-size: 1.1rem; font-weight: 700;
         letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 20px; }
  p    { color: #888; font-size: 0.70rem; line-height: 1.6; margin-bottom: 14px; }
  .green { color: #00d084; }
  .red   { color: #ff3b3b; }
  a    { color: #ff8c00; text-decoration: none; font-size: 0.70rem; }
  a:hover { text-decoration: underline; }
  .btn {
    display: inline-block; margin-top: 18px;
    background: #140e00; color: #ff8c00;
    border: 1px solid #ff8c00; padding: 8px 24px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    cursor: pointer; text-decoration: none;
  }
  .btn:hover { background: #ff8c00; color: #000; text-decoration: none; }
  .spinner {
    width: 28px; height: 28px; border: 2px solid #2a2a2a;
    border-top-color: #ff8c00; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 16px auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
"""


def _success_page():
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>MONARCH PRO — Connected</title>
{_base_style()}
</head><body>
<div class="card">
  <div class="logo">◼ MONARCH PRO</div>
  <h1>✔ Connected</h1>
  <p class="green">Upstox access token saved successfully.<br>
  You are now connected to live market data.</p>
  <p>Redirecting to terminal…</p>
  <div class="spinner"></div>
  <a href="/" class="btn">Open Terminal →</a>
</div>
<script>setTimeout(()=>location.href='/', 1800);</script>
</body></html>"""


def _error_page(msg: str):
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title>MONARCH PRO — Login Error</title>
{_base_style()}
</head><body>
<div class="card">
  <div class="logo">◼ MONARCH PRO</div>
  <h1 style="color:#ff3b3b">✘ Login Failed</h1>
  <p class="red">{msg}</p>
  <a href="/auth/login" class="btn">Try Again</a>
  <br>
  <a href="/" style="display:block;margin-top:14px">← Back to Terminal</a>
</div>
</body></html>"""