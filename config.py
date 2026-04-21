import os

CLIENT_ID     = os.environ.get("UPSTOX_CLIENT_ID",     "2enter id here")
CLIENT_SECRET = os.environ.get("UPSTOX_CLIENT_SECRET", "enter secret key here")
REDIRECT_URI  = os.environ.get("UPSTOX_REDIRECT_URI",  "http://127.0.0.1:8001/auth/callback")
#----http://127.0.0.1:8001/auth/callback  put this url in create api/ redidirect url at upstock

PORT       = int(os.environ.get("MONARCH_PORT", 8001))
HOST       = os.environ.get("MONARCH_HOST", "0.0.0.0")
TOKEN_FILE = ".upstox_token"
