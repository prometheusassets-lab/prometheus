import os

CLIENT_ID     = os.environ.get("UPSTOX_CLIENT_ID",     "256bd63b-fe87-4040-868e-fab15a75fc87")
CLIENT_SECRET = os.environ.get("UPSTOX_CLIENT_SECRET", "k8qqvj8a3e")
REDIRECT_URI  = os.environ.get("UPSTOX_REDIRECT_URI",  "http://127.0.0.1:8001/auth/callback")

PORT       = int(os.environ.get("MONARCH_PORT", 8001))
HOST       = os.environ.get("MONARCH_HOST", "0.0.0.0")
TOKEN_FILE = ".upstox_token"