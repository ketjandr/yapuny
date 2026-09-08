import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router

app = FastAPI(title="Yapuny", version="0.1.0")

# CORS for the direct browser->worker path (a self-hosted worker).
_origins_env = os.environ.get("FRONTEND_ORIGIN", "").strip()
if _origins_env:
    _allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    _allow_regex = None
else:
    _allow_origins = ["http://localhost:5173", "http://localhost:4173", "https://yapuny.vercel.app"]
    _allow_regex = r"https://yapuny-[a-z0-9-]+\.vercel\.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_origin_regex=_allow_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
