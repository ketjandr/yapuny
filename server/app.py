import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router

app = FastAPI(title="Yapuny", version="0.1.0")

# CORS: the static frontend (on a CDN) calls this worker directly in prod. Restrict to
# the frontend origin(s); never "*" with credentials. In dev the Vite proxy avoids CORS
# entirely, so this only matters for the direct browser->worker prod path.
# FRONTEND_ORIGIN may be a comma-separated list.
_origins = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
