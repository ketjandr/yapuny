"""Session gateway for the shared worker.

The worker (server.app) is single-tenant, so instead of making it multi-tenant we run one
throwaway worker subprocess per browser session and proxy to it. Each session is identified by the
`X-Yapuny-Session` header the frontend sends.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

MAX_WORKERS = int(os.environ.get("GW_MAX_WORKERS", "6"))  # cap by RAM on the box
MAX_PER_IP = int(os.environ.get("GW_MAX_PER_IP", "2"))  # concurrent sessions per client IP
IDLE_TIMEOUT = float(os.environ.get("GW_IDLE_TIMEOUT", "600"))  # reap a session after N idle seconds
READY_TIMEOUT = float(os.environ.get("GW_READY_TIMEOUT", "60"))  # worker startup budget
DEVICE_LABEL = os.environ.get("GW_DEVICE_LABEL", "cpu")  # shown in the frontend indicator
ORIGINS = [o.strip() for o in os.environ.get("FRONTEND_ORIGIN", "http://localhost:4173").split(",") if o.strip()]
HOST = "127.0.0.1"

# Compute caps injected into each child worker (it clamps steps/batch, rejects oversized dims).
CHILD_CAPS = {
    "YAPUNY_MAX_STEPS": os.environ.get("GW_MAX_STEPS", "20000"),
    "YAPUNY_MAX_BATCH": os.environ.get("GW_MAX_BATCH", "512"),
    "YAPUNY_MAX_N_EMBD": os.environ.get("GW_MAX_N_EMBD", "1024"),
    "YAPUNY_MAX_BLOCK": os.environ.get("GW_MAX_BLOCK", "1024"),
    "YAPUNY_MAX_VOCAB": os.environ.get("GW_MAX_VOCAB", "50000"),
    "YAPUNY_MAX_LAYERS": os.environ.get("GW_MAX_LAYERS", "16"),
}

# hop-by-hop headers not forwarded through a proxy
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length", "origin",
}


class Session:
    def __init__(self, proc: subprocess.Popen, port: int, models_dir: str, ip: str):
        self.proc = proc
        self.port = port
        self.models_dir = models_dir
        self.ip = ip
        self.last_seen = time.time()


sessions: dict[str, Session] = {}
spawn_lock = asyncio.Lock()
client = httpx.AsyncClient(timeout=None)  # no timeout: training/generation streams run for minutes


def _free_port() -> int:
    s = socket.socket()
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _pump(r: httpx.Response, sess: Session):
    # bump last_seen on every chunk so a long stream (a training run) counts as active and the
    # reaper doesn't kill it mid-flight; without this last_seen is only the request-start time
    try:
        async for chunk in r.aiter_raw():
            sess.last_seen = time.time()
            yield chunk
    finally:
        await r.aclose()


def _cleanup(sess: Session) -> None:
    with contextlib.suppress(Exception):
        sess.proc.terminate()
    shutil.rmtree(sess.models_dir, ignore_errors=True)


async def _spawn(session_id: str, ip: str) -> Session:
    if len(sessions) >= MAX_WORKERS:
        raise HTTPException(503, "shared worker is at capacity - try again shortly, or connect your own worker")
    if sum(1 for s in sessions.values() if s.ip == ip) >= MAX_PER_IP:
        raise HTTPException(
            429, "too many sessions from your network - close a tab, or connect your own worker"
        )
    port = _free_port()
    models_dir = tempfile.mkdtemp(prefix="yapuny-sess-")
    # per-session data dir (nested so one rmtree cleans it): isolates each session's corpus, and the
    # worker seeds it with the bundled default corpus on startup. CHILD_CAPS bound its compute.
    env = {
        **os.environ,
        "YAPUNY_MODELS_DIR": models_dir,
        "YAPUNY_DATA_DIR": os.path.join(models_dir, "data"),
        **CHILD_CAPS,
    }
    env.pop("WORKER_TOKEN", None)  # localhost child - the gateway is the trust boundary
    env.pop("FRONTEND_ORIGIN", None)  # server-side calls, no browser CORS
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", HOST, "--port", str(port)],
        env=env,
    )
    sess = Session(proc, port, models_dir, ip)
    # wait until the worker answers /health
    deadline = time.time() + READY_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            _cleanup(sess)
            raise HTTPException(502, "worker process exited on startup")
        try:
            r = await client.get(f"http://{HOST}:{port}/health")
            if r.status_code == 200:
                sessions[session_id] = sess
                return sess
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.25)
    _cleanup(sess)
    raise HTTPException(504, "worker start timed out")


async def _get_or_spawn(session_id: str, ip: str) -> Session:
    sess = sessions.get(session_id)
    if sess and sess.proc.poll() is None:
        sess.last_seen = time.time()
        return sess
    async with spawn_lock:  # serialize spawns (avoids port races / thundering herd)
        sess = sessions.get(session_id)
        if sess and sess.proc.poll() is None:
            sess.last_seen = time.time()
            return sess
        return await _spawn(session_id, ip)


async def _reaper() -> None:
    while True:
        await asyncio.sleep(30)
        now = time.time()
        for sid, sess in list(sessions.items()):
            if sess.proc.poll() is not None or now - sess.last_seen > IDLE_TIMEOUT:
                sessions.pop(sid, None)
                _cleanup(sess)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_reaper())
    yield
    task.cancel()
    for sess in list(sessions.values()):
        _cleanup(sess)
    await client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/gwhealth")
def gwhealth():
    return {"status": "ok", "sessions": len(sessions), "max": MAX_WORKERS}


@app.get("/api/health")
def health():
    # answered by the gateway itself, so idle browsing / the connect check don't spawn a worker;
    # a session's worker is spawned lazily on its first real /api call
    return {"status": "ok", "device": DEVICE_LABEL, "gpu": None, "shared": True}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    session_id = request.headers.get("x-yapuny-session")
    if not session_id:
        raise HTTPException(400, "missing session id")
    # client IP for the per-IP cap: X-Forwarded-For (Caddy sets it), else the direct peer
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = fwd or (request.client.host if request.client else "unknown")
    sess = await _get_or_spawn(session_id, ip)

    url = f"http://{HOST}:{sess.port}/api/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    body = await request.body()
    upstream = client.build_request(request.method, url, headers=headers, content=body, params=request.query_params)
    r = await client.send(upstream, stream=True)  # stream=True so SSE flows through unbuffered
    sess.last_seen = time.time()
    resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP}
    return StreamingResponse(
        _pump(r, sess),  # _pump refreshes last_seen per chunk and closes r when done
        status_code=r.status_code,
        headers=resp_headers,
    )
