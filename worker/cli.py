"""Console entry point for the worker: `yapuny` starts the FastAPI worker.

Installed as a script (see pyproject [project.scripts]) so `uv tool install` exposes a `yapuny`
command on PATH. Host/port come from env so a server deploy can override them.
"""

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Yapuny worker starting on http://localhost:{port}  (Ctrl+C to stop)")
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
