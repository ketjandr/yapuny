from fastapi import APIRouter, HTTPException

from worker.worker import Worker

router = APIRouter(prefix="/api")
worker = Worker()

# in-memory graph store (swap for DB later)
_graphs: dict[str, dict] = {}


@router.post("/graph")
def save_graph(graph_data: dict):
    graph_id = graph_data.get("id", "default")
    _graphs[graph_id] = graph_data
    return {"id": graph_id, "status": "saved"}


@router.get("/graph/{graph_id}")
def load_graph(graph_id: str):
    if graph_id not in _graphs:
        raise HTTPException(status_code=404, detail="graph not found")
    return _graphs[graph_id]


@router.post("/compile")
def compile_graph(graph_data: dict):
    try:
        result = worker.compile_graph(graph_data)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"invalid graph format: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.post("/generate")
def generate(request: dict):
    prompt_ids = request.get("prompt_ids", [])
    max_new_tokens = request.get("max_new_tokens", 50)
    temperature = request.get("temperature", 1.0)
    top_k = request.get("top_k")

    if not prompt_ids:
        raise HTTPException(status_code=400, detail="prompt_ids required")

    result = worker.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
