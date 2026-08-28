from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

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


@router.post("/data/upload")
async def upload_corpus(file: UploadFile):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="only .txt files accepted")

    content = await file.read()
    result = worker.upload_corpus(content, file.filename)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/data/prepare")
def prepare_data(request: dict = {}):
    vocab_size = request.get("vocab_size", 8000)
    val_fraction = request.get("val_fraction", 0.1)

    result = worker.prepare_data(vocab_size=vocab_size, val_fraction=val_fraction)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


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


@router.post("/train")
def train(request: dict, background_tasks: BackgroundTasks):
    max_steps = request.get("max_steps", 2000)
    batch_size = request.get("batch_size", 32)
    learning_rate = request.get("learning_rate", 3e-4)
    eval_interval = request.get("eval_interval", 200)
    eval_iters = request.get("eval_iters", 50)
    checkpoint_path = request.get("checkpoint_path")

    if worker.model is None:
        raise HTTPException(status_code=400, detail="no model compiled")
    if worker.training:
        raise HTTPException(status_code=409, detail="training already in progress")

    background_tasks.add_task(
        worker.train,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        checkpoint_path=checkpoint_path,
    )

    return {"status": "started", "max_steps": max_steps}


@router.get("/train/status")
def train_status():
    return worker.get_train_status()


@router.post("/train/stop")
def stop_training():
    result = worker.stop_training()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
