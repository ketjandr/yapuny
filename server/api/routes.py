from __future__ import annotations

import uuid

import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from server.api.schemas import (
    BenchRunRequest,
    DecodeRequest,
    GenerateRequest,
    GraphRequest,
    PrepareDataRequest,
    SaveGraphRequest,
    TrainRequest,
)
from worker.worker import Worker

router = APIRouter(prefix="/api")
worker = Worker()

# in-memory stores (swap for DB later)
_graphs: dict[str, dict] = {}
_bench_runs: dict[str, dict] = {}


@router.post("/graph")
def save_graph(request: SaveGraphRequest):
    graph_data = request.model_dump()
    graph_id = request.id
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
def prepare_data(request: PrepareDataRequest = PrepareDataRequest()):
    result = worker.prepare_data(
        vocab_size=request.vocab_size,
        val_fraction=request.val_fraction,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/validate")
def validate_graph(request: GraphRequest):
    from server.compiler.validator import GraphValidator
    from server.models.graph import GraphSpec

    graph = GraphSpec.from_dict(request.model_dump())
    result = GraphValidator().validate(graph)
    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@router.post("/compile")
def compile_graph(request: GraphRequest):
    try:
        result = worker.compile_graph(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.get("/fusion/available")
def fusion_available():
    from server.compiler.fusion_registry import FUSION_AVAILABLE, FUSION_REGISTRY

    return {
        "available": FUSION_AVAILABLE,
        "patterns": [{"nodes": list(f.pattern), "kernel": f.cls.__name__} for f in FUSION_REGISTRY],
    }


@router.post("/fusion/suggest")
def suggest_fusions(request: GraphRequest):
    from server.compiler.fusion_registry import FUSION_AVAILABLE, detect_fusion_groups
    from server.compiler.utils import topo_sort
    from server.models.graph import GraphSpec

    if not FUSION_AVAILABLE:
        return {"available": False, "suggestions": []}

    graph = GraphSpec.from_dict(request.model_dump())
    topo_order = topo_sort(graph)
    node_types = {n.id: n.type for n in graph.nodes}
    edges = [(e.from_node, e.from_port, e.to_node, e.to_port) for e in graph.edges]

    groups = detect_fusion_groups(topo_order, node_types, edges)

    return {
        "available": True,
        "suggestions": [{"nodes": nids, "kernel": fdef.cls.__name__} for nids, fdef in groups],
    }


@router.post("/generate")
def generate(request: GenerateRequest):
    result = worker.generate(
        prompt_ids=request.prompt_ids,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        bench=request.bench,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/decode")
def decode_tokens(request: DecodeRequest):
    result = worker.decode_tokens(request.token_ids)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/graph")
def list_graphs():
    from server.compiler.utils import graph_structure_hash
    from server.models.graph import GraphSpec

    results = []
    for graph_id, data in _graphs.items():
        graph = GraphSpec.from_dict(data)
        results.append(
            {
                "id": graph_id,
                "meta": data.get("meta", {}),
                "structure_hash": graph_structure_hash(graph),
            }
        )
    return {"graphs": results}


@router.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    if worker.model is None:
        raise HTTPException(status_code=400, detail="no model compiled")
    if worker.training:
        raise HTTPException(status_code=409, detail="training already in progress")

    background_tasks.add_task(
        worker.train,
        max_steps=request.max_steps,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        eval_interval=request.eval_interval,
        eval_iters=request.eval_iters,
        checkpoint_path=request.checkpoint_path,
        bench=request.bench,
    )

    return {"status": "started", "max_steps": request.max_steps}


@router.get("/train/status")
def train_status():
    return worker.get_train_status()


@router.post("/train/stop")
def stop_training():
    result = worker.stop_training()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# -- Benchmark --


def _run_bench(run_id: str, request: BenchRunRequest):
    from server.compiler.compiler import GraphCompiler
    from server.compiler.utils import graph_structure_hash
    from server.models.graph import GraphSpec
    from worker.bench import _collect_env

    _bench_runs[run_id]["status"] = "running"
    original_model = worker.model

    try:
        compiler = GraphCompiler()
        device = worker.device
        weight_cache: dict[str, dict] = {}

        results = {}

        for graph_id in request.graph_ids:
            if graph_id not in _graphs:
                _bench_runs[run_id] = {
                    "status": "error",
                    "error": f"graph '{graph_id}' not found",
                }
                return

            graph = GraphSpec.from_dict(_graphs[graph_id])
            s_hash = graph_structure_hash(graph)

            if s_hash not in weight_cache:
                pretrained = worker._weight_store.get(s_hash)
                if pretrained:
                    weight_cache[s_hash] = pretrained

            pretrained = weight_cache.get(s_hash)
            if pretrained:
                model = compiler.compile(graph, pretrained_state=pretrained)
            else:
                model = compiler.compile(graph)

            model.to(device)
            model.eval()

            # warmup pass to avoid Triton JIT skew
            warmup_idx = torch.tensor([request.prompt_ids], device=device)
            block_size = model.meta["block_size"]
            with torch.no_grad():
                model(warmup_idx[:, -block_size:])
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            worker.model = model
            results[graph_id] = worker.generate(
                prompt_ids=request.prompt_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                bench=True,
            )

        worker.model = original_model

        _bench_runs[run_id] = {
            "status": "complete",
            "result": {
                "graphs": results,
                "env": _collect_env(device),
            },
        }

    except Exception as e:
        worker.model = original_model
        _bench_runs[run_id] = {"status": "error", "error": str(e)}


@router.post("/bench/generate")
def start_bench(request: BenchRunRequest, background_tasks: BackgroundTasks):
    run_id = uuid.uuid4().hex[:12]
    _bench_runs[run_id] = {"status": "pending"}
    background_tasks.add_task(_run_bench, run_id, request)
    return {"run_id": run_id, "status": "pending"}


@router.get("/bench/{run_id}")
def get_bench(run_id: str):
    if run_id not in _bench_runs:
        raise HTTPException(status_code=404, detail="run not found")
    return _bench_runs[run_id]
