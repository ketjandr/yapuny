from __future__ import annotations

import json
import time

import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from starlette.responses import StreamingResponse

from server.api.schemas import (
    BenchRunRequest,
    DecodeRequest,
    GenerateRequest,
    GraphRequest,
    PrepareDataRequest,
    ProfileRequest,
    SaveGraphRequest,
    TrainRequest,
)
from worker.worker import Worker

router = APIRouter(prefix="/api")
worker = Worker()

_graphs: dict[str, dict] = {}


# -- Graph --


@router.get("/graph", tags=["graph"])
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


@router.post("/graph", tags=["graph"])
def save_graph(request: SaveGraphRequest):
    graph_data = request.model_dump()
    graph_id = request.id
    _graphs[graph_id] = graph_data
    return {"id": graph_id, "status": "saved"}


@router.get("/graph/{graph_id}", tags=["graph"])
def load_graph(graph_id: str):
    if graph_id not in _graphs:
        raise HTTPException(status_code=404, detail="graph not found")
    return _graphs[graph_id]


@router.post("/graph/validate", tags=["graph"])
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


@router.post("/graph/compile", tags=["graph"])
def compile_graph(request: GraphRequest):
    try:
        result = worker.compile_graph(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


# -- Fusion --


@router.get("/fusion/available", tags=["fusion"])
def fusion_available():
    from server.compiler.fusion_registry import FUSION_AVAILABLE, FUSION_REGISTRY

    return {
        "available": FUSION_AVAILABLE,
        "patterns": [
            {"nodes": list(f.pattern), "kernel": f.cls.__name__} for f in FUSION_REGISTRY
        ],
    }


@router.post("/fusion/suggest", tags=["fusion"])
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


# -- Data --


@router.post("/data/upload", tags=["data"])
async def upload_corpus(file: UploadFile):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="only .txt files accepted")

    content = await file.read()
    result = worker.upload_corpus(content, file.filename)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/data/prepare", tags=["data"])
def prepare_data(request: PrepareDataRequest = PrepareDataRequest()):
    result = worker.prepare_data(
        vocab_size=request.vocab_size,
        val_fraction=request.val_fraction,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# -- Generate --


@router.post("/generate", tags=["generate"])
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


@router.post("/generate/stream", tags=["generate"])
def generate_stream(request: GenerateRequest):
    def event_stream():
        for event in worker.generate_stream(
            prompt_ids=request.prompt_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            bench=request.bench,
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/decode", tags=["generate"])
def decode_tokens(request: DecodeRequest):
    result = worker.decode_tokens(request.token_ids)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# -- Train --


def _start_train(request: TrainRequest, background_tasks: BackgroundTasks):
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


@router.post("/train", tags=["train"])
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    _start_train(request, background_tasks)
    return {"status": "started", "max_steps": request.max_steps}


@router.post("/train/stream", tags=["train"])
def train_stream(request: TrainRequest, background_tasks: BackgroundTasks):
    _start_train(request, background_tasks)

    def event_stream():
        prev = None
        while True:
            state = worker.train_state
            if state is None:
                time.sleep(0.05)
                continue

            serialized = json.dumps(state)
            if serialized != prev:
                prev = serialized
                yield f"event: update\ndata: {serialized}\n\n"

            if state.get("status") in ("completed", "stopped", "error"):
                return

            time.sleep(0.25)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/train/status", tags=["train"])
def train_status():
    return worker.get_train_status()


@router.post("/train/stop", tags=["train"])
def stop_training():
    result = worker.stop_training()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# -- Benchmark --


@router.post("/bench/profile", tags=["benchmark"])
def bench_profile(request: ProfileRequest = ProfileRequest()):
    from dataclasses import asdict

    from worker.bench import profile_graph

    if worker.model is None:
        raise HTTPException(status_code=400, detail="no model compiled")

    if request.mode not in ("decode", "train"):
        raise HTTPException(status_code=400, detail="mode must be 'decode' or 'train'")

    result = profile_graph(
        model=worker.model,
        device=worker.device,
        mode=request.mode,
        prompt_tokens=request.prompt_tokens,
        new_tokens=request.new_tokens,
        warmup=request.warmup,
    )

    return {
        "nodes": [asdict(n) for n in result.nodes],
        "total_us": result.total_us,
        "env": result.env,
    }


def _bench_stream(request: BenchRunRequest):
    from server.compiler.compiler import GraphCompiler
    from server.compiler.utils import graph_structure_hash
    from server.models.graph import GraphSpec
    from worker.bench import _collect_env

    original_model = worker.model
    compiler = GraphCompiler()
    device = worker.device
    weight_cache: dict[str, dict] = {}

    try:
        for graph_id in request.graph_ids:
            if graph_id not in _graphs:
                err = {"error": f"graph {graph_id!r} not found"}
                yield f"event: error\ndata: {json.dumps(err)}\n\n"
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

            warmup_idx = torch.tensor([request.prompt_ids], device=device)
            block_size = model.meta["block_size"]
            with torch.no_grad():
                model(warmup_idx[:, -block_size:])
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            yield f"event: graph_start\ndata: {json.dumps({'graph_id': graph_id})}\n\n"

            worker.model = model
            for event in worker.generate_stream(
                prompt_ids=request.prompt_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                bench=True,
            ):
                event["data"]["graph_id"] = graph_id
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

        yield f"event: done\ndata: {json.dumps({'env': _collect_env(device)})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    finally:
        worker.model = original_model


@router.post("/bench/generate", tags=["benchmark"])
def bench_generate(request: BenchRunRequest):
    return StreamingResponse(_bench_stream(request), media_type="text/event-stream")
