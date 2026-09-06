from __future__ import annotations

import json
import threading
import time

import torch
from fastapi import APIRouter, HTTPException, UploadFile
from starlette.responses import StreamingResponse

from server.api.schemas import (
    BenchRunRequest,
    CorpusSaveRequest,
    DecodeRequest,
    GenerateRequest,
    GraphRequest,
    ModelGraphRequest,
    ProfileRequest,
    TrainBenchRequest,
    TrainRequest,
)
from worker.worker import Worker

router = APIRouter(prefix="/api")
worker = Worker()


# -- Graph --


@router.post("/graph/validate", tags=["graph"])
def validate_graph(request: GraphRequest):
    from server.compiler.utils import expand_blocks
    from server.compiler.validator import GraphValidator
    from server.models.graph import GraphSpec

    graph = expand_blocks(GraphSpec.from_dict(request.model_dump()))
    result = GraphValidator().validate(graph)
    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@router.post("/graph/compile", tags=["graph"])
def compile_graph(request: ModelGraphRequest):
    try:
        result = worker.compile_model(request.id, request.graph.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


# -- Model (status / list / delete) --


@router.post("/model/status", tags=["model"])
def model_status(request: ModelGraphRequest):
    return worker.model_status(request.id, request.graph.model_dump())


@router.get("/models", tags=["model"])
def list_models():
    return {"models": worker.list_models()}


@router.delete("/model/{model_id}", tags=["model"])
def delete_model(model_id: str):
    result = worker.delete_model(model_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="model not found")
    return result


# -- Fusion --


@router.get("/fusion/available", tags=["fusion"])
def fusion_available():
    from server.compiler.fusion_registry import FUSION_AVAILABLE, FUSION_REGISTRY

    return {
        "available": FUSION_AVAILABLE,
        "patterns": [{"nodes": list(f.pattern), "kernel": f.cls.__name__} for f in FUSION_REGISTRY],
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


# -- Quantization --


@router.get("/quantization/available", tags=["quantization"])
def quantization_available():
    from server.compiler.quantization_registry import (
        QUANT_MODES,
        QUANTIZABLE_NODES,
        QUANTIZATION_AVAILABLE,
    )

    return {
        "available": QUANTIZATION_AVAILABLE,
        "modes": sorted(QUANT_MODES),
        "node_types": sorted(QUANTIZABLE_NODES),
    }


# -- Data --


@router.get("/data/status", tags=["data"])
def data_status():
    from worker.worker import RAW_DIR

    corpus_path = RAW_DIR / "corpus.txt"
    has_corpus = corpus_path.exists()

    return {
        "corpus_uploaded": has_corpus,
        "corpus_bytes": corpus_path.stat().st_size if has_corpus else None,
    }


@router.delete("/data/corpus", tags=["data"])
def delete_corpus():
    from worker.worker import RAW_DIR

    corpus_path = RAW_DIR / "corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail="no corpus uploaded")

    corpus_path.unlink()
    return {"status": "deleted"}


@router.post("/data/upload", tags=["data"])
async def upload_corpus(file: UploadFile):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="only .txt files accepted")

    content = await file.read()
    result = worker.upload_corpus(content, file.filename)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/data/corpus", tags=["data"])
def get_corpus():
    from worker.worker import RAW_DIR

    corpus_path = RAW_DIR / "corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail="no corpus uploaded")

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    return {
        "text": text,
        "size_bytes": corpus_path.stat().st_size,
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }


@router.post("/data/corpus", tags=["data"])
def save_corpus(request: CorpusSaveRequest):
    # write edited/new corpus text back to disk (autosave target). Reuses the upload path so the
    # size cap and stats are identical whether text arrives from a file upload or the editor.
    result = worker.upload_corpus(request.text.encode("utf-8"), "corpus.txt")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# -- Generate --


@router.post("/generate", tags=["generate"])
def generate(request: GenerateRequest):
    try:
        prompt_ids = worker.encode_prompt(request.id, request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = worker.generate(
        model_id=request.id,
        prompt_ids=prompt_ids,
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
    try:
        prompt_ids = worker.encode_prompt(request.id, request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def event_stream():
        for event in worker.generate_stream(
            model_id=request.id,
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            bench=request.bench,
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/decode", tags=["generate"])
def decode_tokens(request: DecodeRequest):
    result = worker.decode_tokens(request.id, request.token_ids)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# -- Train --


def _start_train(request: TrainRequest):
    if request.id not in worker.cache:
        raise HTTPException(status_code=400, detail="model not compiled - compile first")
    if worker.training:
        raise HTTPException(status_code=409, detail="training already in progress")

    # seed a fresh running state before streaming so the SSE loop neither spins on a None state nor
    # replays a previous run's terminal frame; the worker overwrites this once it starts stepping
    worker.train_state = {
        "step": 0,
        "max_steps": request.max_steps,
        "train_loss": None,
        "status": "running",
    }

    threading.Thread(
        target=worker.train,
        args=(request.id,),
        kwargs=dict(
            max_steps=request.max_steps,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            bench=request.bench,
        ),
        daemon=True,
    ).start()


@router.post("/train", tags=["train"])
def train(request: TrainRequest):
    _start_train(request)
    return {"status": "started", "max_steps": request.max_steps}


# poll a run's state dict and stream it as SSE `update` frames until it reaches a terminal status.
# `get_state` returns the current dict (or None when there's nothing to follow). Shared by the
# single (train_state) and multi-model bench (bench_state) runs, and their start + follow endpoints.
def _poll_state_stream(get_state):
    prev = None
    while True:
        state = get_state()
        if state is None:
            return  # nothing (more) to follow
        serialized = json.dumps(state)
        if serialized != prev:
            prev = serialized
            yield f"event: update\ndata: {serialized}\n\n"
        if state.get("status") in ("completed", "stopped", "error"):
            return
        time.sleep(0.25)


def _train_stream_resp():
    stream = _poll_state_stream(lambda: worker.train_state)
    return StreamingResponse(stream, media_type="text/event-stream")


def _bench_stream_resp():
    stream = _poll_state_stream(lambda: worker.bench_state)
    return StreamingResponse(stream, media_type="text/event-stream")


@router.post("/train/stream", tags=["train"])
def train_stream(request: TrainRequest):
    _start_train(request)  # seeds a running train_state, so the stream never sees None
    return _train_stream_resp()


@router.get("/train/follow", tags=["train"])
def train_follow():
    # reattach to a run already in progress (started by another client / a since-reloaded page).
    # Does NOT start training - if nothing is running the stream just closes immediately.
    return _train_stream_resp()


@router.get("/train/status", tags=["train"])
def train_status():
    return worker.get_train_status()


@router.post("/train/bench", tags=["train"])
def train_bench(request: TrainBenchRequest):
    if worker.training:
        raise HTTPException(status_code=409, detail="training already in progress")
    for mid in request.model_ids:
        if mid not in worker.cache:
            raise HTTPException(status_code=400, detail="compile all selected models first")

    # seed a running bench_state before streaming so the poller never sees None or a stale run
    worker.bench_state = {
        "status": "running",
        "current": 0,
        "models": [
            {"id": mid, "status": "pending", "step": 0, "max_steps": request.max_steps,
             "train_loss": None, "steps_per_sec": None, "bench": None}
            for mid in request.model_ids
        ],
    }
    threading.Thread(
        target=worker.train_bench,
        args=(request.model_ids,),
        kwargs=dict(
            max_steps=request.max_steps,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        ),
        daemon=True,
    ).start()
    return _bench_stream_resp()


@router.get("/train/bench/follow", tags=["train"])
def train_bench_follow():
    return _bench_stream_resp()


@router.get("/train/bench/status", tags=["train"])
def train_bench_status():
    return worker.get_bench_status()


@router.post("/train/stop", tags=["train"])
def stop_training():
    result = worker.stop_training()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# -- Benchmark --


@router.post("/bench/profile", tags=["benchmark"])
def bench_profile(request: ProfileRequest):
    from dataclasses import asdict

    if request.mode not in ("decode", "train"):
        raise HTTPException(status_code=400, detail="mode must be 'decode' or 'train'")

    result = worker.profile(
        model_id=request.id,
        mode=request.mode,
        prompt_tokens=request.prompt_tokens,
        new_tokens=request.new_tokens,
        warmup=request.warmup,
    )
    if result is None:
        raise HTTPException(status_code=400, detail="model not compiled - compile first")

    return {
        "nodes": [asdict(n) for n in result.nodes],
        "total_us": result.total_us,
        "env": result.env,
    }


def _bench_stream(request: BenchRunRequest):
    from data.tokenizer import encode
    from server.compiler.compiler import GraphCompiler
    from server.compiler.utils import graph_structure_hash
    from server.models.graph import GraphSpec
    from worker import store
    from worker.bench import _collect_env

    # bench sits on top of trained models: each entry loads its own package from the
    # locker (its own tokenizer + weights), so different vocabs coexist across graphs.
    compiler = GraphCompiler()
    device = worker.device

    try:
        for i, entry in enumerate(request.graphs):
            graph = GraphSpec.from_dict(entry.graph.model_dump())
            s_hash = graph_structure_hash(graph)

            pkg = store.load(entry.id)
            if pkg is None or pkg.structure_hash != s_hash:
                err = {"graph_idx": i, "error": "model not trained"}
                yield f"event: error\ndata: {json.dumps(err)}\n\n"
                continue

            model = compiler.compile(graph, pretrained_state=pkg.weights)
            model.to(device)
            model.eval()

            prompt_ids = encode(pkg.tokenizer, request.prompt)

            # warmup both prefill and decode kernels so Triton JIT doesn't corrupt timings
            for _ in worker._stream_tokens(model, pkg.tokenizer, prompt_ids, max_new_tokens=3):
                pass
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            yield f"event: graph_start\ndata: {json.dumps({'graph_idx': i})}\n\n"

            for event in worker._stream_tokens(
                model,
                pkg.tokenizer,
                prompt_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                bench=True,
            ):
                event["data"]["graph_idx"] = i
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

        yield f"event: done\ndata: {json.dumps({'env': _collect_env(device)})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@router.post("/bench/generate", tags=["benchmark"])
def bench_generate(request: BenchRunRequest):
    return StreamingResponse(_bench_stream(request), media_type="text/event-stream")
