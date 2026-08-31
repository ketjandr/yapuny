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
    ModelGraphRequest,
    PrepareDataRequest,
    ProfileRequest,
    TrainRequest,
)
from worker.worker import Worker

router = APIRouter(prefix="/api")
worker = Worker()


# -- Graph --


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
    from worker.worker import DATA_DIR, RAW_DIR, TOKENIZER_PATH

    corpus_path = RAW_DIR / "corpus.txt"
    has_corpus = corpus_path.exists()
    has_tokenizer = TOKENIZER_PATH.exists()
    has_train = (DATA_DIR / "train.bin").exists()
    has_val = (DATA_DIR / "val.bin").exists()

    return {
        "corpus_uploaded": has_corpus,
        "corpus_bytes": corpus_path.stat().st_size if has_corpus else None,
        "tokenizer_trained": has_tokenizer,
        "data_prepared": has_train and has_val,
    }


@router.delete("/data/corpus", tags=["data"])
def delete_corpus():
    from worker.worker import DATA_DIR, RAW_DIR, TOKENIZER_PATH

    corpus_path = RAW_DIR / "corpus.txt"
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail="no corpus uploaded")

    corpus_path.unlink()
    for f in [TOKENIZER_PATH, DATA_DIR / "train.bin", DATA_DIR / "val.bin"]:
        if f.exists():
            f.unlink()

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


def _start_train(request: TrainRequest, background_tasks: BackgroundTasks):
    if request.id not in worker.cache:
        raise HTTPException(status_code=400, detail="model not compiled - compile first")
    if worker.training:
        raise HTTPException(status_code=409, detail="training already in progress")

    background_tasks.add_task(
        worker.train,
        request.id,
        max_steps=request.max_steps,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        eval_interval=request.eval_interval,
        eval_iters=request.eval_iters,
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
