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


def _require_idle():
    # the GPU runs one job at a time; reject a new inference while anything holds it
    if worker.busy:
        raise HTTPException(status_code=409, detail="training or inference in progress")


@router.post("/generate", tags=["generate"])
def generate(request: GenerateRequest):
    _require_idle()
    try:
        prompt_ids = worker.encode_prompt(request.id, request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    worker.inferring = True
    worker.stop_inference = False
    worker.set_activity("generate", request.id)
    try:
        result = worker.generate(
            model_id=request.id,
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            bench=request.bench,
        )
    finally:
        worker.inferring = False
        worker.clear_activity()

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/generate/stream", tags=["generate"])
def generate_stream(request: GenerateRequest):
    _require_idle()
    try:
        prompt_ids = worker.encode_prompt(request.id, request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # mark the GPU busy before returning the response so a concurrent train/generate is rejected;
    # the generator's finally clears it when the stream ends (or the client disconnects)
    worker.inferring = True
    worker.stop_inference = False
    worker.set_activity("generate", request.id)

    def event_stream():
        try:
            for event in worker.generate_stream(
                model_id=request.id,
                prompt_ids=prompt_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                bench=request.bench,
            ):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        finally:
            worker.inferring = False
            worker.clear_activity()

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
    if worker.inferring:
        raise HTTPException(status_code=409, detail="inference in progress")

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
    if worker.inferring:
        raise HTTPException(status_code=409, detail="inference in progress")
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


@router.post("/generate/stop", tags=["generate"])
def stop_generate():
    # cooperative cancel for a streaming generate or inference benchmark: the generation loop
    # breaks on its next step and its stream ends, freeing the GPU (a separate request from the
    # aborted stream, so it lands even after the client stops reading)
    worker.stop_inference = True
    return {"status": "stopping"}


# -- Benchmark --


@router.post("/bench/profile", tags=["benchmark"])
def bench_profile(request: ProfileRequest):
    from dataclasses import asdict

    _require_idle()
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

    # results snapshot kept on the worker so a page reload can rehydrate the table (mirrors the
    # training benchmark's bench_state); one column per model, filled in as its events arrive
    cols = [
        {
            "id": g.id,
            "tokens_per_sec": None,
            "ms_per_token": None,
            "prefill_ms": None,
            "peak_vram_mb": None,
            "profile": [],
            "text": "",
            "gen_tokens": 0,
            "running": False,
            "error": None,
            "done": False,
        }
        for g in request.graphs
    ]
    worker.gen_bench = {"owner": request.graphs[0].id, "status": "running", "columns": cols}

    try:
        for i, entry in enumerate(request.graphs):
            if worker.stop_inference:
                break  # a /generate/stop landed - halt the whole compare, not just this model
            graph = GraphSpec.from_dict(entry.graph.model_dump())
            s_hash = graph_structure_hash(graph)

            pkg = store.load(entry.id)
            if pkg is None or pkg.structure_hash != s_hash:
                cols[i]["error"], cols[i]["done"] = "model not trained", True
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

            cols[i]["running"] = True
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
                ev, data = event["event"], event["data"]
                if ev == "prefill":
                    cols[i]["prefill_ms"] = data["prefill_ms"]
                elif ev == "token":
                    cols[i]["text"] += data.get("text") or ""
                    cols[i]["gen_tokens"] += 1
                elif ev == "profile":
                    cols[i]["profile"] = data["nodes"]
                elif ev == "done":
                    b = data.get("bench") or {}
                    cols[i].update(
                        tokens_per_sec=b.get("tokens_per_sec"),
                        ms_per_token=b.get("decode_ms_per_token"),
                        prefill_ms=b.get("prefill_ms", cols[i]["prefill_ms"]),
                        peak_vram_mb=b.get("peak_vram_mb"),
                        text=data.get("text") or cols[i]["text"],
                        running=False,
                        done=True,
                    )
                data["graph_idx"] = i
                yield f"event: {ev}\ndata: {json.dumps(data)}\n\n"

        yield f"event: done\ndata: {json.dumps({'env': _collect_env(device)})}\n\n"

    except Exception as e:
        if worker.gen_bench:
            worker.gen_bench["status"] = "error"
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


@router.post("/generate/bench", tags=["benchmark"])
def bench_generate(request: BenchRunRequest):
    _require_idle()
    worker.inferring = True
    worker.stop_inference = False
    worker.set_activity("gen_bench", request.graphs[0].id)

    def stream():
        try:
            yield from _bench_stream(request)
        finally:
            worker.inferring = False
            worker.clear_activity()
            # settle the snapshot's status so a reload after a completed / stopped / disconnected
            # run rehydrates a finished table rather than a stuck "running" one
            if worker.gen_bench and worker.gen_bench.get("status") == "running":
                worker.gen_bench["status"] = "done"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/generate/bench/status", tags=["benchmark"])
def bench_generate_status():
    # last inference-benchmark's results (owner + per-model columns), kept in memory so a page
    # reload can rehydrate the table; empty until a compare has run
    return worker.gen_bench or {"owner": None, "status": "idle", "columns": []}


@router.get("/worker/activity", tags=["worker"])
def worker_activity():
    # one universal busy signal: what (if anything) currently occupies the GPU, so every project
    # can gate its Train / Generate controls (training and inference are mutually exclusive)
    return worker.get_activity()
