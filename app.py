#!/usr/bin/env python3
"""Kyutai Pocket TTS API for Railway, CPU only.

Pocket TTS (100M parameters, MIT code, CC-BY-4.0 weights) behind one authenticated, OpenAI-compatible HTTP API:
POST /v1/audio/speech takes text and returns 24 kHz speech as WAV, MP3, FLAC or raw PCM.
The model is the `pocket-tts` package, used unchanged. Weights and the 19 preset voices are baked into the image
from the ungated `kyutai/pocket-tts-without-voice-cloning` repository; nothing downloads at boot.

What this file adds around the model, because upstream ships a command-line tool and a demo server, not an
authenticated one:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until the model and every voice are loaded and warmed, then 200, unauthenticated.
  3. Voices are the preset embeddings in the image. Cloning a voice from an uploaded sample is not in this build
     (the weights that include the cloning encoder are gated), so a request that names anything else is refused.
  4. One synthesis at a time (the model is not thread-safe), a short queue, 429 when it is full.
  5. A wall-clock budget checked between model chunks, so a request answers 504 before Railway's roughly 5 minute
     edge timeout and the model is told to stop.
  6. Text and audio are held in memory for the request only and are never logged.
"""

import asyncio
import io
import os
import re
import socket
import sys
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY") or ""
DEFAULT_VOICE = (os.environ.get("DEFAULT_VOICE") or "alba").strip().lower()
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS") or 5000)
MAX_SECONDS = float(os.environ.get("MAX_SECONDS") or 240)
MAX_PENDING = int(os.environ.get("MAX_PENDING") or 3)
PUBLIC_PATHS = {"/health"}
MODEL_ID = "pocket-tts"
ALIASES = {"tts-1": MODEL_ID, "tts-1-hd": MODEL_ID, "pocket": MODEL_ID, "pocket-tts-100m": MODEL_ID}
# Same list as fetch_weights.py; the source recordings are CC0 or CC-BY 4.0 (see NOTICE).
VOICES = {
    "alba": "female", "anna": "female", "vera": "female", "fantine": "female", "eponine": "female",
    "azelma": "female", "mary": "female", "jane": "female", "eve": "female", "caro_davy": "female",
    "charles": "male", "paul": "male", "george": "male", "michael": "male", "marius": "male",
    "javert": "male", "bill_boerst": "male", "peter_yearsley": "male", "stuart_bell": "male",
}
# OpenAI voice names, for clients that send one. Fixed table; X-Voice on the reply says which voice was used.
OPENAI_VOICES = {"alloy": "alba", "ash": "charles", "ballad": "george", "coral": "anna", "echo": "paul",
                 "fable": "michael", "nova": "vera", "onyx": "javert", "sage": "jane", "shimmer": "eve",
                 "verse": "marius"}
FORMATS = {"wav": ("audio/wav", "WAV", "PCM_16"), "mp3": ("audio/mpeg", "MP3", None),
           "flac": ("audio/flac", "FLAC", "PCM_16"), "pcm": ("audio/L16;rate=24000;channels=1", None, None)}

_state = {"ready": False, "error": None, "loaded_in": None}
_model = None
_voice_states: dict = {}
_run_lock = threading.Lock()      # one synthesis at a time
_pending = 0
_pending_lock = threading.Lock()

app = FastAPI(title="Pocket TTS API", docs_url=None, redoc_url=None, openapi_url=None)


class _Deadline(Exception):
    pass


def _boot() -> None:
    global _model
    t0 = time.time()
    try:
        from pocket_tts import TTSModel
        torch.set_num_threads(1)  # measured: 1 thread is 2x faster than 2 or 4 for this 100M batch-1 model
        _model = TTSModel.load_model("english")
        if _model.has_voice_cloning:
            raise RuntimeError("the gated voice-cloning weights are loaded; this image carries the ungated ones")
        for name in VOICES:
            _voice_states[name] = _model.get_state_for_audio_prompt(name)
        with _run_lock:
            _synthesise("Hello.", DEFAULT_VOICE, None, time.time() + 120)
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[pocket-tts] ready voices={len(VOICES)} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[pocket-tts] LOAD FAILED: {_state['error']}", flush=True)


def _synthesise(text: str, voice: str, seed: int | None, deadline: float):
    """Run the model piece by piece; stop it between pieces once the deadline passes."""
    assert _model is not None
    if seed is not None:
        torch.manual_seed(seed % (2 ** 31))
    stop = threading.Event()
    pieces = []
    try:
        for chunk in _model.generate_audio_stream(_voice_states[voice], text, copy_state=True, stop=stop):
            pieces.append(chunk.detach().cpu().numpy().reshape(-1).astype(np.float32))
            if time.time() > deadline:
                raise _Deadline()
    finally:
        stop.set()
    if not pieces:
        raise HTTPException(422, "input has no speakable text")
    return np.concatenate(pieces), _model.sample_rate


def _encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    import soundfile as sf
    if fmt == "pcm":
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _mime, sf_format, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=sf_format, subtype=subtype)
    return buf.getvalue()


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = Field(default=None, description="a preset voice name (GET /v1/audio/voices) or an OpenAI voice name")
    response_format: str | None = "wav"
    speed: float | None = Field(default=None, description="only 1.0 is supported")
    seed: int | None = None


def _resolve(req: SpeechRequest):
    raw = (req.model or MODEL_ID).strip().lower()
    if ALIASES.get(raw, raw) != MODEL_ID:
        raise HTTPException(422, f"unknown model '{raw}'; use '{MODEL_ID}'")
    fmt = (req.response_format or "wav").strip().lower()
    if fmt not in FORMATS:
        raise HTTPException(422, f"unknown response_format '{fmt}'; one of {sorted(FORMATS)}")
    voice_raw = (req.voice or DEFAULT_VOICE).strip()
    voice = OPENAI_VOICES.get(voice_raw.lower(), voice_raw.lower())
    if voice not in VOICES:
        raise HTTPException(422, f"unknown voice '{voice_raw}'; one of {sorted(VOICES)} or {sorted(OPENAI_VOICES)}. "
                                 "Voice cloning from a sample is not available in this build.")
    if req.speed is not None and abs(req.speed - 1.0) > 1e-9:
        raise HTTPException(422, "`speed` is not supported by the model; omit it or send 1.0")
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(422, "`input` is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"`input` is {len(text)} characters; the limit is {MAX_INPUT_CHARS}")
    if not re.search(r"\w", text):
        raise HTTPException(422, "input has no text the model can read")
    return text, voice, fmt


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and token == API_KEY):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > 1024 * 1024:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "loaded_in_s": _state["loaded_in"], "voices": len(VOICES),
            "rss_mb": _status_kb("VmRSS") // 1024, "peak_rss_mb": _status_kb("VmHWM") // 1024}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "description": "Kyutai Pocket TTS, 100M parameters, English, CPU",
         "license": "code MIT; weights CC-BY-4.0 (kyutai/pocket-tts-without-voice-cloning), attribution: Kyutai",
         "sample_rate": _model.sample_rate if _model else 24000}],
        "limits": {"max_input_chars": MAX_INPUT_CHARS, "max_seconds": MAX_SECONDS, "max_pending": MAX_PENDING,
                   "formats": sorted(FORMATS), "default_voice": DEFAULT_VOICE, "voice_cloning": False}}


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [{"id": k, "gender": v} for k, v in sorted(VOICES.items())], "openai_aliases": OPENAI_VOICES}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    global _pending
    text, voice, fmt = _resolve(req)
    if not _state["ready"]:
        raise HTTPException(503, "model is still loading")
    with _pending_lock:
        if _pending >= MAX_PENDING:
            return JSONResponse({"error": "busy: too many requests are running or waiting"}, status_code=429,
                                headers={"Retry-After": "30"})
        _pending += 1
    deadline = time.time() + MAX_SECONDS

    def job():
        if not _run_lock.acquire(timeout=max(1.0, deadline - time.time())):
            raise _Deadline()
        try:
            t_gen = time.time()
            audio, sr = _synthesise(text, voice, req.seed, deadline)
            return audio, sr, time.time() - t_gen
        finally:
            _run_lock.release()

    try:
        try:
            audio, sr, took = await asyncio.to_thread(job)
        except _Deadline:
            return JSONResponse({"error": f"could not finish within {MAX_SECONDS:.0f} s; send less text"},
                                status_code=504)
        body = _encode(audio, sr, fmt)
        seconds = len(audio) / sr
        headers = {"X-Processing-Seconds": f"{took:.2f}", "X-Audio-Seconds": f"{seconds:.2f}",
                   "X-Realtime-Factor": f"{took / seconds:.3f}" if seconds else "0",
                   "X-Chars-Per-Second": f"{len(text) / took:.0f}" if took else "0",
                   "X-Model": MODEL_ID, "X-Voice": voice, "X-Sample-Rate": str(sr),
                   "X-Attribution": "Kyutai Pocket TTS, CC-BY-4.0"}
        return Response(body, media_type=FORMATS[fmt][0], headers=headers)
    finally:
        with _pending_lock:
            _pending -= 1


def main() -> int:
    if not API_KEY:
        print("[pocket-tts] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    if DEFAULT_VOICE not in VOICES:
        print(f"[pocket-tts] FATAL: DEFAULT_VOICE '{DEFAULT_VOICE}' is not one of {sorted(VOICES)}",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_boot, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[pocket-tts] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
