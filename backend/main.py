import asyncio
import base64
import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel

# Add project root to Python path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_engine import FlashHeadInferenceEngine
from tts_engine import TTSEngine


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CKPT_DIR = os.path.join(BASE_DIR, "..", "models", "SoulX-FlashHead-1_3B")
    WAV2VEC_DIR = os.path.join(BASE_DIR, "..", "models", "wav2vec2-base-960h")
    MODEL_TYPE = "lite"
    DEFAULT_IMAGE = os.path.join(BASE_DIR, "..", "examples", "cond_image.png")
    DEFAULT_SEED = 42
    HOST = "0.0.0.0"
    PORT = 8000
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHUNK_MS_DEFAULT = 80
    AUDIO_QUEUE_MAXSIZE = 64
    JPEG_QUALITY = 85
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    APP_LOG_FILE = os.path.join(LOG_DIR, "flashhead-backend.log")
    STREAM_LOG_FILE = os.path.join(LOG_DIR, "flashhead-stream.jsonl")


config = Config()


def configure_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        enqueue=True,
        colorize=True,
        backtrace=False,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        config.APP_LOG_FILE,
        level="DEBUG",
        enqueue=True,
        rotation="20 MB",
        retention="7 days",
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        config.STREAM_LOG_FILE,
        level="INFO",
        enqueue=True,
        rotation="20 MB",
        retention="7 days",
        backtrace=False,
        diagnose=False,
        serialize=True,
    )


configure_logging()
app = FastAPI(title="FlashHead Real-time Avatar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoxiaoNeural"


class ImageRequest(BaseModel):
    image_path: str
    seed: int = 42


class AppState:
    def __init__(self):
        self.engine: Optional[FlashHeadInferenceEngine] = None
        self.tts: Optional[TTSEngine] = None
        self.initialized = False
        self.engine_ready = False
        self.session_ready = False
        self.initializing = False
        self.last_error: Optional[str] = None
        self.queue_depth = 0
        self._lock = threading.Lock()

    def initialize(self, load_default_session: bool = True):
        with self._lock:
            if self.initializing:
                wait_for_existing = True
            else:
                wait_for_existing = False
                self.initializing = True
        if wait_for_existing:
            # Another thread is initializing. Wait for completion and surface any failure.
            for _ in range(600):
                time.sleep(0.1)
                with self._lock:
                    if not self.initializing:
                        break
            if not self.engine_ready and self.last_error:
                raise RuntimeError(self.last_error)
            return

        try:
            if self.engine is None:
                logger.info("Initializing FlashHead inference engine...")
                self.engine = FlashHeadInferenceEngine(
                    ckpt_dir=config.CKPT_DIR,
                    model_type=config.MODEL_TYPE,
                    wav2vec_dir=config.WAV2VEC_DIR,
                )
            self.engine_ready = True

            if self.tts is None:
                self.tts = TTSEngine(voice="zh-CN-XiaoxiaoNeural")

            if load_default_session:
                if not os.path.exists(config.DEFAULT_IMAGE):
                    raise FileNotFoundError(
                        f"Default image not found: {config.DEFAULT_IMAGE}. "
                        "Please call POST /api/initialize with a valid image_path."
                    )
                self.engine.initialize_session(
                    image_path=config.DEFAULT_IMAGE,
                    seed=config.DEFAULT_SEED,
                )
                self.session_ready = True

            self.initialized = self.engine_ready and self.session_ready and self.tts is not None
            self.last_error = None
            logger.info(
                f"App state initialized: engine_ready={self.engine_ready}, "
                f"session_ready={self.session_ready}, initialized={self.initialized}"
            )
        except Exception as exc:
            self.initialized = False
            self.session_ready = False
            self.last_error = str(exc)
            logger.exception(f"Initialization failed: {exc}")
            raise
        finally:
            with self._lock:
                self.initializing = False

    def initialize_session(self, image_path: str, seed: int):
        if self.engine is None:
            raise RuntimeError("Engine not ready")
        self.engine.initialize_session(image_path=image_path, seed=seed)
        self.session_ready = True
        self.initialized = self.engine_ready and self.session_ready and self.tts is not None
        self.last_error = None

    def ensure_session_ready(self):
        if not self.engine_ready:
            raise RuntimeError("Engine is not ready")
        if not self.session_ready:
            raise RuntimeError(
                "Session is not ready. Please initialize with POST /api/initialize or send session.init with image_path."
            )
        if self.tts is None:
            raise RuntimeError("TTS is not ready")

    def on_oom(self):
        with self._lock:
            if self.engine is not None:
                self.engine.reset()
            self.session_ready = False
            self.initialized = False
            self.last_error = "GPU out of memory"

    def gpu_mem_info(self):
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        }


state = AppState()


def error_payload(code: str, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload = {"type": "error", "code": code, "message": message}
    if details:
        payload["details"] = details
    return payload


async def send_error(websocket: WebSocket, code: str, message: str, details: Optional[dict[str, Any]] = None):
    await safe_send_json(websocket, error_payload(code=code, message=message, details=details))


async def safe_send_json(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


def chunk_audio(audio_array: np.ndarray, chunk_ms: int, sample_rate: int) -> list[np.ndarray]:
    chunk_ms = max(20, min(200, int(chunk_ms)))
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
    chunks = []
    for start in range(0, len(audio_array), chunk_samples):
        chunks.append(audio_array[start : start + chunk_samples])
    return chunks


def encode_frame_to_base64_jpeg(frame: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buffer).decode("utf-8")


@dataclass
class StreamContext:
    websocket: WebSocket
    queue: asyncio.Queue
    stream_id: str
    frame_seq: int = 0
    total_frames: int = 0
    source_fps: int = 25
    target_fps: int = 25
    frame_stride: int = 1
    output_fps: float = 25.0
    generated_frame_count: int = 0
    stream_start_ts: float = 0.0
    first_frame_latency_ms: int = -1
    chunk_ms: int = config.AUDIO_CHUNK_MS_DEFAULT
    cancelled: bool = False
    closed: bool = False


async def stream_worker(ctx: StreamContext):
    try:
        logger.info(f"[sid={ctx.stream_id}] stream_worker started")
        while True:
            item = await ctx.queue.get()
            kind = item.get("kind")
            state.queue_depth = ctx.queue.qsize()

            if kind == "stop":
                logger.info(f"[sid={ctx.stream_id}] stream_worker stop signal")
                break

            if kind == "cancel":
                ctx.cancelled = True
                while not ctx.queue.empty():
                    _ = ctx.queue.get_nowait()
                await asyncio.to_thread(state.engine.clear_stream_pending)
                sent = await safe_send_json(
                    ctx.websocket,
                    {
                        "type": "stream.done",
                        "total_frames": ctx.total_frames,
                        "elapsed_ms": int((time.perf_counter() - ctx.stream_start_ts) * 1000) if ctx.stream_start_ts else 0,
                        "cancelled": True,
                    },
                )
                if not sent:
                    ctx.closed = True
                    return
                logger.info(f"[sid={ctx.stream_id}] stream cancelled")
                continue

            if kind != "audio":
                continue

            if ctx.stream_start_ts == 0.0:
                ctx.stream_start_ts = time.perf_counter()
                ctx.cancelled = False
                ctx.generated_frame_count = 0
                ctx.first_frame_latency_ms = -1

            audio_chunk = item.get("audio", np.array([], dtype=np.float32))
            flush = bool(item.get("flush", False))

            try:
                frames = await asyncio.to_thread(state.engine.process_audio_stream_chunk, audio_chunk, flush)
            except torch.cuda.OutOfMemoryError:
                logger.exception("GPU Out of Memory in stream worker")
                state.on_oom()
                await send_error(ctx.websocket, "GPU_OOM", "GPU memory exhausted. Session has been reset.")
                continue
            except Exception as exc:
                logger.exception(f"[sid={ctx.stream_id}] Stream worker failed: {exc}")
                await send_error(ctx.websocket, "STREAM_PROCESS_FAILED", str(exc))
                continue

            if isinstance(frames, np.ndarray) and frames.size > 0:
                for frame in frames:
                    if ctx.frame_stride > 1 and (ctx.generated_frame_count % ctx.frame_stride) != 0:
                        ctx.generated_frame_count += 1
                        continue
                    frame_base64 = encode_frame_to_base64_jpeg(frame, config.JPEG_QUALITY)
                    seq = ctx.frame_seq
                    is_key = seq % 25 == 0
                    sent = await safe_send_json(
                        ctx.websocket,
                        {
                            "type": "video.frame",
                            "seq": seq,
                            "ts": int(time.time() * 1000),
                            "is_key": is_key,
                            "fps": ctx.output_fps,
                            "data": frame_base64,
                        },
                    )
                    if not sent:
                        ctx.closed = True
                        return
                    if ctx.first_frame_latency_ms < 0 and ctx.stream_start_ts > 0:
                        ctx.first_frame_latency_ms = int((time.perf_counter() - ctx.stream_start_ts) * 1000)
                        logger.info(f"[sid={ctx.stream_id}] first_frame_latency_ms={ctx.first_frame_latency_ms}")
                    ctx.frame_seq += 1
                    ctx.total_frames += 1
                    ctx.generated_frame_count += 1

            if flush:
                elapsed_ms = int((time.perf_counter() - ctx.stream_start_ts) * 1000) if ctx.stream_start_ts else 0
                elapsed_s = max(1e-6, elapsed_ms / 1000.0)
                effective_gen_fps = float(ctx.total_frames) / elapsed_s
                real_time_ratio = effective_gen_fps / max(1e-6, float(ctx.output_fps))
                logger.info(
                    f"[sid={ctx.stream_id}] stream.done: total_frames={ctx.total_frames}, elapsed_ms={elapsed_ms}, "
                    f"target_fps={ctx.target_fps}, output_fps={ctx.output_fps:.3f}, stride={ctx.frame_stride}, "
                    f"effective_gen_fps={effective_gen_fps:.3f}, real_time_ratio={real_time_ratio:.3f}, "
                    f"cancelled={ctx.cancelled}, "
                    f"first_frame_latency_ms={ctx.first_frame_latency_ms}"
                )
                sent = await safe_send_json(
                    ctx.websocket,
                    {
                        "type": "stream.done",
                        "total_frames": ctx.total_frames,
                        "elapsed_ms": elapsed_ms,
                        "fps": ctx.output_fps,
                        "effective_gen_fps": effective_gen_fps,
                        "real_time_ratio": real_time_ratio,
                        "cancelled": ctx.cancelled,
                    },
                )
                if not sent:
                    ctx.closed = True
                    return
                ctx.stream_start_ts = 0.0
                ctx.total_frames = 0
                ctx.frame_seq = 0
                ctx.generated_frame_count = 0
    except (asyncio.CancelledError, WebSocketDisconnect):
        logger.info(f"[sid={ctx.stream_id}] stream_worker cancelled/disconnected")
        return


async def enqueue_audio(ctx: StreamContext, audio_chunk: np.ndarray, flush: bool = False) -> bool:
    if ctx.closed:
        return False

    if flush and ctx.queue.full():
        # Never drop a flush marker, otherwise stream.done may never be emitted.
        wait_start = time.perf_counter()
        logger.warning(
            f"[sid={ctx.stream_id}] queue full before flush, waiting: "
            f"queue_depth={ctx.queue.qsize()}, chunk_ms={ctx.chunk_ms}"
        )
        while ctx.queue.full():
            if ctx.closed:
                return False
            await asyncio.sleep(0.005)
        wait_ms = int((time.perf_counter() - wait_start) * 1000)
        logger.info(
            f"[sid={ctx.stream_id}] flush enqueue resumed after {wait_ms}ms: "
            f"queue_depth={ctx.queue.qsize()}"
        )

    if ctx.queue.full():
        sent = await safe_send_json(
            ctx.websocket,
            {
                "type": "stream.ack",
                "queue_depth": ctx.queue.qsize(),
                "estimated_latency_ms": int(ctx.queue.qsize() * ctx.chunk_ms),
                "dropped": True,
            },
        )
        if not sent:
            ctx.closed = True
            return False
        return True

    await ctx.queue.put({"kind": "audio", "audio": audio_chunk, "flush": flush})
    state.queue_depth = ctx.queue.qsize()
    sent = await safe_send_json(
        ctx.websocket,
        {
            "type": "stream.ack",
            "queue_depth": ctx.queue.qsize(),
            "estimated_latency_ms": int(ctx.queue.qsize() * ctx.chunk_ms),
            "dropped": False,
        },
    )
    if not sent:
        ctx.closed = True
        return False
    return True


def clamp_target_fps(target_fps: Any, source_fps: int) -> int:
    try:
        fps = int(target_fps)
    except (TypeError, ValueError):
        fps = source_fps
    fps = max(1, min(source_fps, fps))
    return fps


@app.get("/")
async def root():
    dist_path = os.path.join(config.BASE_DIR, "..", "frontend", "dist", "index.html")
    if os.path.exists(dist_path):
        with open(dist_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    fallback = """
    <html>
      <head><title>FlashHead Web</title></head>
      <body style="font-family: sans-serif; padding: 24px;">
        <h2>Frontend build not found</h2>
        <p>Run <code>cd frontend && npm run build</code> and refresh this page,
        or use dev server at <a href="http://localhost:3000">http://localhost:3000</a>.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=fallback)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "initialized": state.initialized,
        "engine_ready": state.engine_ready,
        "session_ready": state.session_ready,
        "queue_depth": state.queue_depth,
        "gpu_mem": state.gpu_mem_info(),
        "last_error": state.last_error,
    }


@app.post("/api/initialize")
async def initialize_session(request: ImageRequest):
    try:
        if not state.engine_ready:
            await asyncio.to_thread(state.initialize, False)
        await asyncio.to_thread(state.initialize_session, request.image_path, request.seed)
        return {"status": "success", "message": "Session initialized"}
    except Exception as exc:
        logger.exception(f"Initialize failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/tts")
async def text_to_speech(request: TextRequest):
    try:
        if state.tts is None:
            await asyncio.to_thread(state.initialize, False)
        if state.tts is None:
            raise RuntimeError("TTS is not ready")

        if request.voice != state.tts.voice:
            state.tts = TTSEngine(voice=request.voice)

        audio_array = await state.tts.text_to_speech(request.text)
        audio_bytes = (audio_array * 32768.0).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "status": "success",
            "audio": audio_base64,
            "duration": len(audio_array) / config.AUDIO_SAMPLE_RATE,
            "sample_rate": config.AUDIO_SAMPLE_RATE,
            "encoding": "pcm_s16le",
        }
    except Exception as exc:
        logger.exception(f"TTS failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws/avatar")
async def avatar_websocket(websocket: WebSocket):
    await websocket.accept()
    stream_id = uuid4().hex[:8]
    logger.info(f"[sid={stream_id}] WebSocket connection established")

    if not state.engine_ready or state.tts is None or not state.session_ready:
        try:
            await asyncio.to_thread(state.initialize, True)
        except Exception as exc:
            await send_error(websocket, "INIT_FAILED", str(exc))

    ctx = StreamContext(
        websocket=websocket,
        queue=asyncio.Queue(maxsize=config.AUDIO_QUEUE_MAXSIZE),
        stream_id=stream_id,
    )
    if state.engine is not None and state.engine.infer_params is not None:
        ctx.source_fps = int(state.engine.infer_params.get("tgt_fps", 25))
        ctx.target_fps = ctx.source_fps
        ctx.frame_stride = 1
        ctx.output_fps = float(ctx.source_fps)
    worker_task = asyncio.create_task(stream_worker(ctx))

    if state.session_ready:
        sent = await safe_send_json(websocket, {"type": "session.ready", "engine_ready": True, "session_ready": True})
        if not sent:
            return
    else:
        sent = await safe_send_json(
            websocket,
            {
                "type": "session.error",
                "engine_ready": state.engine_ready,
                "session_ready": state.session_ready,
                "message": "Session not ready. Send session.init with image_path or call /api/initialize.",
            },
        )
        if not sent:
            return

    try:
        while True:
            packet = await websocket.receive()
            if packet.get("type") == "websocket.disconnect":
                break

            if packet.get("bytes") is not None:
                try:
                    state.ensure_session_ready()
                except Exception as exc:
                    await send_error(websocket, "SESSION_NOT_READY", str(exc))
                    continue

                pcm_i16 = np.frombuffer(packet["bytes"], dtype=np.int16)
                audio_chunk = pcm_i16.astype(np.float32) / 32768.0
                if not await enqueue_audio(ctx, audio_chunk, flush=False):
                    raise WebSocketDisconnect(code=1006)
                continue

            raw_text = packet.get("text")
            if raw_text is None:
                continue

            try:
                message = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_error(websocket, "BAD_JSON", "Invalid JSON payload")
                continue

            msg_type = message.get("type", "")

            if msg_type == "session.init":
                image_path = message.get("image_path")
                seed = int(message.get("seed", config.DEFAULT_SEED))
                voice = message.get("voice")

                try:
                    if image_path:
                        await asyncio.to_thread(state.initialize_session, image_path, seed)
                    else:
                        if not state.session_ready:
                            await asyncio.to_thread(state.initialize, True)
                        state.ensure_session_ready()
                    if voice and state.tts and voice != state.tts.voice:
                        state.tts = TTSEngine(voice=voice)
                    sent = await safe_send_json(
                        websocket,
                        {"type": "session.ready", "engine_ready": state.engine_ready, "session_ready": state.session_ready}
                    )
                    if not sent:
                        raise WebSocketDisconnect(code=1006)
                except Exception as exc:
                    sent = await safe_send_json(
                        websocket,
                        {
                            "type": "session.error",
                            "engine_ready": state.engine_ready,
                            "session_ready": state.session_ready,
                            "message": str(exc),
                        },
                    )
                    if not sent:
                        raise WebSocketDisconnect(code=1006)
                continue

            if msg_type in {"tts.request", "text"}:
                text = (message.get("text") or "").strip()
                voice = message.get("voice")
                chunk_ms = int(message.get("chunk_ms", config.AUDIO_CHUNK_MS_DEFAULT))
                ctx.chunk_ms = chunk_ms
                requested_fps = message.get("render_fps", ctx.source_fps)
                ctx.target_fps = clamp_target_fps(requested_fps, ctx.source_fps)
                ctx.frame_stride = max(1, int(round(ctx.source_fps / ctx.target_fps)))
                ctx.output_fps = float(ctx.source_fps) / float(ctx.frame_stride)

                if not text:
                    await send_error(websocket, "EMPTY_TEXT", "Text is empty")
                    continue

                try:
                    state.ensure_session_ready()
                except Exception as exc:
                    await send_error(websocket, "SESSION_NOT_READY", str(exc))
                    continue

                try:
                    if voice and state.tts and voice != state.tts.voice:
                        state.tts = TTSEngine(voice=voice)
                    audio_array = await state.tts.text_to_speech(text)
                    logger.info(
                        f"[sid={ctx.stream_id}] tts.request accepted: chars={len(text)}, audio_samples={len(audio_array)}, "
                        f"chunk_ms={chunk_ms}, target_fps={ctx.target_fps}, stride={ctx.frame_stride}"
                    )

                    # Send audio to frontend for real-time playback.
                    audio_bytes = (audio_array * 32768.0).astype(np.int16).tobytes()
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    sent = await safe_send_json(
                        websocket,
                        {
                            "type": "audio.pcm",
                            "sample_rate": config.AUDIO_SAMPLE_RATE,
                            "encoding": "pcm_s16le",
                            "data": audio_base64,
                        },
                    )
                    if not sent:
                        raise WebSocketDisconnect(code=1006)
                    logger.info(
                        f"[sid={ctx.stream_id}] audio.pcm sent: bytes={len(audio_bytes)}, "
                        f"samples={len(audio_array)}"
                    )

                    chunks = chunk_audio(audio_array, chunk_ms=chunk_ms, sample_rate=config.AUDIO_SAMPLE_RATE)
                    logger.info(
                        f"[sid={ctx.stream_id}] audio chunked: chunks={len(chunks)}, "
                        f"sample_rate={config.AUDIO_SAMPLE_RATE}"
                    )
                    for chunk in chunks:
                        if not await enqueue_audio(ctx, chunk.astype(np.float32), flush=False):
                            raise WebSocketDisconnect(code=1006)
                    if not await enqueue_audio(ctx, np.array([], dtype=np.float32), flush=True):
                        raise WebSocketDisconnect(code=1006)
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    await send_error(websocket, "TTS_OR_STREAM_FAILED", str(exc))
                continue

            if msg_type == "audio":
                # Legacy JSON+base64 path: interpret as PCM16.
                try:
                    state.ensure_session_ready()
                except Exception as exc:
                    await send_error(websocket, "SESSION_NOT_READY", str(exc))
                    continue

                audio_base64 = message.get("data", "")
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    pcm_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_chunk = pcm_i16.astype(np.float32) / 32768.0
                    if not await enqueue_audio(ctx, audio_chunk, flush=False):
                        raise WebSocketDisconnect(code=1006)
                    if bool(message.get("flush", True)):
                        if not await enqueue_audio(ctx, np.array([], dtype=np.float32), flush=True):
                            raise WebSocketDisconnect(code=1006)
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    await send_error(websocket, "BAD_AUDIO_PAYLOAD", str(exc))
                continue

            if msg_type == "stream.end":
                if not await enqueue_audio(ctx, np.array([], dtype=np.float32), flush=True):
                    raise WebSocketDisconnect(code=1006)
                continue

            if msg_type == "stream.cancel":
                await ctx.queue.put({"kind": "cancel"})
                continue

            await send_error(websocket, "UNKNOWN_MESSAGE_TYPE", f"Unsupported type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[sid={stream_id}] WebSocket disconnected")
    except Exception as exc:
        logger.exception(f"[sid={stream_id}] WebSocket error: {exc}")
        with contextlib.suppress(Exception):
            await send_error(websocket, "WS_RUNTIME_ERROR", str(exc))
    finally:
        with contextlib.suppress(Exception):
            await ctx.queue.put({"kind": "stop"})
        if not worker_task.done():
            worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker_task
        state.queue_depth = 0
        logger.info(f"[sid={stream_id}] WebSocket cleanup complete")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting FlashHead API server...")

    async def _deferred_init():
        try:
            await asyncio.to_thread(state.initialize, True)
        except Exception as exc:
            logger.warning(f"Deferred init failed: {exc}")

    asyncio.create_task(_deferred_init())


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
