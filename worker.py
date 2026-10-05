#!/usr/bin/env python3
"""Agent Orange embedding worker (v1).

Embeds offerings from the Postgres view public.offering_embed_queue with
nomic-embed-text-v1.5 (text) and nomic-embed-vision-v1.5 (image), which share
one 768-d space, and writes halfvec(768) vectors back.

Runtime: onnxruntime + tokenizers + numpy + Pillow. No torch.

Modes
  DRY_RUN=1   read a local JSONL/CSV of (id, embed_text, main_image_url), write
              vectors to a local JSONL. Never connects to any database.
  default     DATABASE_URL (psycopg, preferred) or SUPABASE_URL +
              SUPABASE_SERVICE_ROLE_KEY (PostgREST).

Logging: progress counts, timings and row ids only. Never row text, never secrets.
"""
from __future__ import annotations

import csv
import gc
import io
import json
import logging
import os
import re
import resource
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

def _tune_glibc_malloc():
    """Before big allocations: cap malloc arenas and serve >=128 KB blocks via mmap so
    ORT activation buffers go straight back to the OS. Measured: peak RSS 431 -> 391 MB
    for ~10% slower text inference. Same effect as MALLOC_ARENA_MAX=2
    MALLOC_MMAP_THRESHOLD_=131072 (also set in the Dockerfile). Set MALLOC_TUNE=0 to skip."""
    if os.environ.get("MALLOC_TUNE", "1") == "0":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-8, int(os.environ.get("MALLOC_ARENA_MAX", "2")))           # M_ARENA_MAX
        libc.mallopt(-3, int(os.environ.get("MALLOC_MMAP_THRESHOLD_", "131072")))  # M_MMAP_THRESHOLD
    except Exception:
        pass


_tune_glibc_malloc()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image
from tokenizers import Tokenizer

# --------------------------------------------------------------------------- config

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return default if not v else v in ("1", "true", "yes", "on")


HERE = os.path.dirname(os.path.abspath(__file__))
DRY_RUN = env_bool("DRY_RUN")
# LIMIT: unset/empty = drain until queue empty (production). Set e.g. 500 for test-only caps.
_limit_raw = os.environ.get("LIMIT", "").strip()
LIMIT = int(_limit_raw) if _limit_raw else None
BATCH_SIZE = env_int("BATCH_SIZE", 50)           # rows per fetch + per write transaction
MODEL_PRECISION = os.environ.get("MODEL_PRECISION", "int8").strip().lower()
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "models"))
ORT_THREADS = env_int("ORT_THREADS", 1)          # intra-op threads; 1 is fine on Standard 1 CPU
MAX_TOKENS = env_int("MAX_TOKENS", 512)          # truncation for embed_text (model supports 8192; RAM/CPU bound)
TEXT_PREFIX = "search_document: "
LOAD_MODE = os.environ.get("LOAD_MODE", "both").strip().lower()  # both | sequential
IMAGE_WIDTH_PARAM = env_int("SHOPIFY_IMAGE_WIDTH", 512)
IMAGE_TIMEOUT = float(os.environ.get("IMAGE_TIMEOUT", "15"))
IMAGE_MAX_BYTES = env_int("IMAGE_MAX_BYTES", 15 * 1024 * 1024)
IMAGE_MAX_PIXELS = env_int("IMAGE_MAX_PIXELS", 16_000_000)  # non-JPEG decode guard (~64 MB RGBA)
FETCH_WORKERS = env_int("FETCH_WORKERS", 4)      # parallel image downloads (I/O only)
IDLE_AFTER_DONE = env_bool("IDLE_AFTER_DONE", default=not DRY_RUN)
TARGET_TABLE = os.environ.get("TARGET_TABLE", "public.offering")
QUEUE_VIEW = os.environ.get("QUEUE_VIEW", "public.offering_embed_queue")
DRY_RUN_INPUT = os.environ.get("DRY_RUN_INPUT", os.path.join(HERE, "sample", "sample.jsonl"))
DRY_RUN_OUTPUT = os.environ.get("DRY_RUN_OUTPUT", os.path.join(HERE, "out", f"vectors_{MODEL_PRECISION}.jsonl"))
USER_AGENT = os.environ.get("IMAGE_USER_AGENT", "AgentOrangeEmbedWorker/1.0 (+product image embedding)")
DIM = 768

# CLIPImageProcessor settings from nomic-embed-vision-v1.5/preprocessor_config.json
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
IMG_SIZE = 224
Image.MAX_IMAGE_PIXELS = 60_000_000

_SECRETS = [s for s in (os.environ.get("DATABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")) if s]


def redact(msg: str) -> str:
    """Strip anything secret-looking from a log message."""
    for s in _SECRETS:
        msg = msg.replace(s, "[REDACTED]")
    msg = re.sub(r"(postgres(?:ql)?://)[^@\s]+@", r"\1[REDACTED]@", msg)
    msg = re.sub(r"(password=)\S+", r"\1[REDACTED]", msg, flags=re.I)
    msg = re.sub(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+", "[REDACTED_JWT]", msg)
    return msg


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
log = logging.getLogger("embed")
log.addHandler(_h)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
log.propagate = False
for noisy in ("urllib3", "PIL", "psycopg"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

STOP = False


def _on_signal(signum, _frame):
    global STOP
    STOP = True
    log.info("signal %s received; finishing current rows, committing, then exiting", signum)


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)


def malloc_trim():
    """Hand freed heap back to the OS between batches (glibc only; no-op elsewhere)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def cur_rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- models

def _model_file(kind: str) -> str:
    """Resolve the ONNX path for text|vision under MODEL_PRECISION.

    int8 (default): build-time MatMulNBits weight-only 8-bit (model_w8.onnx). Activations
    stay fp32; ORT uses int8 weight kernels (accuracy_level=4). Chosen because the HF
    dynamic-quant files (MatMulInteger + DynamicQuantizeLinear) are non-deterministic
    under CPU preemption on AMX_INT8 hosts — measured self-cos as low as -0.08 when
    another thread shares the core. Shared/quota CPUs preempt; MatMulNBits is stable
    (self-cos 1.0) and agrees with fp32 at cos ≥ 0.997 (text) / ≥ 0.977 (vision).

    fp32: model.onnx for both. Fits Standard 2 GB with LOAD_MODE=both (~1 GB RSS);
    use LOAD_MODE=sequential only if you need more headroom.
    """
    if MODEL_PRECISION == "fp32":
        name = "model.onnx"
    elif MODEL_PRECISION == "int8":
        name = "model_w8.onnx"
    else:
        raise SystemExit(f"MODEL_PRECISION must be int8 or fp32, got {MODEL_PRECISION!r}")
    path = os.path.join(MODEL_DIR, kind, "onnx", name)
    if not os.path.exists(path):
        raise SystemExit(f"model file missing: {path} (run tools/download_models.py / rebuild the image)")
    return path


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Keep resident memory flat: no growing arena / memory-pattern caches across
    # variable-length text batches.
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class TextEmbedder:
    def __init__(self):
        t0 = time.time()
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "text", "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_TOKENS)
        self.tok.no_padding()
        self.sess = _session(_model_file("text"))
        self.input_names = {i.name for i in self.sess.get_inputs()}
        log.info("text model loaded (%s) in %.1fs, rss=%.0fMB", MODEL_PRECISION, time.time() - t0, cur_rss_mb())

    def embed(self, texts: list[str], prefix: str = TEXT_PREFIX) -> np.ndarray:
        """nomic v1.5 recipe: prefix -> mean pool (mask) -> layer_norm -> L2 normalise. Full 768 dims."""
        out = []
        for t in texts:  # batch of 1: no padding waste, smallest activation memory
            enc = self.tok.encode(prefix + (t or ""))
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feeds)[0]  # (1, seq, 768)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            mu = pooled.mean(axis=-1, keepdims=True)
            var = pooled.var(axis=-1, keepdims=True)
            pooled = (pooled - mu) / np.sqrt(var + 1e-5)  # F.layer_norm, no affine
            out.append(l2norm(pooled)[0])
        return np.stack(out) if out else np.zeros((0, DIM), np.float32)


class VisionEmbedder:
    def __init__(self):
        t0 = time.time()
        self.sess = _session(_model_file("vision"))
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name  # last_hidden_state
        log.info("vision model loaded (%s) in %.1fs, rss=%.0fMB", MODEL_PRECISION, time.time() - t0, cur_rss_mb())

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        """CLIPImageProcessor as configured in preprocessor_config.json:
        convert RGB, resize to 224x224 (bicubic; config size is {height:224,width:224}
        so HF resizes to exactly 224x224), center-crop 224 (no-op), rescale 1/255,
        normalise with CLIP mean/std. Transparent pixels are composited on white."""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((arr - CLIP_MEAN) / CLIP_STD)[None].astype(np.float32)

    def embed(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img)
        hidden = self.sess.run([self.output_name], {self.input_name: x})[0]  # (1, 197, 768)
        return l2norm(hidden[:, 0])[0]  # CLS token, L2 normalised


# --------------------------------------------------------------------------- images

def image_fetch_url(url: str | None) -> str | None:
    """Append width=512 only for cdn.shopify.com; everything else is fetched as-is."""
    if not url or not str(url).strip():
        return None
    url = str(url).strip()
    if url.startswith("//"):
        url = "https:" + url
    host = (urlparse(url).hostname or "").lower()
    if host == "cdn.shopify.com" and not re.search(r"[?&]width=", url):
        url += ("&" if "?" in url else "?") + f"width={IMAGE_WIDTH_PARAM}"
    return url


_http = None


def _session_http():
    global _http
    if _http is None:
        import requests
        from requests.adapters import HTTPAdapter
        _http = requests.Session()
        _http.headers.update({"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"})
        ad = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=1)
        _http.mount("https://", ad)
        _http.mount("http://", ad)
    return _http


def fetch_image_bytes(url: str) -> tuple[bytes | None, str | None]:
    """Returns (bytes, error). Never raises."""
    try:
        r = _session_http().get(url, timeout=IMAGE_TIMEOUT, stream=True)
        if r.status_code != 200:
            r.close()
            return None, f"http {r.status_code}"
        buf = io.BytesIO()
        for chunk in r.iter_content(64 * 1024):
            buf.write(chunk)
            if buf.tell() > IMAGE_MAX_BYTES:
                r.close()
                return None, "too large"
        return buf.getvalue(), None
    except Exception as e:  # network errors, timeouts, bad URLs
        return None, type(e).__name__


def decode_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))  # lazy: header only
    if img.format == "JPEG":
        img.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))  # cheap DCT downscale for big JPEGs
    elif img.size[0] * img.size[1] > IMAGE_MAX_PIXELS:
        # Non-JPEG decode cost is w*h*4 bytes; refuse giant PNG/WebP/GIF rather than risk OOM.
        raise ValueError("image too many pixels")
    img.seek(0) if getattr(img, "is_animated", False) else None
    img.load()
    return img


# --------------------------------------------------------------------------- halfvec

def to_vec_literal(v: np.ndarray) -> str:
    """pgvector text literal '[a,b,...]'; cast with ::halfvec(768) in SQL.
    Values are rounded to float16 first so what we send == what is stored."""
    h = v.astype(np.float16).astype(np.float32)
    return "[" + ",".join(f"{x:.5g}" for x in h.tolist()) + "]"


# --------------------------------------------------------------------------- sources/sinks

class DryRunSource:
    def __init__(self, path: str):
        self.rows = []
        if path.endswith(".csv"):
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows.append(r)
        else:
            with open(path, encoding="utf-8") as fh:
                self.rows = [json.loads(l) for l in fh if l.strip()]
        for r in self.rows:
            r.setdefault("want_hash", "dry-run")
        self.pos = 0
        log.info("DRY_RUN: %d rows from local file (no database connection)", len(self.rows))

    def fetch(self, n: int) -> list[dict]:
        out = self.rows[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def write(self, results: list[dict]):
        os.makedirs(os.path.dirname(DRY_RUN_OUTPUT) or ".", exist_ok=True)
        with open(DRY_RUN_OUTPUT, "a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "id": r["id"],
                    "text_embedding": r["text_vec"].astype(np.float16).astype(float).round(5).tolist(),
                    "image_embedding": None if r["image_vec"] is None else r["image_vec"].astype(np.float16).astype(float).round(5).tolist(),
                    "image_vec_source": r["image_src"],
                    "image_attempted": bool(r.get("image_attempted")),
                    "image_embedded_at_would_set": bool(r.get("image_attempted")),
                    "image_error": r.get("image_error"),
                    "text_hash": r["want_hash"],
                    "t_text_ms": r["t_text_ms"], "t_img_fetch_ms": r.get("t_img_fetch_ms"),
                    "t_img_embed_ms": r.get("t_img_embed_ms"),
                }) + "\n")

    def close(self):
        pass


class PostgresSource:
    """Direct Postgres via psycopg 3. Keyset pagination by id so a row that keeps
    failing can't wedge the loop within one process run."""

    def __init__(self, dsn: str):
        import psycopg
        self.psycopg = psycopg
        # prepare_threshold=None keeps it safe behind Supabase's transaction pooler (port 6543)
        self.conn = psycopg.connect(dsn, autocommit=False, prepare_threshold=None,
                                    connect_timeout=20, application_name="ao-embed-worker")
        with self.conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
        self.conn.commit()
        self.last_id = None
        log.info("connected to Postgres via DATABASE_URL")

    def fetch(self, n: int) -> list[dict]:
        q = (f"SELECT id::text, embed_text, want_hash, have_hash, main_image_url FROM {QUEUE_VIEW} "
             "WHERE (have_hash IS NULL OR have_hash <> want_hash) "
             + ("AND id > %s::uuid " if self.last_id else "") + "ORDER BY id LIMIT %s")
        params = (self.last_id, n) if self.last_id else (n,)
        with self.conn.cursor() as cur:
            cur.execute(q, params)
            cols = [d.name for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        self.conn.commit()
        if rows:
            self.last_id = rows[-1]["id"]
        return rows

    def write(self, results: list[dict]):
        # One transaction per batch. text_hash is assigned last in the SET list; the
        # UPDATE is atomic, so a row is either fully written (and marked done) or untouched.
        #
        # image_embedded_at:
        #   no URL          -> NULL (not attempted)
        #   URL + success   -> now(), image_embedding = vector
        #   URL + fail      -> now(), image_embedding = NULL  (dead URL; do not retry)
        # Find failed attempts: image_embedded_at IS NOT NULL AND image_embedding IS NULL
        sql = (f"UPDATE {TARGET_TABLE} SET "
               "text_embedding = %s::halfvec(768), "
               "image_embedding = %s::halfvec(768), "
               "embedded_at = now(), "
               "image_embedded_at = CASE WHEN %s THEN now() ELSE NULL END, "
               "image_vec_source = %s, "
               "text_hash = %s "
               "WHERE id = %s::uuid")
        params = []
        for r in results:
            has_img = r["image_vec"] is not None
            # Attempted if we had a URL (success or fail). image_attempted set in process_batch.
            attempted = bool(r.get("image_attempted"))
            params.append((to_vec_literal(r["text_vec"]),
                           to_vec_literal(r["image_vec"]) if has_img else None,
                           attempted,
                           r["image_src"] if has_img else None,
                           r["want_hash"], r["id"]))
        with self.conn.transaction():
            with self.conn.cursor() as cur:
                cur.executemany(sql, params)
        # psycopg3: conn.transaction() commits on exit (autocommit off -> savepoint-free top-level tx)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class PostgrestSource:
    """Fallback: SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY via PostgREST.
    PostgREST can't filter have_hash <> want_hash (column-to-column), so pending
    rows are filtered client-side while paging by id. No multi-row transactions:
    each PATCH is atomic per row (text_hash is written in the same PATCH)."""

    def __init__(self, url: str, key: str):
        import requests
        self.base = url.rstrip("/") + "/rest/v1"
        self.s = requests.Session()
        self.s.headers.update({"apikey": key, "Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
        self.schema_view = QUEUE_VIEW.split(".")[-1]
        self.table = TARGET_TABLE.split(".")[-1]
        self.last_id = None
        log.info("using PostgREST via SUPABASE_URL")

    def fetch(self, n: int) -> list[dict]:
        out = []
        while len(out) < n:
            params = {"select": "id,embed_text,want_hash,have_hash,main_image_url", "order": "id.asc", "limit": "1000"}
            if self.last_id:
                params["id"] = f"gt.{self.last_id}"
            r = self.s.get(f"{self.base}/{self.schema_view}", params=params, timeout=60)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            for row in page:
                self.last_id = row["id"]
                if row.get("have_hash") is None or row["have_hash"] != row["want_hash"]:
                    out.append(row)
                    if len(out) >= n:
                        break
        return out

    def write(self, results: list[dict]):
        now = datetime.now(timezone.utc).isoformat()
        for r in results:
            has_img = r["image_vec"] is not None
            attempted = bool(r.get("image_attempted"))
            body = {"text_embedding": to_vec_literal(r["text_vec"]),
                    "image_embedding": to_vec_literal(r["image_vec"]) if has_img else None,
                    "embedded_at": now,
                    # Same semantics as PostgresSource: stamp attempt even when vector is null.
                    "image_embedded_at": now if attempted else None,
                    "image_vec_source": r["image_src"] if has_img else None,
                    "text_hash": r["want_hash"]}
            resp = self.s.patch(f"{self.base}/{self.table}", params={"id": f"eq.{r['id']}"},
                                data=json.dumps(body), headers={"Prefer": "return=minimal"}, timeout=60)
            resp.raise_for_status()

    def close(self):
        pass


def make_source():
    if DRY_RUN:
        return DryRunSource(DRY_RUN_INPUT)
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if dsn:
        return PostgresSource(dsn)
    url, key = os.environ.get("SUPABASE_URL", "").strip(), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if url and key:
        return PostgrestSource(url, key)
    raise SystemExit("no DB configured: set DATABASE_URL (preferred) or SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY, or DRY_RUN=1")


# --------------------------------------------------------------------------- main loop

class Models:
    def __init__(self):
        self.text = None
        self.vision = None
        if LOAD_MODE == "both":
            self.text = TextEmbedder()
            self.vision = VisionEmbedder()

    @staticmethod
    def _drop(obj):
        """Force ORT session teardown so sequential mode can reclaim RSS between passes."""
        if obj is None:
            return
        for attr in ("sess", "tok"):
            if hasattr(obj, attr):
                setattr(obj, attr, None)
        gc.collect()
        malloc_trim()

    def get_text(self):
        if self.text is None:
            if LOAD_MODE == "sequential" and self.vision is not None:
                self._drop(self.vision); self.vision = None
            self.text = TextEmbedder()
        return self.text

    def get_vision(self):
        if self.vision is None:
            if LOAD_MODE == "sequential" and self.text is not None:
                self._drop(self.text); self.text = None
            self.vision = VisionEmbedder()
        return self.vision


def process_batch(rows: list[dict], models: Models, pool: ThreadPoolExecutor, stats: dict) -> list[dict]:
    # 1) kick off image downloads in background threads (I/O overlaps with text inference)
    futures = {}
    for r in rows:
        u = image_fetch_url(r.get("main_image_url"))
        r["_fetch_url"] = u
        if u:
            futures[r["id"]] = (time.time(), pool.submit(fetch_image_bytes, u))

    results = []
    # 2) text pass
    tm = models.get_text()
    for r in rows:
        if STOP:
            break
        t0 = time.perf_counter()
        vec = tm.embed([r.get("embed_text") or ""])[0]
        results.append({"id": r["id"], "want_hash": r["want_hash"], "text_vec": vec,
                        "t_text_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "image_vec": None, "image_src": None,
                        # Set True only after a real fetch/embed attempt (not merely because a URL existed).
                        "image_attempted": False,
                        "_fetch_url": r["_fetch_url"]})
    # 3) image pass
    vm = models.get_vision() if any(res["_fetch_url"] for res in results) else None
    for res in results:
        u = res.pop("_fetch_url")
        if not u:
            stats["img_missing"] += 1
            continue
        t_start, fut = futures[res["id"]]
        tf = time.perf_counter()
        data, err = fut.result()
        res["t_img_fetch_ms"] = round((time.perf_counter() - tf) * 1000, 1)  # wait time beyond overlap
        if data is None:
            stats["img_failed"] += 1
            res["image_attempted"] = True  # URL present; fetch failed — stamp image_embedded_at
            res["image_error"] = err
            log.warning("image fetch failed id=%s host=%s err=%s", res["id"], urlparse(u).hostname, err)
            continue
        try:
            t0 = time.perf_counter()
            img = decode_image(data)
            res["image_vec"] = vm.embed(img)
            res["image_src"] = u
            res["image_attempted"] = True
            res["t_img_embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            stats["img_ok"] += 1
        except Exception as e:
            stats["img_failed"] += 1
            res["image_attempted"] = True  # URL present; decode/embed failed — stamp image_embedded_at
            res["image_error"] = type(e).__name__
            log.warning("image decode/embed failed id=%s host=%s err=%s", res["id"], urlparse(u).hostname, type(e).__name__)
    for fut in futures.values():  # rows skipped by STOP
        fut[1].cancel()
    return results


def idle_forever():
    log.info("done (queue drained or LIMIT reached), idle forever; delete/suspend this worker in Render")
    while not STOP:
        time.sleep(30)
    log.info("exiting after signal")
    sys.exit(0)


def main():
    limit_label = "none (drain)" if LIMIT is None else str(LIMIT)
    log.info("embed worker start: dry_run=%s precision=%s load_mode=%s limit=%s batch=%d threads=%d max_tokens=%d idle_after_done=%s",
             DRY_RUN, MODEL_PRECISION, LOAD_MODE, limit_label, BATCH_SIZE, ORT_THREADS, MAX_TOKENS, IDLE_AFTER_DONE)
    if DRY_RUN and os.path.exists(DRY_RUN_OUTPUT):
        os.remove(DRY_RUN_OUTPUT)
    t_start = time.time()
    models = Models()
    source = make_source()
    stats = {"done": 0, "img_ok": 0, "img_failed": 0, "img_missing": 0, "batches": 0}
    pool = ThreadPoolExecutor(max_workers=FETCH_WORKERS)
    try:
        # LIMIT unset/empty => drain until queue empty. LIMIT set (test-only) => stop after N rows.
        while (LIMIT is None or stats["done"] < LIMIT) and not STOP:
            want = BATCH_SIZE if LIMIT is None else min(BATCH_SIZE, LIMIT - stats["done"])
            rows = source.fetch(want)
            if not rows:
                log.info("queue empty")
                break
            results = process_batch(rows, models, pool, stats)
            if results:
                source.write(results)
            stats["done"] += len(results)
            stats["batches"] += 1
            del results, rows
            gc.collect()
            malloc_trim()
            el = time.time() - t_start
            log.info("progress done=%d/%s img_ok=%d img_failed=%d img_missing=%d elapsed=%.0fs rate=%.2f rows/s rss=%.0fMB peak=%.0fMB",
                     stats["done"], LIMIT if LIMIT is not None else "drain",
                     stats["img_ok"], stats["img_failed"], stats["img_missing"], el,
                     stats["done"] / max(el, 1e-9), cur_rss_mb(), peak_rss_mb())
    except Exception as e:
        log.error("fatal: %s: %s", type(e).__name__, redact(str(e))[:500])
        source.close()
        raise SystemExit(1)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    source.close()
    log.info("finished: %s total_elapsed=%.0fs peak_rss=%.0fMB", json.dumps(stats), time.time() - t_start, peak_rss_mb())
    if STOP:
        sys.exit(0)
    # Idle forever so Render does not restart into another run; Martin deletes the service.
    if IDLE_AFTER_DONE:
        idle_forever()


if __name__ == "__main__":
    main()
