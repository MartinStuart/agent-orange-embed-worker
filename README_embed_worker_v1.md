# Agent Orange embed worker v1

Background worker that embeds offerings from `public.offering_embed_queue` with
**nomic-embed-text-v1.5** + **nomic-embed-vision-v1.5** (shared 768-d space) and
writes `halfvec(768)` vectors back. Built for Render **Background Worker /
Standard** (2 GB RAM, 1 CPU — 1c-2g). No torch — onnxruntime + tokenizers +
numpy + Pillow.

## What Martin does to deploy (no DB writes from this folder)

1. Push this directory as a GitHub repo (or a subfolder with `rootDir` set). Do
   **not** commit secrets, `.venv/`, `out/`, or `models/` (models download at
   Docker build). Reed handles GitHub auth / push.
2. In Render: **New → Blueprint** → select the repo → create. Plan is
   `standard` (2 GB / 1 CPU), type is `worker`, runtime is `docker`.
3. When prompted for `sync: false` secrets, paste **one** of:
   - `DATABASE_URL` — **preferred**. Use the Supabase **session** pooler
     (port 5432) or direct connection. Avoid the transaction pooler unless you
     keep `prepare_threshold=None` (already set in `worker.py`).
   - **or** `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (PostgREST fallback).
4. Leave `LIMIT` **unset** (Blueprint does not set it). Leave
   `MODEL_PRECISION=int8`, `IDLE_AFTER_DONE=1`. Deploy.
5. Watch logs. You should see model load, then `progress done=N/drain …`, then
   `queue empty` / `done … idle forever`.
6. After the queue drains the process **idles forever** so Render does not
   restart it. **Delete or suspend** the worker when done (Standard bills while
   idle). Do **not** design around repeated 500-row restarts — one deploy drains
   the full queue.

Blueprint: `render.yaml`. Image: `Dockerfile` (models baked at build).

## Env vars

| Var | Default | Notes |
|-----|---------|-------|
| `DATABASE_URL` | — | Preferred. Postgres DSN (session pooler). Never logged. |
| `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` | — | PostgREST fallback if no `DATABASE_URL`. |
| `LIMIT` | unset | **Test-only.** Unset/empty = drain until queue empty. Set e.g. `500` only for local/test caps. |
| `BATCH_SIZE` | `50` | Fetch + write transaction size. |
| `MODEL_PRECISION` | `int8` | `int8` (MatMulNBits w8) or `fp32`. Default int8 for full drain; use `fp32` for A/B on the same Standard box. |
| `ORT_THREADS` | `1` | Intra-op threads; 1 is fine on 1 CPU. |
| `LOAD_MODE` | `both` | `both` loads text+vision together (int8 and fp32 both fit Standard 2 GB). `sequential` if you need more headroom. |
| `IDLE_AFTER_DONE` | `1` | Sleep forever after drain (or after LIMIT if set). Required on Render so the service is not restarted. |
| `DRY_RUN` | `0` | `1` = read local JSONL/CSV, write vectors to a file, no DB. |
| `DRY_RUN_INPUT` / `DRY_RUN_OUTPUT` | `sample/sample.jsonl` / `out/vectors_<prec>.jsonl` | |
| `TARGET_TABLE` | `public.offering` | UPDATE target. |
| `QUEUE_VIEW` | `public.offering_embed_queue` | Columns: `id, embed_text, want_hash, have_hash, main_image_url`. |
| `MAX_TOKENS` | `512` | Truncation for text (model supports 8192; 512 keeps RAM/CPU in check). |

## Gotchas implemented

1. **Text prefix + pooling:** embed as `search_document: <embed_text>`; mean-pool
   with attention mask; `layer_norm` (no affine); **full 768 dims**; then L2.
2. **Vision:** CLIP resize/center-crop 224, CLIP mean/std; **CLS token** then L2.
3. **L2-normalise every vector** before write (and again after float16 round-trip
   the stored halfvec is ~unit length).
4. **Shopify images:** append `?width=512` (or `&width=512`) only for
   `cdn.shopify.com`. Other hosts fetched as-is, resized locally.
5. **Image attempt recording** (avoids retrying dead URLs):
   - **No URL** → `image_embedding` NULL, `image_embedded_at` NULL (not attempted).
   - **URL + success** → vector written, `image_embedded_at = now()`.
   - **URL + fetch/embed failure** → `image_embedding` NULL,
     `image_embedded_at = now()` so the attempt is recorded.
   - Find failures: `image_embedded_at is not null and image_embedding is null`.
6. **Write-back (one UPDATE, atomic):** `text_embedding`, `image_embedding`,
   `embedded_at=now()`, `image_embedded_at` per rules above,
   `image_vec_source` = exact URL fetched on success only,
   `text_hash = want_hash` (SET last in the same statement). Batches of ~50 in a
   transaction.
7. **Drain then idle:** unset `LIMIT` drains the queue in one process run; then
   `IDLE_AFTER_DONE` sleeps forever. Martin deletes the service. Job is still
   resumable via `text_hash` / `have_hash` if the process is killed mid-run.

## Why not the HF `model_quantized.onnx`?

Those files use `MatMulInteger` + `DynamicQuantizeLinear`. On this box’s
**AMX_INT8** Xeon (and under any CPU preemption / shared quota) they are
**non-deterministic**: self-cosine of the same input dropped as low as **-0.08**
when another thread shared the core. fp32 and **MatMulNBits** weight-only int8
stay at self-cos **1.0**.

So at Docker build we download the fp32 ONNX and run
`onnxruntime.quantization.matmul_nbits_quantizer` (8-bit, block 128,
`accuracy_level=4`) → `model_w8.onnx` for both text and vision. Activations stay
fp32; weights are int8. Measured vs fp32 on the 30-row sample:

| | cos mean | cos min | ms/item (1 thread, full core) |
|--|--|--|--|
| text w8 | 0.9971 | 0.9963 | ~263 |
| vision w8 | 0.9967 | 0.9769 | ~297 |

## Memory fit (Standard 2 GB / 1 CPU)

| Mode | Peak RSS (measured) | Verdict |
|------|---------------------|---------|
| int8 both loaded | **~469 MB** | Comfortable on Standard |
| fp32 both loaded | ~1039 MB | Fits Standard — use for A/B on the same box |
| fp32 sequential (unload between passes) | ~691 MB | Optional; not required on 2 GB |

**Verdict:** ship `MODEL_PRECISION=int8` + `LOAD_MODE=both` for the full drain.
For fp32 A/B against int8, set `MODEL_PRECISION=fp32` on the same Standard
worker (or a second service); both models loaded fit in 2 GB.

Glibc malloc is tuned (`MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=131072`) so
ORT activation buffers return to the OS between batches.

## Local dry-run

```bash
cd /workspace/embed-worker
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-build.txt
python tools/download_models.py          # ~1.2 GB under models/
python sample/build_sample.py            # ~30 rows from weconnect/ccp pulls
DRY_RUN=1 MODEL_PRECISION=int8 ORT_THREADS=1 python worker.py
DRY_RUN=1 MODEL_PRECISION=fp32 ORT_THREADS=1 python worker.py   # ~1 GB RSS
# Optional test cap (never for production drain):
DRY_RUN=1 LIMIT=5 MODEL_PRECISION=int8 python worker.py
python tools/evaluate.py                 # cos agreement + retrieval sanity
```

Sample includes Shopify + non-Shopify image hosts and one missing-image row.

## Wall-time estimate (int8, Standard 1 CPU)

Earlier Starter-like (50% duty-cycle) measurement was **~1.2 s/row**. On a full
1 CPU expect roughly **~0.5–0.8 s/row** (I/O and image hosts vary). Order of
magnitude for ~100k queue rows: **~15–25 hours** wall in one continuous drain,
then idle until you delete the service.

Throughput is dominated by MatMulNBits kernels (similar speed to fp32 on this
CPU); int8’s win is memory headroom and stable determinism.

## halfvec literal format

Writes the pgvector text form `'[v1,v2,…,v768]'::halfvec(768)` with values
pre-rounded through float16. Confirm with Claude that the `offering` columns are
`halfvec(768)` (not `vector(768)`) and that PostgREST accepts the same literal
string if the PostgREST path is used.

## Open questions for Claude

1. **halfvec cast** — confirm `::halfvec(768)` on `public.offering` and that
   `text_hash` / `embedded_at` / `image_embedded_at` / `image_vec_source` columns
   exist with the expected types.
2. **Connection** — prefer `DATABASE_URL` (session pooler). Confirm which URI to
   paste; transaction mode is OK with `prepare_threshold=None` (already).
3. **PostgREST path** — column-to-column `have_hash <> want_hash` is filtered
   client-side while paging; confirm the view is exposed to the service role and
   that PATCH on `offering` with a halfvec literal works (or stick to Postgres).
4. **Failed-image retries** — stamped rows
   (`image_embedded_at is not null and image_embedding is null`) are left alone
   by the queue (text_hash already matches). Say if you want a separate re-fetch
   path later.
5. **fp32 A/B** — Standard fits fp32 both-loaded; say if you want a second
   service or a temporary env flip on the same box for a sample comparison.

## Files

```
embed-worker/
  worker.py                     # main
  requirements.txt              # runtime
  requirements-build.txt        # builder (onnx + onnx-ir)
  Dockerfile                    # multi-stage; models baked in; LIMIT unset
  render.yaml                   # Blueprint: worker / standard / docker
  README_embed_worker_v1.md     # this file
  .dockerignore
  tools/download_models.py      # HF fetch + MatMulNBits → model_w8.onnx
  tools/quantize.py             # optional manual re-quant
  tools/evaluate.py             # dry-run metrics
  tools/cpulimit.py             # local CPU quota emulator
  sample/build_sample.py
  sample/sample.jsonl           # 30-row local fixture
```

Secrets never go in the repo or logs (DSN/JWT redacted by the logger).
