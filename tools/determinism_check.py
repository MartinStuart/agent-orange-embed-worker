"""Embed all sample texts sequentially twice with the int8 text model under the worker's
session settings; report self-consistency and agreement with the fp32 reference.
usage: MALLOC_TUNE=0|1 ORT_ARENA=0|1 python tools/determinism_check.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MODEL_PRECISION"] = "int8"
import worker  # noqa
import onnxruntime as ort
arena = os.environ.get("ORT_ARENA", "0") == "1"
orig = worker._session
def sess(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = arena; so.enable_mem_pattern = arena
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
worker._session = sess
rows = [json.loads(l) for l in open("sample/sample.jsonl")]
R = np.load("out/cache/text_fp32_ref.npy")
tm = worker.TextEmbedder()
E1 = tm.embed([r["embed_text"] for r in rows]); E2 = tm.embed([r["embed_text"] for r in rows])
# fresh-session-per-row baseline
c1 = (E1 * R).sum(1); self_ = (E1 * E2).sum(1)
print(f"MALLOC_TUNE={os.environ.get('MALLOC_TUNE','1')} arena={arena}: vs fp32 mean={c1.mean():.4f} min={c1.min():.4f} | "
      f"pass1 vs pass2 min={self_.min():.4f}", flush=True)
