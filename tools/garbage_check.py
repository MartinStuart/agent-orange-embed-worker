"""Reproduce: does int8 text output depend on prior heap contents (uninitialised read)?
Fill+free heap with garbage before/while embedding. Optional ORT config via env:
DISABLE_PREPACK=1, OPT=all|basic|none."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MODEL_PRECISION"] = "int8"
import worker  # noqa
import onnxruntime as ort
def sess(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
    if os.environ.get("DISABLE_PREPACK") == "1":
        so.add_session_config_entry("session.disable_prepacking", "1")
    opt = os.environ.get("OPT", "all")
    so.graph_optimization_level = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                   "ext": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[opt]
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
worker._session = sess
def garbage():
    # many medium blocks of non-zero junk, then free -> stays in heap when not mmapped
    junk = [np.full(256 * 1024, 0x7f7f7f7f, dtype=np.int32) for _ in range(200)]
    del junk
rows = [json.loads(l) for l in open("sample/sample.jsonl")]
R = np.load("out/cache/text_fp32_ref.npy")
garbage()
tm = worker.TextEmbedder()
E = []
for r in rows:
    garbage()
    E.append(tm.embed([r["embed_text"]])[0])
c = (np.stack(E) * R).sum(1)
print(f"MALLOC_TUNE={os.environ.get('MALLOC_TUNE','1')} PREPACK_OFF={os.environ.get('DISABLE_PREPACK','0')} OPT={os.environ.get('OPT','all')}: "
      f"vs fp32 mean={c.mean():.4f} min={c.min():.4f} n_bad(<0.9)={(c<0.9).sum()}", flush=True)
