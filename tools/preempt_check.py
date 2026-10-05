"""Does int8 output get corrupted when the inference thread is preempted on the same
core? Spins N busy threads (numpy work that releases the GIL) during text embedding.
usage: taskset -c K python tools/preempt_check.py [n_spinners] [text|vision] [model_file]"""
import json, os, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
kind = sys.argv[2] if len(sys.argv) > 2 else "text"
mfile = sys.argv[3] if len(sys.argv) > 3 else None
os.environ["MODEL_PRECISION"] = "int8"
import worker  # noqa
if mfile:
    worker._model_file = lambda k: os.path.join(worker.MODEL_DIR, k, "onnx", mfile)
stop = False
def spin():
    a = np.random.rand(200, 200)
    while not stop:
        a = a @ a.T; a /= np.abs(a).max()  # releases GIL inside BLAS-free numpy matmul
        time.sleep(0)
ths = [threading.Thread(target=spin, daemon=True) for _ in range(n)]
rows = [json.loads(l) for l in open("sample/sample.jsonl")]
if kind == "text":
    R = np.load("out/cache/text_fp32_ref.npy"); m = worker.TextEmbedder()
    f = lambda: np.stack([m.embed([r["embed_text"]])[0] for r in rows])
else:
    R = np.load("out/cache/vision_fp32_ref.npy"); X = np.load("out/cache/vision_inputs.npy"); m = worker.VisionEmbedder()
    f = lambda: np.stack([worker.l2norm(m.sess.run([m.output_name], {m.input_name: x[None]})[0][:, 0])[0] for x in X])
base = f()
for t in ths: t.start()
E = f()
stop = True
c0, c = (base * R).sum(1), (E * R).sum(1)
print(f"{kind} {mfile or 'default'} spinners={n}: quiet vs fp32 min={c0.min():.4f} | under preemption min={c.min():.4f} "
      f"bad(<0.9)={(c<0.9).sum()} | quiet-vs-preempted self min={(base*E).sum(1).min():.4f}", flush=True)
