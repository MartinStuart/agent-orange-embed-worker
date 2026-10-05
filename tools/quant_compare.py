"""Compare a quantised ONNX against fp32 on the sample: cosine, top-5 neighbour
overlap, ms/item (1 thread), RSS after load. One model per call to keep RAM low.
usage: python tools/quant_compare.py text|vision path/to/model.onnx"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MODEL_PRECISION"] = "fp32"
import worker  # noqa: E402

kind, path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open("sample/sample.jsonl")]
os.makedirs("out/cache", exist_ok=True)
if kind == "text":
    tm = worker.TextEmbedder.__new__(worker.TextEmbedder)
    from tokenizers import Tokenizer
    tm.tok = Tokenizer.from_file("models/text/tokenizer.json"); tm.tok.enable_truncation(max_length=worker.MAX_TOKENS); tm.tok.no_padding()
    def run(p):
        tm.sess = worker._session(p); tm.input_names = {i.name for i in tm.sess.get_inputs()}
        r0 = worker.cur_rss_mb(); t = time.perf_counter()
        E = tm.embed([r["embed_text"] for r in rows])
        return E, (time.perf_counter() - t) / len(rows), r0
else:
    cache = "out/cache/vision_inputs.npy"
    if not os.path.exists(cache):
        X = []
        for r in rows:
            u = worker.image_fetch_url(r["main_image_url"])
            if u:
                d, e = worker.fetch_image_bytes(u)
                X.append(worker.VisionEmbedder.preprocess(worker.decode_image(d))[0])
        np.save(cache, np.stack(X))
    X = np.load(cache)
    def run(p):
        s = worker._session(p); o = s.get_outputs()[0].name; r0 = worker.cur_rss_mb(); t = time.perf_counter()
        E = np.stack([worker.l2norm(s.run([o], {"pixel_values": x[None]})[0][:, 0])[0] for x in X])
        return E, (time.perf_counter() - t) / len(X), r0
ref_path = f"out/cache/{kind}_fp32_ref.npy"
if not os.path.exists(ref_path):
    E, _, _ = run(f"models/{kind}/onnx/model.onnx"); np.save(ref_path, E)
R = np.load(ref_path)
E, dt, rss = run(path)
c = (E * R).sum(1)
SR, SQ = R @ R.T, E @ E.T
ov = np.mean([len(set(np.argsort(-SR[i])[1:6]) & set(np.argsort(-SQ[i])[1:6])) / 5 for i in range(len(R))])
print(f"{kind} {os.path.basename(path):24s} {os.path.getsize(path)/2**20:4.0f}MB cos mean={c.mean():.4f} min={c.min():.4f} "
      f"top5-overlap={ov:.2f} {dt*1000:.0f} ms/item rss_after_load={rss:.0f}MB", flush=True)
