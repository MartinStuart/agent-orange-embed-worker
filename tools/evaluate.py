"""Compare int8 vs fp32 dry-run outputs, report latencies, and run retrieval sanity
checks (search_query text -> sample texts/images). Local testing only.

Writes a human-readable summary to out/eval_report.txt as well as stdout.
"""
from __future__ import annotations
import json, os, sys, statistics as st
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("MODEL_PRECISION", "int8")
import worker  # noqa: E402


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def main():
    lines = []
    def out(s=""):
        print(s, flush=True); lines.append(s)

    a, b = load(os.path.join(HERE, "out/vectors_int8.jsonl")), load(os.path.join(HERE, "out/vectors_fp32.jsonl"))
    sample = {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, "sample/sample.jsonl")))}
    tc, ic = [], []
    for k in a:
        ta, tb = np.array(a[k]["text_embedding"]), np.array(b[k]["text_embedding"])
        tc.append(float(ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb))))
        if a[k]["image_embedding"] and b[k]["image_embedding"]:
            ia, ib = np.array(a[k]["image_embedding"]), np.array(b[k]["image_embedding"])
            ic.append(float(ia @ ib / (np.linalg.norm(ia) * np.linalg.norm(ib))))
    out(f"int8 vs fp32 cosine  text: mean={np.mean(tc):.4f} min={np.min(tc):.4f}   "
        f"image: mean={np.mean(ic):.4f} min={np.min(ic):.4f}  (n={len(tc)}/{len(ic)})")
    norms = [np.linalg.norm(np.array(r["text_embedding"])) for r in a.values()] + [
        np.linalg.norm(np.array(r["image_embedding"])) for r in a.values() if r["image_embedding"]]
    out(f"vector norms after float16 rounding: min={min(norms):.4f} max={max(norms):.4f}")
    for name, d in (("int8", a), ("fp32", b)):
        t = [r["t_text_ms"] for r in d.values()]
        ie = [r["t_img_embed_ms"] for r in d.values() if r.get("t_img_embed_ms")]
        out(f"{name} latency (ORT_THREADS=1, full core) text ms: median={st.median(t):.0f} mean={st.mean(t):.0f} max={max(t):.0f} | "
            f"image decode+preprocess+embed ms: median={st.median(ie):.0f} mean={st.mean(ie):.0f} max={max(ie):.0f}")
    nulls = [k for k, r in a.items() if r["image_embedding"] is None]
    out(f"rows with image_embedding NULL: {[(k[:8], a[k]['image_error'] or 'no image url') for k in nulls]}")
    out("non-shopify sources kept as-is: " + str(
        [r["image_vec_source"][:90] for r in a.values()
         if r["image_vec_source"] and "cdn.shopify.com" not in r["image_vec_source"]]))
    shopify = [r["image_vec_source"] for r in a.values()
               if r["image_vec_source"] and "cdn.shopify.com" in r["image_vec_source"]]
    out(f"shopify URLs with width=512: {sum('width=512' in u for u in shopify)}/{len(shopify)}")

    tm = worker.TextEmbedder()
    ids = list(a)
    T = np.array([a[k]["text_embedding"] for k in ids]); T /= np.linalg.norm(T, axis=1, keepdims=True)
    img_ids = [k for k in ids if a[k]["image_embedding"]]
    I = np.array([a[k]["image_embedding"] for k in img_ids]); I /= np.linalg.norm(I, axis=1, keepdims=True)
    title = lambda k: sample[k]["embed_text"].split(" | ")[0][:45]
    queries = [
        "search_query: blue linen sofa",
        "search_query: dangly gold earrings",
        "search_query: a cozy blanket for the bed",
        "search_query: coffee mug",
        "search_query: bed for my dog",
        "search_query: summer dress",
        "search_query: dark chocolate gift box",
        "search_query: sparkling rose wine",
        "search_query: floor lamp for the living room",
        "search_query: hot sauce for barbecue",
    ]
    # embed without double-prefix: TextEmbedder adds search_document by default; pass prefix=''
    for q in queries:
        qv = tm.embed([q[len("search_query: "):]], prefix="search_query: ")[0]
        st_ = T @ qv; si = I @ qv
        tt = [f"{title(ids[i])} ({st_[i]:.2f})" for i in np.argsort(-st_)[:3]]
        ti = [f"{title(img_ids[i])} ({si[i]:.3f})" for i in np.argsort(-si)[:3]]
        out(f"\nQ: {q}\n  text top3 : {tt}\n  image top3: {ti}")
    ranks = []
    for j, k in enumerate(img_ids):
        s = I @ np.array(a[k]["text_embedding"])
        ranks.append(int((s > s[j]).sum()) + 1)
    out(f"\ndoc-text -> own image rank among {len(img_ids)} images: "
        f"top1={sum(r==1 for r in ranks)} top3={sum(r<=3 for r in ranks)} median={st.median(ranks)}")

    # wall-time estimate for Starter (0.5 CPU ≈ 2× the full-core single-thread times)
    t_med = st.median([r["t_text_ms"] for r in a.values()])
    i_med = st.median([r["t_img_embed_ms"] for r in a.values() if r.get("t_img_embed_ms")])
    # fetch overlaps with text; assume net adds ~50 ms amortized on Starter
    per_row_full_core_s = (t_med + i_med + 50) / 1000
    per_row_starter_s = per_row_full_core_s * 2.0  # pessimistic 0.5-CPU duty
    out("\n--- wall-time estimate (int8, Starter 0.5 CPU, ORT_THREADS=1) ---")
    out(f"per-row (full core measured median): text={t_med:.0f}ms image={i_med:.0f}ms +~50ms fetch = {per_row_full_core_s*1000:.0f}ms")
    out(f"per-row Starter estimate (×2 for 0.5 CPU): {per_row_starter_s*1000:.0f}ms")
    for n in (500, 101_967):
        out(f"  {n:>7,} rows ≈ {n*per_row_starter_s/3600:.1f} hours "
            f"({n*per_row_starter_s/60:.0f} min) — plus idle after each LIMIT=500 slice")
    report = os.path.join(HERE, "out/eval_report.txt")
    open(report, "w").write("\n".join(lines) + "\n")
    out(f"\nwrote {report}")


if __name__ == "__main__":
    main()
