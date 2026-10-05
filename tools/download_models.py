"""Download nomic-embed-text/vision v1.5 ONNX + tokenizers into MODEL_DIR, then
produce MatMulNBits weight-only int8 (model_w8.onnx) for both.

Why not HF model_quantized.onnx: those use MatMulInteger + DynamicQuantizeLinear and
are non-deterministic under CPU preemption on AMX_INT8 CPUs (self-cos can drop below
0). Render Starter (0.5 CPU) preempts constantly. MatMulNBits keeps activations in
fp32 and is stable.

usage: python tools/download_models.py [/path/to/models]
"""
from __future__ import annotations
import os, sys, urllib.request

TEXT_REV = "e9b6763023c676ca8431644204f50c2b100d9aab"
VISION_REV = "e3a725bce72db07ca4adb1d83da08903f3ee02f8"
TEXT = f"https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/{TEXT_REV}"
VISION = f"https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/{VISION_REV}"

TEXT_FILES = [
    "onnx/model.onnx",
    "tokenizer.json", "tokenizer_config.json", "config.json",
    "1_Pooling/config.json", "special_tokens_map.json", "vocab.txt",
]
VISION_FILES = [
    "onnx/model.onnx",
    "preprocessor_config.json", "config.json",
]


def fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip {dest}", flush=True)
        return
    print(f"GET {url} -> {dest}", flush=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def matmul_nbits(src: str, dst: str, bits: int = 8, block: int = 128, accuracy_level: int = 4) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"skip {dst}", flush=True)
        return
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    print(f"MatMulNBits {src} -> {dst} (bits={bits} block={block} acc={accuracy_level})", flush=True)
    q = MatMulNBitsQuantizer(onnx.load(src), bits=bits, block_size=block, is_symmetric=True,
                             accuracy_level=accuracy_level, op_types_to_quantize=("MatMul",))
    q.process()
    q.model.save_model_to_file(dst, use_external_data_format=False)


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    for f in TEXT_FILES:
        fetch(f"{TEXT}/{f}", os.path.join(model_dir, "text", f))
    for f in VISION_FILES:
        fetch(f"{VISION}/{f}", os.path.join(model_dir, "vision", f))
    for kind in ("text", "vision"):
        matmul_nbits(os.path.join(model_dir, kind, "onnx", "model.onnx"),
                     os.path.join(model_dir, kind, "onnx", "model_w8.onnx"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
