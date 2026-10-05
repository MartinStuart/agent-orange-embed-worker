"""Build-time weight-only 8-bit quantisation (ORT MatMulNBits, blockwise) of the
fp32 nomic ONNX models. Weights int8 per 128-element block; activations stay fp32
(accuracy_level selects compute: 0/1 = fp32, 4 = int8 dot products).

usage: python tools/quantize.py MODEL_DIR [kind ...] [--acc N] [--block N] [--suffix S]
Writes MODEL_DIR/<kind>/onnx/model_w8<suffix>.onnx. Needs onnx + onnx_ir at build time only."""
import argparse, os, time
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("kinds", nargs="*", default=["text", "vision"])
ap.add_argument("--acc", type=int, default=4)
ap.add_argument("--block", type=int, default=128)
ap.add_argument("--suffix", default="")
a = ap.parse_args()
for kind in a.kinds:
    src = os.path.join(a.model_dir, kind, "onnx", "model.onnx")
    dst = os.path.join(a.model_dir, kind, "onnx", f"model_w8{a.suffix}.onnx")
    t = time.time()
    q = MatMulNBitsQuantizer(onnx.load(src), bits=8, block_size=a.block, is_symmetric=True,
                             accuracy_level=a.acc, op_types_to_quantize=("MatMul",))
    q.process()
    q.model.save_model_to_file(dst, use_external_data_format=False)
    print(f"{kind}: {dst} {os.path.getsize(dst)/2**20:.0f} MB in {time.time()-t:.0f}s", flush=True)
