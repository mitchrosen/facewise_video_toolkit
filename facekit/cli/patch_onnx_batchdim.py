# facekit/cli/patch_onnx_batchdim.py
import argparse
from pathlib import Path
from facekit.onnx.patching import patch_batch_dim_dynamic

def main():
    p = argparse.ArgumentParser(description="Patch ONNX to use dynamic batch dim on inputs/outputs.")
    p.add_argument("--in", dest="in_path", required=True, help="Path to source .onnx")
    p.add_argument("--out", dest="out_path", required=True, help="Path to write patched .onnx")
    p.add_argument("--symbol", dest="symbol", default="N", help="Symbolic batch name to use (default: N)")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patch_batch_dim_dynamic(str(in_path), str(out_path), batch_symbol=args.symbol)
    print(f"✅ Patched dynamic batch dimension written to: {out_path}")

if __name__ == "__main__":
    main()
