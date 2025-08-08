# facekit/onnx/patching.py
import onnx
from onnx import checker

def patch_batch_dim_dynamic(in_path: str, out_path: str, batch_symbol: str = "N") -> None:
    """
    Make the *first dimension* (batch) of all graph inputs & outputs symbolic (dynamic).

    This removes ORT warnings like:
      Expected shape from model of {1,512} does not match actual shape of {K,512}

    Only touches the top-level ValueInfo (inputs/outputs) — internal shapes are left as-is.
    """
    model = onnx.load(in_path)

    def _make_dim0_symbolic_vi(vi):
        t = vi.type.tensor_type
        if not t.HasField("shape") or len(t.shape.dim) == 0:
            return
        d0 = t.shape.dim[0]
        # clear any fixed numeric batch (e.g., 1) and set to symbolic 'N'
        if d0.HasField("dim_value"):
            d0.ClearField("dim_value")
        d0.dim_param = batch_symbol

    # Patch all graph inputs & outputs
    for vi in model.graph.input:
        _make_dim0_symbolic_vi(vi)
    for vo in model.graph.output:
        _make_dim0_symbolic_vi(vo)

    # Validate & save
    checker.check_model(model)
    onnx.save(model, out_path)
