import numpy as np
import pytest
from facekit.output.json_v2 import EmbeddingCollector
from facekit.output.json_v2 import ObservationsCollector

def test_embedding_sidecar_atomic(tmp_path):
    c = EmbeddingCollector("sidecar", dim=3)
    import numpy as np
    for i in range(5):
        c.assign(np.ones(3, dtype=np.float32))
    p = tmp_path/"e.npz"
    desc = c.finalize_sidecar(p)
    arr = np.load(desc["path"])["embeddings"]
    assert arr.shape == (5,3)

def test_observations_sidecar_atomic(tmp_path):
    oc = ObservationsCollector()
    # append empty → should still write a loadable npz with empty array
    out = tmp_path/"o.npz"
    desc = oc.finalize_sidecar(out)
    with np.load(desc["path"]) as z:
        assert "observations" in z
