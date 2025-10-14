import pytest
import sys, types

def test_inline_forbidden_by_cli(monkeypatch, tmp_path):
    # Stub heavy deps used by CLI to exit early after arg parsing
    monkeypatch.setenv("PYTHONHASHSEED","0")
    from facekit.cli import resolve_face_ids_v2_cli as cli

    argv = [
        "prog",
        "--input", str(tmp_path/"fake.mp4"),
        "--schema_version","2.1",
        "--emb-store","inline",
        "--output_global_json",
    ]
    (tmp_path/"fake.mp4").write_bytes(b"\x00")  # make it exist

    # Monkeypatch functions used after the error point, to be safe
    monkeypatch.setattr(cli, "ReaderCoordinator", lambda p: None)
    monkeypatch.setattr(cli, "generate_shot_features_json", lambda **kw: None)

    with pytest.raises(SystemExit):
        monkeypatch.setenv("PYTEST_CLI_BLOCK","1")
        monkeypatch.setattr(sys, "argv", argv)
        cli.main()
