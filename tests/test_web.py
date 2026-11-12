"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from piezo1_magik.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -----------------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402


def test_info_counts_are_real():
    i = client.get("/api/info").json()
    assert i["stages"] == 3 and i["n_scripts"] == len(list((ROOT / "scripts").glob("[0-9]*.py")))
    assert i["demo"]["tracks"] == 8 and i["demo"]["frames"] == 40


def test_params_lists_configs_and_options_with_units():
    p = client.get("/api/params").json()
    files = {c["file"] for c in p["configs"]}
    assert files == {"configs/decode_training.yaml", "configs/magik_training.yaml", "configs/pipeline.yaml"}
    magik = next(c for c in p["configs"] if c["file"].endswith("magik_training.yaml"))
    row = next(r for r in magik["rows"] if r["key"] == "data.max_spatial_distance")
    assert row["value"] == 30 and row["unit"] == "px" and abs(row["um"] - 3.9) < 1e-9
    s04 = next(s for s in p["scripts"] if s["script"] == "scripts/04_run_pipeline.py")
    assert any(o["name"] == "--roi_size" and o["default"] == 5 for o in s04["options"])


def test_roi_endpoint_matches_analyzer_and_validates():
    r = client.post("/api/roi", json={})
    assert r.status_code == 200
    out = r.json()
    assert out["computed_by"] == "python" and len(out["tracks"]) == 8
    placed = {e["track_id"] for e in client.get("/api/demo").json()["true_events"]}
    assert any(t["events"] for t in out["tracks"] if t["track_id"] in placed)
    assert client.post("/api/roi", json={"roi_size": 0}).status_code == 422
    r4 = client.post("/api/roi", json={"roi_size": 4})
    assert r4.status_code == 422 and "odd" in r4.text


def test_tracks_endpoint_gap_filling_rejoins_demo_tracks():
    out = client.post("/api/tracks", json={}).json()
    assert out["n_gt_tracks"] == 8
    assert out["filled"]["num_tracks"] <= out["linked"]["num_tracks"]
    assert np.mean(out["filled"]["fragmentation"]) <= np.mean(out["linked"]["fragmentation"])


def test_demo_payload_decodes():
    import base64
    d = client.get("/api/demo").json()
    T, H, W = d["shape"]
    assert len(base64.b64decode(d["calcium_b64"])) == 2 * T * H * W
    assert len(base64.b64decode(d["piezo1_u8_b64"])) == T * H * W


def _close(a, b, path="root"):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and not isinstance(b, bool):
        assert a is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b)), f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a} vs {b}"


def test_engine_fixture_is_current():
    """tests/fixtures/engine_cases.json (used by the node tests) still matches the Python code."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    stored = json.loads((ROOT / "tests" / "fixtures" / "engine_cases.json").read_text())
    _close(json.loads(json.dumps(make_engine_fixture.build())), stored)
