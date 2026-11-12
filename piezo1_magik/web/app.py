"""FastAPI web app for PIEZO1 DECODE-MAGIK: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8209, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The POST
routes (/roi, /tracks) are answered in the browser by static/static_api.js on the static site; /models needs PyTorch
and exists only in the local app.
"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from piezo1_magik.web import content, demo, params

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="PIEZO1 DECODE-MAGIK")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary and real counts for the home page."""
    p = params.all_params()
    d = demo.demo()
    return {"title": "PIEZO1 DECODE-MAGIK", "stages": len(content.STAGES),
            "n_scripts": len(list((ROOT / "scripts").glob("[0-9]*.py"))),
            "n_config_values": p["n_config_values"], "n_options": p["n_options"],
            "demo": {"frames": int(d["calcium"].shape[0]), "height": int(d["calcium"].shape[1]),
                     "width": int(d["calcium"].shape[2]), "tracks": int(d["tracks"]["track_id"].nunique()),
                     "detections": int(len(d["tracks"])), "events": len(d["events"])},
            "facts": content.FACTS, "roi_defaults": demo.ROI_DEFAULTS, "track_defaults": demo.TRACK_DEFAULTS}


@app.get("/api/pipeline")
def pipeline():
    return {"stages": content.STAGES, "facts": content.FACTS}


@app.get("/api/params")
def all_params():
    return params.all_params()


@app.get("/api/demo")
def demo_data():
    return demo.demo_json()


class RoiParams(BaseModel):
    roi_size: int = Field(5, ge=1, le=31)
    baseline_method: str = "percentile"
    baseline_percentile: float = Field(10.0, ge=0, le=100)
    threshold: float = Field(2.0, ge=0)
    min_duration: int = Field(2, ge=1)
    min_distance: int = Field(5, ge=1)

    @field_validator("roi_size")
    @classmethod
    def odd(cls, v):
        if v % 2 == 0:
            raise ValueError("roi_size must be an odd number of pixels")
        return v

    @field_validator("baseline_method")
    @classmethod
    def method(cls, v):
        if v not in ("percentile", "rolling", "mean"):
            raise ValueError("baseline_method must be percentile, rolling or mean")
        return v


class TrackParams(BaseModel):
    max_temporal_gap: int = Field(2, ge=1, le=40)
    max_spatial_distance: float = Field(30.0, gt=0)
    max_gap: int = Field(10, ge=1, le=40)
    max_distance: float = Field(30.0, gt=0)


@app.post("/api/roi")
def roi(p: RoiParams):
    """CalciumROIAnalyzer on the demo's tracks (Python)."""
    return demo.run_roi(p.model_dump())


@app.post("/api/tracks")
def tracks(p: TrackParams):
    """Candidate graph, oracle linking, greedy_linking and gap filling on the demo's detections (Python)."""
    return demo.run_tracks(p.model_dump())


@app.get("/api/models")
def models(base_channels: Optional[int] = 32):
    """Build the (untrained) DECODENet and MAGIKNet from the training configs and report their size and output
    shapes on a demo-sized input. Needs PyTorch; local app only."""
    try:
        import torch
        from piezo1_magik.models.decode_net import DECODENet
        from piezo1_magik.models.magik_gnn import MAGIKNet
    except ImportError as e:
        raise HTTPException(503, f"PyTorch is not installed: {e}")
    torch.manual_seed(0)
    dec = DECODENet(base_channels=base_channels).eval()
    with torch.no_grad():
        out = dec(torch.zeros(1, 3, 64, 64))
    mag = MAGIKNet(node_features=5, edge_features=4, hidden_dim=64, num_layers=3)
    count = lambda m: int(sum(p.numel() for p in m.parameters()))  # noqa: E731
    return {"torch": torch.__version__, "decode": {"parameters": count(dec), "input": [1, 3, 64, 64],
                                                    "outputs": {k: list(v.shape) for k, v in out.items()}},
            "magik": {"parameters": count(mag), "hidden_dim": 64, "num_layers": 3},
            "note": "untrained networks built from the code; trained checkpoints are not part of the repo"}


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/pipeline", "/params", "/demo"]
