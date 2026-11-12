"""Web interface (FastAPI + plain ES-module front end) for PIEZO1 DECODE-MAGIK.

    app.py      -- FastAPI routes under /api, STATIC_ENDPOINTS for the GitHub Pages export
    content.py  -- pipeline stage descriptions (from the README and the code) for the Pipeline page
    params.py   -- every configs/*.yaml value and every script command-line option, with units
    demo.py     -- synthetic demo dataset as JSON; Python ROI and track analysis behind POST /api/roi, /api/tracks
    static/     -- front end; roi_engine.js, track_engine.js, psf_engine.js are JS ports tested against Python
"""
