#!/usr/bin/env python3
"""Launch KnowFeat Web Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from knowfeat.web.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")
