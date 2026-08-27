"""
DepthWizard Production Server for Hugging Face Spaces (FastAPI + Three.js 3D WebGL Dashboard)
"""

import os
import uvicorn
from api_server import app

# Hugging Face Spaces provides PORT=7860
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[DepthWizard] Launching full 3D WebGL application on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
