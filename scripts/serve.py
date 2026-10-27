"""Docker/Render entry point with a validated platform port."""
import os
import sys
port = int(os.environ.get("PORT", "8501"))
if not 1 <= port <= 65535:
    raise ValueError("Invalid PORT")
os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py",
                         "--server.address=0.0.0.0", f"--server.port={port}"])
