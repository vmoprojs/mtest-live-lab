import os
import urllib.request
port = int(os.environ.get("PORT", "8501"))
with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=4) as response:
    assert response.status == 200 and response.read() == b"ok"
