"""Process-wide installation and capacity controls; no user data is cached here."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
from threading import BoundedSemaphore, Lock

from lab.engines import EngineError, ROOT, r_probe, rscript_path

_install_lock = Lock()
_capacity = BoundedSemaphore(1)
MTEST_SHA256 = "e3deebf821b4fae0effe64bac2454821c12d5674d33e9e8c7fc9395b3fbe9128"


def prepare_runtime():
    # Streamlit calls this through cache_resource once per server process.
    # R's install lock also prevents concurrent installs across processes.
    with _install_lock:
        status = r_probe()
        if status.get("available"):
            return status
        archive = ROOT/"vendor/MTest_1.0.4.tar.gz"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != MTEST_SHA256:
            return {"available": False, "error": "Falló la integridad del paquete oficial MTest."}
        try:
            proc = subprocess.run([rscript_path(), "--vanilla", str(ROOT/"scripts/install_r.R")],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120)
            if proc.returncode:
                return {"available": False, "error": "No se pudo preparar MTest 1.0.4; verifique las dependencias del entorno."}
            return r_probe()
        except (EngineError, OSError, subprocess.TimeoutExpired):
            return {"available": False, "error": "R no está instalado o la preparación excedió 120 s."}


@contextmanager
def analysis_slot():
    if not _capacity.acquire(blocking=False):
        raise EngineError("El servidor está ejecutando otro análisis. Reintente en unos segundos.")
    try:
        yield
    finally:
        _capacity.release()
