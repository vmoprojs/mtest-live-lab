import os
from pathlib import Path
os.environ["R_LIBS_USER"] = str(Path(__file__).resolve().parents[1]/".r-library")
