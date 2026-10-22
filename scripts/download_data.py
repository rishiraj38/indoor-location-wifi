"""Download and unzip UJIIndoorLoc into data/raw/."""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indoorloc.data import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"

if (RAW_DIR / "trainingData.csv").exists():
    print(f"already present: {RAW_DIR}")
else:
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW_DIR.parent)
    print(f"saved to {RAW_DIR}")
