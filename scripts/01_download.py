"""Download the CMS and OIG files and record their size and SHA-256.

The OIG exclusion list is replaced every month, so the manifest keeps the date and hash
of the copy the results were produced from.

Usage: python scripts/01_download.py
"""

import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(".part")
    print(f"downloading {path.name}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            tmp.rename(path)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            # data.cms.gov answers 429 when files are requested back to back.
            wait = 30 * 2**attempt
            print(f"  {err}, retrying in {wait} s")
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}")


def main() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(config.LEIE_URL, config.LEIE_PATH)]
    for y in config.YEARS:
        jobs.append((config.PART_B_URLS[y], config.part_b_path(y)))
        jobs.append((config.PART_D_URLS[y], config.part_d_path(y)))
    manifest_path = config.RESULTS / "data_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for url, path in jobs:
        fetch(url, path)
        digest = sha256(path)
        prev = old.get(path.name, {})
        if prev and prev["sha256"] != digest:
            print(f"warning: {path.name} differs from the copy in the manifest")
        manifest[path.name] = {
            "url": url,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "downloaded": prev.get("downloaded") if prev.get("sha256") == digest
            else dt.date.today().isoformat(),
        }
    config.RESULTS.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest)} files, manifest in {manifest_path}")


if __name__ == "__main__":
    main()
