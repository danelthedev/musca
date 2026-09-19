"""Download MaleCNS v1.0 raw files (~1.1 GB). stdlib only, tiny RAM.

    .venv/bin/python downloader.py
"""
import os
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
OUT = Path(__file__).resolve().parent.parent / "data" / "malecns"
RAW = OUT / "raw"


def download(name, fname):
    RAW.mkdir(parents=True, exist_ok=True)
    dest, url = RAW / fname, BASE + fname
    have = dest.stat().st_size if dest.exists() else 0
    req0 = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req0, timeout=30) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception:
        total = 0
    if total and have == total:
        print(f"  {name:18s} already complete ({have / 1e6:.0f} MB)")
        return dest
    # ponytail: Range resume when partial exists
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    print(f"  {name:18s} downloading {total / 1e6:.0f} MB" + (f" (resume {have / 1e6:.0f} MB)" if have else ""))
    with urllib.request.urlopen(req, timeout=60) as r:
        resumed = have and "206" in str(r.status)
        mode = "ab" if resumed else "wb"
        if not resumed:
            have = 0
        done = have
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {100 * done / total:5.1f}%  {done / 1e6:7.0f} MB", end="")
    print()
    return dest


def main():
    print("Downloading MaleCNS v1.0 (CC-BY, Janelia FlyEM / Cambridge / Google)")
    paths = {k: download(k, v) for k, v in FILES.items()}
    print("\nDone:")
    for k, p in paths.items():
        print(f"  {k:18s} {p.stat().st_size / 1e6:7.0f} MB  {p}")
    print("Next: .venv/bin/python process.py")


if __name__ == "__main__":
    main()
