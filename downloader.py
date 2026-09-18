"""Download MaleCNS v1.0 raw files (~1.1 GB). Download only, tiny RAM.

    python downloader.py

Outputs into data/malecns/raw/*.feather. Then run: python process.py
"""

from pathlib import Path

import requests

BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
        "connectome-data/flat-connectome/")
FILES = {
    "annotations":       "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights":           "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

OUT = Path("data/malecns")
RAW = OUT / "raw"


def download(name: str, fname: str) -> Path:
    """Stream a file to disk, resuming if a partial copy exists."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest, url = RAW / fname, BASE + fname

    total = int(requests.head(url, timeout=30).headers.get("content-length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"  {name:18s} already complete ({have/1e6:.0f} MB)")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    print(f"  {name:18s} downloading {total/1e6:.0f} MB"
          + (f" (resuming at {have/1e6:.0f} MB)" if have else ""))

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        # ponytail: server may ignore Range (200 not 206) — append then corrupts
        resumed = have and r.status_code == 206
        with open(dest, "ab" if resumed else "wb") as f:
            done = have if resumed else 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {pct:5.1f}%  {done/1e6:7.0f} MB", end="")
    print()
    return dest


def main() -> None:
    print("Downloading MaleCNS v1.0 (CC-BY, Janelia FlyEM / Cambridge / Google)")
    paths = {k: download(k, v) for k, v in FILES.items()}
    print("\nDone. Raw files:")
    for k, p in paths.items():
        print(f"  {k:18s} {p.stat().st_size / 1e6:7.0f} MB  {p}")
    print("Next: python process.py")


if __name__ == "__main__":
    main()
