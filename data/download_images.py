#!/usr/bin/env python3
"""Download just the COCO images the evaluation splits refer to.

The two splits together touch 5,664 images (~700 MB) spread across three COCO
releases whose full archives are ~32 GB, so this fetches them individually
instead. Files land under ``--image-root`` in the layout the JSONL expects:

    $COCO_ROOT/2014/train2014/COCO_train2014_000000045840.jpg
    $COCO_ROOT/train2017/000000335323.jpg
    $COCO_ROOT/val2017/000000012345.jpg

Already-present files are skipped, so the script is safe to re-run after an
interruption. If you already have COCO locally, point ``--image-root`` at it
and this will only fetch what is missing.

Usage:
    python download_images.py --image-root /path/to/COCO
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://images.cocodataset.org"
SPLITS = ("rsc_id_test.jsonl", "rsc_ood_test.jsonl")

HERE = os.path.dirname(os.path.abspath(__file__))


def image_url(rel_path: str) -> str:
    """``2014/train2014/X.jpg`` and ``train2017/X.jpg`` both serve from <split>/X.jpg."""
    parts = rel_path.split("/")
    return f"{BASE_URL}/{parts[-2]}/{parts[-1]}"


def collect_paths(split_files) -> list:
    paths = set()
    for name in split_files:
        path = name if os.path.isabs(name) else os.path.join(HERE, name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.add(json.loads(line)["image_path"])
    return sorted(paths)


def fetch(rel_path: str, image_root: str, retries: int = 3) -> str:
    dest = os.path.join(image_root, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(image_url(rel_path), timeout=60) as r, open(tmp, "wb") as out:
                out.write(r.read())
            os.replace(tmp, dest)
            return "ok"
        except (urllib.error.URLError, OSError) as e:
            last = e
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"{rel_path}: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-root", default=os.environ.get("COCO_ROOT"),
                    help="destination COCO root (env: COCO_ROOT)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS),
                    help="evaluation JSONL files to read image paths from")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not args.image_root:
        ap.error("--image-root is required (or set COCO_ROOT)")

    paths = collect_paths(args.splits)
    print(f"{len(paths)} unique images -> {args.image_root}")

    done = skipped = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, p, args.image_root): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                if fut.result() == "skip":
                    skipped += 1
                else:
                    done += 1
            except Exception as e:
                failures.append(str(e))
            if i % 100 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}  downloaded={done} skipped={skipped} failed={len(failures)}",
                      flush=True)

    print(f"\n✓ {done} downloaded, {skipped} already present, {len(failures)} failed")
    if failures:
        for f in failures[:10]:
            print(f"  {f}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        print("Re-run to retry the failures.")
        sys.exit(1)


if __name__ == "__main__":
    main()
