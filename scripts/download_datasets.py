#!/usr/bin/env python3
"""
Download datasets for QFormer-RevMark.

Datasets:
- Kodak (24 images, ~50 MB)        : direct PNG download
- USC-SIPI Misc (color subset)      : individual TIFFs
- CLIC-2020 professional validation : ZIP archive
- COCO val2017 (trimmed to 1K)      : COCO official
- ISIC dermoscopy (200 subset)      : ISIC archive API

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --kodak --sipi
"""
import argparse
import os
import sys
import time
import urllib.request
import zipfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[" + ts + "] " + str(msg), flush=True)


def download_file(url, dest, retries=2):
    """Download with progress bar; retry on transient failures."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log("  SKIP (exists): " + dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            line = "  " + dest.name + ": " + ("%5.1f" % pct) + "% (" + ("%6.1f" % mb) + "/" + ("%6.1f" % total_mb) + " MB)"
            print("\r" + line, end="", flush=True)

    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(url, dest, reporthook=progress)
            print()
            return dest
        except Exception as e:
            print()
            log("  ERROR (attempt " + str(attempt + 1) + ") downloading " + url + ": " + str(e))
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                time.sleep(2)
    return None


def download_kodak():
    log("=== Kodak (24 images) ===")
    out_dir = DATA / "kodak"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "http://r0k.us/graphics/kodak/kodak/"
    n_ok = 0
    for i in range(1, 25):
        fname = "kodim" + ("%02d" % i) + ".png"
        url = base + fname
        if download_file(url, out_dir / fname):
            n_ok += 1
    log("  Done. " + str(n_ok) + "/24 files in " + str(out_dir))


def download_sipi():
    log("=== USC-SIPI Misc (selected color) ===")
    out_dir = DATA / "sipi"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "https://sipi.usc.edu/database/misc/"
    files = [
        "4.1.01.tiff", "4.1.02.tiff", "4.1.03.tiff", "4.1.04.tiff",
        "4.1.05.tiff", "4.1.06.tiff", "4.1.07.tiff", "4.1.08.tiff",
        "4.2.01.tiff", "4.2.03.tiff", "4.2.05.tiff", "4.2.06.tiff",
        "4.2.07.tiff", "house.tiff",
    ]
    n_ok = 0
    for f in files:
        if download_file(base + f, out_dir / f):
            n_ok += 1
    log("  Done. " + str(n_ok) + "/" + str(len(files)) + " files in " + str(out_dir))


def download_clic():
    log("=== CLIC-2020 professional validation ===")
    out_dir = DATA / "clic"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip"
    zip_path = out_dir / "clic_pro_valid.zip"
    if not zip_path.exists():
        download_file(url, zip_path)
    if zip_path.exists() and zip_path.stat().st_size > 0:
        log("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        n_png = len(list(out_dir.rglob("*.png")))
        log("  Extracted " + str(n_png) + " PNG files.")
        zip_path.unlink()
        log("  Zip removed to save disk.")
    else:
        log("  WARN: CLIC zip not downloaded.")


def download_coco():
    log("=== COCO val2017 (trim to 1K) ===")
    out_dir = DATA / "coco"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = out_dir / "val2017.zip"
    if not zip_path.exists():
        download_file(url, zip_path)
    if zip_path.exists() and zip_path.stat().st_size > 0:
        extract_dir = out_dir / "val2017"
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            log("  Extracting val2017...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(out_dir)
        all_files = sorted(extract_dir.glob("*.jpg"))
        log("  Total extracted: " + str(len(all_files)))
        if len(all_files) > 1000:
            log("  Trimming to first 1000 files to save disk...")
            for f in all_files[1000:]:
                f.unlink()
        zip_path.unlink()
        log("  Zip removed. Final count: " + str(len(list(extract_dir.glob("*.jpg")))))
    else:
        log("  WARN: COCO zip not downloaded.")


def download_isic():
    log("=== ISIC dermoscopy (200) ===")
    out_dir = DATA / "isic"
    out_dir.mkdir(parents=True, exist_ok=True)
    api_url = "https://api.isic-archive.com/api/v2/images/?limit=200"
    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        log("  ERROR querying ISIC API: " + str(e))
        log("  HINT: download manually later from https://api.isic-archive.com")
        return

    results = data.get("results", [])
    log("  API returned " + str(len(results)) + " entries")
    n_ok = 0
    for entry in results:
        files_obj = entry.get("files") or {}
        full_obj = files_obj.get("full") or {}
        url = full_obj.get("url")
        isic_id = entry.get("isic_id")
        if not url or not isic_id:
            continue
        fname = isic_id + ".jpg"
        if download_file(url, out_dir / fname):
            n_ok += 1
    log("  Done. " + str(n_ok) + " images in " + str(out_dir))


def summarize():
    log("\n=== Summary ===")
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    for ds in ["kodak", "sipi", "clic", "coco", "isic"]:
        d = DATA / ds
        if not d.exists():
            log("  " + ("%-8s" % ds) + ": MISSING")
            continue
        n_imgs = sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 / 1024
        log("  " + ("%-8s" % ds) + ": " + ("%5d" % n_imgs) + " images, " + ("%7.1f" % size_mb) + " MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kodak", action="store_true")
    parser.add_argument("--sipi", action="store_true")
    parser.add_argument("--clic", action="store_true")
    parser.add_argument("--coco", action="store_true")
    parser.add_argument("--isic", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.kodak = args.sipi = args.clic = args.coco = args.isic = True

    if not any([args.kodak, args.sipi, args.clic, args.coco, args.isic]):
        parser.print_help()
        return

    if args.kodak: download_kodak()
    if args.sipi:  download_sipi()
    if args.clic:  download_clic()
    if args.coco:  download_coco()
    if args.isic:  download_isic()

    summarize()


if __name__ == "__main__":
    main()
