import os
import json
import time
import random
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any, Set, Tuple

import requests
from PIL import Image
import torchvision.transforms as T

"""
image_collection.py  ▸  machine-hallucinations, v2.1
====================================================
Unified crawler for
  • NASA Image & Video Library (multi-query: base, "hubble", "james webb")
  • Optional Hubble Legacy Archive (HST)    (--hst flag)
  • Optional JWST mosaics via MAST          (--jwst flag)

Features
--------
1️⃣  Pad→resize (1024²) and N random crops (default 5).  
2️⃣  Perceptual-hash deduplication (imagehash, Hamming<5).  
3️⃣  FITS→PNG conversion for HST / JWST (astropy + numpy).  
4️⃣  Resumable: skips any already-downloaded NASA ID or hash.  
5️⃣  Metadata recorded to `metadata.jsonl` (one JSON per line).  

Dependencies (extras: hst & jwst)
--------------------------------
core   : pillow  torchvision  requests  imagehash
astro : numpy  astropy  astroquery           (only if --hst/--jwst)

Examples
--------
NASA only:
    python image_collection.py --query "galaxy" --pages 30
Full ingest:
    pip install imagehash numpy astropy astroquery
    python image_collection.py --query "galaxy" --pages 10 --crops 5 --hst --jwst
"""

# ---------------------------------------------------------------------------
# Lazy heavy-deps
# ---------------------------------------------------------------------------
try:
    import imagehash  # perceptual hash
except ImportError:
    imagehash = None

try:
    import numpy as np
    from astropy.io import fits  # FITS
except ImportError:
    np = None
    fits = None

try:
    from astroquery.mast import Observations  # HST/JWST search
except ImportError:
    Observations = None

# ---------------------------------------------------------------------------
# CLI & env
# ---------------------------------------------------------------------------
import argparse

ENV_API_KEY = os.getenv("NASA_API_KEY")
DEFAULT_QUERY = os.getenv("NASA_QUERY", "galaxy")
DEFAULT_PAGES = int(os.getenv("NASA_PAGES", "50"))
DEFAULT_DIR = Path(os.getenv("NASA_SAVE_DIR", "nasa_galaxy_images")).expanduser()
DEFAULT_CROPS = int(os.getenv("NASA_CROPS", "5"))

parser = argparse.ArgumentParser(description="Collect NASA/HST/JWST images for ML datasets")
parser.add_argument("--query", default=DEFAULT_QUERY, help="Base search term")
parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="Pages per variant / collection")
parser.add_argument("--outdir", default=str(DEFAULT_DIR))
parser.add_argument("--api_key", default=ENV_API_KEY)
parser.add_argument("--delay", type=float, default=0.1)
parser.add_argument("--crops", type=int, default=DEFAULT_CROPS, help="Random crops per image (0 to disable)")
parser.add_argument("--hst", action="store_true", help="Harvest Hubble Legacy Archive (requires astroquery & astropy)")
parser.add_argument("--jwst", action="store_true", help="Harvest JWST mosaics (requires astroquery & astropy)")
args = parser.parse_args()

BASE_QUERY: str = args.query
PAGE_LIMIT: int = args.pages
SAVE_DIR: Path = Path(args.outdir)
API_KEY: str | None = args.api_key
DELAY: float = args.delay
CROP_COUNT: int = max(0, args.crops)
TARGET_SIZE: int = 1024

SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESIZED_DIR = SAVE_DIR / "resized"
RESIZED_DIR.mkdir(exist_ok=True)
META_FILE = SAVE_DIR / "metadata.jsonl"
HASH_FILE = SAVE_DIR / "hashes.json"

# ---------------------------------------------------------------------------
# Requests session with retries
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update({"User-Agent": "machine-hallucinations/2.1"})
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

session = make_session()

# ---------------------------------------------------------------------------
# NASA Image & Video Library helpers
# ---------------------------------------------------------------------------

def nasa_search(query: str, page: int) -> List[Dict[str, Any]]:
    base = "https://images-api.nasa.gov/search"
    params = {"q": query, "media_type": "image", "page": page}
    if API_KEY:
        params["api_key"] = API_KEY
    r = session.get(base, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("collection", {}).get("items", [])


def nasa_asset_urls(nasa_id: str) -> List[str]:
    base = f"https://images-api.nasa.gov/asset/{nasa_id}"
    if API_KEY:
        base += f"?api_key={API_KEY}"
    r = session.get(base, timeout=30)
    r.raise_for_status()
    items = r.json().get("collection", {}).get("items", [])
    return [i.get("href", "") for i in items if i.get("href", "").lower().endswith(".jpg")]

# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
random_crop = T.RandomResizedCrop(size=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33))

def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    m = max(w, h)
    out = Image.new("RGB", (m, m), (0, 0, 0))
    out.paste(img, ((m - w) // 2, (m - h) // 2))
    return out

# ---------------------------------------------------------------------------
# Dedup helpers using perceptual hash
# ---------------------------------------------------------------------------
_hash_set: Set[str] = set(json.loads(HASH_FILE.read_text()) if HASH_FILE.exists() else [])

def compute_phash(img: Image.Image) -> str | None:
    return str(imagehash.phash(img)) if imagehash else None


def is_duplicate(ph: str, thresh: int = 5) -> bool:
    if imagehash is None:
        return False
    for existing in _hash_set:
        if existing == ph:
            return True
        if imagehash.hex_to_hash(existing) - imagehash.hex_to_hash(ph) < thresh:
            return True
    return False


def register_phash(ph: str):
    _hash_set.add(ph)
    HASH_FILE.write_text(json.dumps(list(_hash_set)))

# ---------------------------------------------------------------------------
# Resize + crops + dedup
# ---------------------------------------------------------------------------

def process_image(src_path: Path, uid: str) -> Tuple[Dict[str, Any] | None, bool]:
    """Return (resize_info, duplicate_flag)."""
    canonical_png = RESIZED_DIR / f"{uid}_1024.png"
    if canonical_png.exists():
        crops = sorted(RESIZED_DIR.glob(f"{uid}_crop*.png"))
        return {"canonical": canonical_png.name, "crops": [p.name for p in crops]}, False

    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im_p = pad_to_square(im)
        can = im_p.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
        ph = compute_phash(can)
        if ph and is_duplicate(ph):
            return None, True
        can.save(canonical_png, "PNG")
        if ph:
            register_phash(ph)
        crop_names: List[str] = []
        for i in range(CROP_COUNT):
            crop = random_crop(im_p)
            cpath = RESIZED_DIR / f"{uid}_crop{i}.png"
            crop.save(cpath, "PNG")
            crop_names.append(cpath.name)
    return {"canonical": canonical_png.name, "crops": crop_names}, False

# ---------------------------------------------------------------------------
# Utility I/O
# ---------------------------------------------------------------------------

def write_meta(record: Dict[str, Any]):
    with META_FILE.open("a") as fp:
        json.dump(record, fp)
        fp.write("\n")


def already_downloaded(uid: str, path: Path) -> bool:
    return path.exists() or any(uid in line for line in META_FILE.read_text().splitlines()) if META_FILE.exists() else path.exists()

# ---------------------------------------------------------------------------
# Core download primitive
# ---------------------------------------------------------------------------

def download(url: str, dst: Path):
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(1 << 15):
                f.write(chunk)

# ---------------------------------------------------------------------------
# NASA harvesting
# ---------------------------------------------------------------------------

def harvest_nasa_variant(query: str):
    page = 1
    while page <= PAGE_LIMIT:
        print(f"\n[NASA] {query} – page {page}/{PAGE_LIMIT}")
        results = nasa_search(query, page)
        if not results:
            break
        for item in results:
            meta = item.get("data", [{}])[0]
            nasa_id = meta.get("nasa_id")
            if not nasa_id:
                continue
            urls = nasa_asset_urls(nasa_id)
            if not urls:
                continue
            url = max(urls, key=lambda u: len(urlparse(u).path))
            ext = os.path.splitext(urlparse(url).path)[1]
            raw_path = SAVE_DIR / f"{nasa_id}{ext}"
            if not already_downloaded(nasa_id, raw_path):
                try:
                    download(url, raw_path)
                except Exception as e:
                    print("! download fail", nasa_id, e)
                    continue
            resize_info, dup = process_image(raw_path, nasa_id)
            if dup:
                print("• duplicate", nasa_id)
                continue
            rec = {
                "uid": nasa_id,
                "source": "NASA",
                "search_query": query,
                "title": meta.get("title"),
                "date_created": meta.get("date_created"),
                "keywords": meta.get("keywords", []),
                "description": meta.get("description"),
                "original_file": raw_path.name,
                "resized": resize_info,
                "url": url,
            }
            write_meta(rec)
            print("+", nasa_id)
            time.sleep(DELAY)
        page += 1

# ---------------------------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------------------------

def fits_to_png(src: Path, dest: Path):
    if fits is None or np is None:
        raise RuntimeError("astropy & numpy required for FITS conversion")
    hdul = fits.open(src)
    if "SCI" in hdul:
        data = hdul["SCI"].data.astype(np.float32)
    else:
        data = hdul[0].data.astype(np.float32)
    data -= data.min()
    data /= (data.max() or 1.0)
    img = (data * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).convert("RGB").save(dest, "PNG")

# ---------------------------------------------------------------------------
# MAST harvesting (HST/JWST)
# ---------------------------------------------------------------------------

def mast_search(collection: str, filters: Dict[str, Any]) -> List[Any]:
    if Observations is None:
        print("astroquery unavailable – skipping", collection)
        return []
    obs = Observations.query_criteria(obs_collection=collection, **filters)
    return obs


def mast_download_products(obs_rows, subgroup: str, collection: str, limit: int):
    if Observations is None:
        return
    products = Observations.get_product_list(obs_rows)
    products = Observations.filter_products(products, productSubGroupDescription=subgroup, extension="fits")
    for row in products[:limit]:
        url = row["dataURL"]
        fits_path = Path(Observations.download_file(url, cache=True))
        uid = fits_path.stem.replace(".", "_")
        png_path = SAVE_DIR / f"{uid}.png"
        try:
            if not png_path.exists():
                fits_to_png(fits_path, png_path)
        except Exception as e:
            print("! fits convert fail", uid, e)
            continue
        resize_info, dup = process_image(png_path, uid)
        if dup:
            print("• duplicate", uid)
            continue
        rec = {
            "uid": uid,
            "source": collection,
            "search_query": BASE_QUERY,
            "instrument": row.get("instrument_name"),
            "filter": row.get("filters"),
            "original_file": png_path.name,
            "resized": resize_info,
            "mast_url": url,
        }
        write_meta(rec)
        print("+", uid)
        time.sleep(DELAY)


def harvest_hst(limit_per_page: int):
    if not args.hst:
        return
    rows = mast_search("HST", {"dataproduct_type": "image", "target_name": BASE_QUERY})
    mast_download_products(rows, "DRZ", "HST", limit_per_page)


def harvest_jwst(limit_per_page: int):
    if not args.jwst:
        return
    rows = mast_search("JWST", {"dataproduct_type": "image", "target_name": BASE_QUERY})
    mast_download_products(rows, "I2D", "JWST", limit_per_page)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # 1. NASA Image & Video Library (three variants)
    for variant in [BASE_QUERY, f"{BASE_QUERY} hubble", f"{BASE_QUERY} \"james webb\""]:
        harvest_nasa_variant(variant)

    # 2. HST & JWST via MAST (optional, heavy)
    harvest_hst(limit_per_page=200 * PAGE_LIMIT)
    harvest_jwst(limit_per_page=200 * PAGE_LIMIT)

    print("\n✅ Done. Metadata lines:", sum(1 for _ in META_FILE.open()))


if __name__ == "__main__":
    main()
