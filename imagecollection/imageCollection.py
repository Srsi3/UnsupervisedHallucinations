import os
import json
import time
from pathlib import Path
from urllib.parse import urlparse
import random

import requests
from PIL import Image
import torchvision.transforms as T

"""
Image collection pipeline for NASA Galaxy-style datasets.

search three variants in sequence:

  1. **Q** (baseline)
  2. **Q + "hubble"**
  3. **Q + "james webb"**

This pulls extra Hubble and JWST imagery without any extra CLI flags.  The
pipeline is still idempotent because duplicate NASA IDs are skipped.
"""

import argparse
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# CLI & env ------------------------------------------------------------------
# ---------------------------------------------------------------------------
ENV_API_KEY = os.getenv("NASA_API_KEY")
DEFAULT_QUERY = os.getenv("NASA_QUERY", "galaxy")
DEFAULT_PAGES = int(os.getenv("NASA_PAGES", "50"))
DEFAULT_DIR = Path(os.getenv("NASA_SAVE_DIR", "nasa_galaxy_images")).expanduser()
DEFAULT_CROPS = int(os.getenv("NASA_CROPS", "5"))

parser = argparse.ArgumentParser(description="Download NASA image assets + metadata")
parser.add_argument("--query", default=DEFAULT_QUERY, help="Base search term (default: galaxy)")
parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="Pages per query variant (default: 50)")
parser.add_argument("--outdir", default=str(DEFAULT_DIR))
parser.add_argument("--api_key", default=ENV_API_KEY)
parser.add_argument("--delay", type=float, default=0.1)
parser.add_argument("--crops", type=int, default=DEFAULT_CROPS)
args = parser.parse_args()

BASE_QUERY = args.query
PAGE_LIMIT = args.pages
SAVE_DIR = Path(args.outdir)
API_KEY = args.api_key
DELAY = args.delay
CROP_COUNT = max(0, args.crops)
TARGET_SIZE = 1024

SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESIZED_DIR = SAVE_DIR / "resized"
RESIZED_DIR.mkdir(exist_ok=True)
META_FILE = SAVE_DIR / "metadata.jsonl"

# ---------------------------------------------------------------------------
# Networking helpers ---------------------------------------------------------
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    from requests.adapters import HTTPAdapter, Retry

    s = requests.Session()
    s.headers.update({"User-Agent": "machine-hallucinations/1.2 (+https://example.com)"})
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

session = make_session()


def search(query: str, page: int) -> List[Dict[str, Any]]:
    base = "https://images-api.nasa.gov/search"
    params = {"q": query, "media_type": "image", "page": page}
    if API_KEY:
        params["api_key"] = API_KEY
    r = session.get(base, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("collection", {}).get("items", [])


def get_asset_urls(nasa_id: str) -> List[str]:
    base = f"https://images-api.nasa.gov/asset/{nasa_id}"
    if API_KEY:
        base += f"?api_key={API_KEY}"
    r = session.get(base, timeout=30)
    r.raise_for_status()
    items = r.json().get("collection", {}).get("items", [])
    return [i.get("href", "") for i in items if i.get("href", "").lower().endswith(".jpg")]

# ---------------------------------------------------------------------------
# Image processing helpers ----------------------------------------------------
# ---------------------------------------------------------------------------
random_crop = T.RandomResizedCrop(
    size=TARGET_SIZE,
    scale=(0.8, 1.0),
    ratio=(0.75, 1.33),
)

def pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_im = Image.new("RGB", (size, size), (0, 0, 0))
    new_im.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_im


def generate_resized_and_crops(src_path: Path, nasa_id: str) -> Dict[str, Any]:
    canonical_name = f"{nasa_id}_1024.png"
    canonical_path = RESIZED_DIR / canonical_name
    crop_names: List[str] = []

    if canonical_path.exists():
        existing_crops = sorted(RESIZED_DIR.glob(f"{nasa_id}_crop*.png"))
        if existing_crops:
            crop_names = [p.name for p in existing_crops]
        return {"canonical": canonical_name, "crops": crop_names}

    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img_pad = pad_to_square(img)
        canonical = img_pad.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
        canonical.save(canonical_path, "PNG")

        for i in range(CROP_COUNT):
            crop = random_crop(img_pad)
            crop_path = RESIZED_DIR / f"{nasa_id}_crop{i}.png"
            crop.save(crop_path, "PNG")
            crop_names.append(crop_path.name)

    return {"canonical": canonical_name, "crops": crop_names}

# ---------------------------------------------------------------------------
# Metadata helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def write_metadata(record: Dict[str, Any]):
    with META_FILE.open("a") as f:
        json.dump(record, f)
        f.write("\n")


def already_downloaded(nasa_id: str, original: Path) -> bool:
    if original.exists():
        return True
    if META_FILE.exists():
        with META_FILE.open() as f:
            for line in f:
                try:
                    if json.loads(line).get("nasa_id") == nasa_id:
                        return True
                except json.JSONDecodeError:
                    continue
    return False

# ---------------------------------------------------------------------------
# Download helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def download_image(url: str, dest: Path):
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fp:
            for chunk in resp.iter_content(chunk_size=1 << 15):
                fp.write(chunk)


def select_largest(urls: List[str]) -> str:
    return max(urls, key=lambda u: len(urlparse(u).path))

# ---------------------------------------------------------------------------
# Main loop ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def harvest(query: str):
    """Process one query variant."""
    page = 1
    while page <= PAGE_LIMIT:
        print(f"\n>>> [{query}] page {page}/{PAGE_LIMIT}…")
        results = search(query, page)
        if not results:
            print("No more results for this variant; moving on.")
            break
        for item in results:
            meta = item.get("data", [{}])[0]
            nasa_id = meta.get("nasa_id")
            if not nasa_id:
                continue

            asset_urls = get_asset_urls(nasa_id)
            if not asset_urls:
                continue

            img_url = select_largest(asset_urls)
            ext = os.path.splitext(urlparse(img_url).path)[1]
            original_path = SAVE_DIR / f"{nasa_id}{ext}"

            if not already_downloaded(nasa_id, original_path):
                try:
                    download_image(img_url, original_path)
                except Exception as e:
                    print(f"! Failed {nasa_id}: {e}")
                    continue
            else:
                print(f"✓ {nasa_id} (cached)")

            resize_info = generate_resized_and_crops(original_path, nasa_id)
            record = {
                "nasa_id": nasa_id,
                "search_query": query,
                "title": meta.get("title"),
                "date_created": meta.get("date_created"),
                "keywords": meta.get("keywords", []),
                "description": meta.get("description"),
                "original_file": original_path.name,
                "resized": resize_info,
                "url": img_url,
            }
            write_metadata(record)
            print(f"+ {nasa_id} → {resize_info['canonical']} (+{len(resize_info['crops'])} crops)")
            time.sleep(DELAY)

        page += 1


def main():
    query_variants = [
        BASE_QUERY,
        f"{BASE_QUERY} hubble",
        f"{BASE_QUERY} \"james webb\"",
    ]
    for q in query_variants:
        harvest(q)
    print("\nDone! All variants processed.")


if __name__ == "__main__":
    main()
