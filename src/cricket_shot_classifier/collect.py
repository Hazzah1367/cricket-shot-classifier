"""Download real Wikimedia photos with per-image provenance; never infer usage rights."""

import argparse
import csv
import hashlib
import html
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import CLASSES

API = "https://commons.wikimedia.org/w/api.php"
CATEGORIES = dict(
    zip(
        CLASSES,
        ["Cricket cover drive shots", "Cricket pull shots", "Cricket sweep shots"],
        strict=True,
    )
)


def plain(value):
    return html.unescape(re.sub("<[^>]+>", "", value or "")).strip()


def source_group(title, metadata):
    """Conservatively group a named match and photographer/date; review before splitting."""
    name = re.sub(r"\.(jpg|jpeg|png)$", "", title.removeprefix("File:"), flags=re.I)
    name = re.sub(r"\s*\(\d+\)$", "", name)  # Flickr photo identifiers
    name = re.sub(r"[ _-]*\d+$", "", name)
    artist = plain(metadata.get("Artist", {}).get("value", ""))
    date = plain(metadata.get("DateTimeOriginal", {}).get("value", ""))[:10]
    # Match name takes precedence: repeated photos of the same fixture remain together.
    key = (
        name.casefold()
        if " v" in name.casefold() or "match" in name.casefold()
        else (f"{artist}|{date}" if artist and date else name.casefold())
    )
    return "source_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def collect(output: Path, limit: int):
    if (output / "candidates.csv").exists():
        raise FileExistsError("Candidate manifest exists; choose a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "CricketShotClassifier/0.1 (educational research; attribution retained)"
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        ),
    )
    rows, errors = [], []
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    for label, category in CATEGORIES.items():
        members, continuation = [], {}
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": "Category:" + category,
                "cmlimit": 200,
                "cmnamespace": 6,
                "format": "json",
                **continuation,
            }
            response = session.get(API, params=params, timeout=45)
            response.raise_for_status()
            result = response.json()
            members += result.get("query", {}).get("categorymembers", [])
            if "continue" not in result:
                break
            continuation = result["continue"]
        print(f"{label}: {len(members)} source files", flush=True)
        # Fetch source metadata in API-sized batches; use four bounded download workers.
        members = list({m["pageid"]: m for m in members}.values())
        selected = sorted(members, key=lambda m: m["pageid"])[:limit]
        missing = [m["pageid"] for m in selected if not (cache / f"{m['pageid']}.json").exists()]
        for start in range(0, len(missing), 50):
            r = session.get(
                API,
                params={
                    "action": "query",
                    "pageids": "|".join(map(str, missing[start : start + 50])),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "iiurlwidth": 1024,
                    "format": "json",
                },
                timeout=60,
            )
            r.raise_for_status()
            for key, page in r.json()["query"]["pages"].items():
                (cache / f"{key}.json").write_text(json.dumps(page, ensure_ascii=False, indent=2))

        def download_one(member):
            page_id = member["pageid"]
            path = output / "images" / label / f"commons_{page_id}.jpg"
            try:
                page = json.loads((cache / f"{page_id}.json").read_text())
                info = page["imageinfo"][0]
                metadata = info["extmetadata"]
                licence = plain(metadata.get("LicenseShortName", {}).get("value", ""))
                licence_url = plain(metadata.get("LicenseUrl", {}).get("value", ""))
                if not (
                    re.fullmatch(r"CC BY(?:-SA)? (?:1\.0|2\.0|2\.5|3\.0|4\.0)", licence)
                    or licence in ["CC0", "Public domain"]
                ):
                    raise ValueError(f"Unsupported/unknown licence: {licence}")
                download = info.get("thumburl", info["url"])
                if not path.exists():
                    # A separate session avoids sharing mutable HTTP state between threads.
                    with requests.Session() as worker:
                        worker.headers["User-Agent"] = session.headers["User-Agent"]
                        worker.mount(
                            "https://",
                            HTTPAdapter(
                                max_retries=Retry(
                                    total=3,
                                    backoff_factor=1,
                                    status_forcelist=[429, 500, 502, 503, 504],
                                )
                            ),
                        )
                        r = worker.get(download, timeout=60)
                        r.raise_for_status()
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert(
                        "RGB"
                    )
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path, quality=90)
                with Image.open(path) as image:
                    width, height = image.size
                row = {
                    "path": path.relative_to(output.parent).as_posix(),
                    "label": label,
                    "source_group": source_group(page["title"], metadata),
                    "source_url": info["descriptionurl"],
                    "original_url": info["url"],
                    "download_url": download,
                    "title": page["title"],
                    "artist": plain(metadata.get("Artist", {}).get("value", "")),
                    "license": licence,
                    "license_url": licence_url,
                    "description": plain(metadata.get("ImageDescription", {}).get("value", "")),
                    "date": plain(metadata.get("DateTimeOriginal", {}).get("value", "")),
                    "width": width,
                    "height": height,
                    "review_status": "pending",
                    "review_notes": "Category label is a candidate, not expert ground truth.",
                    "modifications": "EXIF orientation; RGB; resized longest edge <=1024; JPEG quality 90",
                }
                print(f"  saved {label}/{page_id}", flush=True)
                return row, None
            except (requests.RequestException, ValueError, KeyError, OSError) as exc:
                print(f"  skipped {page_id}: {exc}", flush=True)
                return None, {"page_id": page_id, "label": label, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=4) as pool:
            for row, error in pool.map(download_one, selected):
                if row:
                    rows.append(row)
                if error:
                    errors.append(error)
    if not rows:
        raise RuntimeError("No usable images downloaded; refusing to create an empty dataset.")
    manifest = output / "candidates.csv"
    if manifest.exists():
        raise FileExistsError(
            f"{manifest} exists: preserve reviews or choose a fresh output directory."
        )
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "collection_errors.json").write_text(json.dumps(errors, indent=2))
    print(f"Saved {len(rows)} candidates. Review labels and source groups before preparing splits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--limit-per-class", type=int, default=100)
    args = parser.parse_args()
    if args.limit_per_class < 1:
        parser.error("limit must be positive")
    collect(args.output, args.limit_per_class)


if __name__ == "__main__":
    main()
