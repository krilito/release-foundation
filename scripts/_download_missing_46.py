"""
Batch download missing corpus PDFs from Sci-Hub.
Reads DOIs with status=need_download from pdf_corpus_match_manifest.csv.
As of 2026-06-14, 35 of 46 were downloaded; 12 remain (post-2020 Sci-Hub gaps).
"""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.st",
    "https://sci-hub.se",
]

SAVE_DIR = Path(r"D:\学术研究\药物控释\文献\scihub_batch_downloads")
SAVE_DIR.mkdir(exist_ok=True)

# Load DOIs from manifest
manifest = Path(r"D:\release-foundation\data\pdf_corpus_match_manifest.csv")
DOIS: list[str] = []
with open(manifest, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "need_download":
            DOIS.append(row["doi"])

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def safe_filename(doi: str) -> str:
    return doi.replace("/", "-").replace(":", "-") + ".pdf"


def find_pdf_url(html: str, base_url: str) -> str | None:
    patterns = [
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        r'<embed[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'<iframe[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'onclick="location\.href=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'src=["\']//([^"\']+\.pdf[^"\']*)["\']',
        r'location\.href=["\']([^"\']+\.pdf[^"\']*)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            url = m.group(1)
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = base_url.rstrip("/") + url
            elif not url.startswith("http"):
                url = urljoin(base_url, url)
            return url

    m = re.search(r'https?://[^\s"\'<>]+\.pdf(?:\?[^\s"\'<>]*)?', html, re.IGNORECASE)
    if m:
        return m.group(0)
    return None


def download_doi(doi: str, session: requests.Session) -> bool:
    filename = safe_filename(doi)
    save_path = SAVE_DIR / filename

    if save_path.exists() and save_path.stat().st_size > 10_000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    for mirror in SCIHUB_MIRRORS:
        url = f"{mirror}/{doi}"
        try:
            print(f"  Trying {mirror} ...", end=" ", flush=True)
            resp = session.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "pdf" in content_type:
                save_path.write_bytes(resp.content)
                print(f"OK (direct PDF, {len(resp.content)//1024} KB)")
                return True

            pdf_url = find_pdf_url(resp.text, mirror)
            if not pdf_url:
                print("no PDF link found")
                continue

            pdf_resp = session.get(pdf_url, headers=HEADERS, timeout=60, stream=True)
            if pdf_resp.status_code != 200:
                print(f"PDF fetch HTTP {pdf_resp.status_code}")
                continue

            data = b"".join(pdf_resp.iter_content(chunk_size=8192))
            if len(data) < 5000:
                print(f"too small ({len(data)} bytes), skipping")
                continue

            save_path.write_bytes(data)
            print(f"OK ({len(data)//1024} KB)")
            return True

        except Exception as e:
            print(f"error: {e}")
            continue

    return False


def main() -> None:
    print(f"DOIs to download: {len(DOIS)}")
    print(f"Saving to: {SAVE_DIR}\n")
    success, failed = [], []

    with requests.Session() as session:
        for i, doi in enumerate(DOIS, 1):
            print(f"[{i:02d}/{len(DOIS)}] {doi}")
            ok = download_doi(doi, session)
            (success if ok else failed).append(doi)
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Downloaded: {len(success)}/{len(DOIS)}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for d in failed:
            print(f"  {d}")
    else:
        print("All downloads succeeded!")


if __name__ == "__main__":
    main()
