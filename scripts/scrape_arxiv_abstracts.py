"""
Scrape arxiv abstracts for STAR seed strategies.

Uses arxiv API to fetch abstracts (minimal data footprint).
Output will be used in Colab with Gemma 3 12B to convert to STAR format.
"""

import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests


ARXIV_API_URL = "http://export.arxiv.org/api/query"


def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arxiv ID from URL like https://arxiv.org/abs/2502.01241"""
    match = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url)
    if match:
        return match.group(1)
    return None


def fetch_abstract(arxiv_id: str, max_retries: int = 3) -> Optional[dict]:
    """
    Fetch abstract from arxiv API.

    Returns dict with title, abstract, authors, or None on failure.
    """
    url = f"{ARXIV_API_URL}?id_list={arxiv_id}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            xml_data = response.text

            # Parse XML
            root = ET.fromstring(xml_data)

            # Namespace handling for arxiv API
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)

            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            return {
                "arxiv_id": arxiv_id,
                "title": title.text.strip().replace("\n", " ") if title is not None else None,
                "abstract": summary.text.strip().replace("\n", " ") if summary is not None else None,
                "authors": authors,
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:  # Rate limited
                wait_time = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP error {e.response.status_code} for {arxiv_id}")
                return None
        except Exception as e:
            print(f"  Error fetching {arxiv_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def scrape_abstracts(
    strategies_path: Path,
    output_path: Path,
    delay: float = 0.5,
):
    """
    Scrape arxiv abstracts for all strategies.

    Args:
        strategies_path: Path to star_seed_strategies.json
        output_path: Where to save scraped abstracts
        delay: Delay between requests (arxiv rate limit is 1 req/3s)
    """
    # Load strategies
    with open(strategies_path) as f:
        strategies = json.load(f)

    print(f"Loaded {len(strategies)} strategies")

    # Check for existing progress
    results = {}
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} already scraped")

    # Scrape
    for i, strategy in enumerate(strategies):
        strategy_id = strategy["id"]

        # Skip if already done
        if strategy_id in results:
            continue

        paper_url = strategy.get("paper_url", "")
        arxiv_id = extract_arxiv_id(paper_url)

        if not arxiv_id:
            print(f"[{i+1}/{len(strategies)}] {strategy_id}: No valid arxiv URL")
            results[strategy_id] = {
                "strategy_id": strategy_id,
                "paper_url": paper_url,
                "error": "no_arxiv_id",
            }
            continue

        print(f"[{i+1}/{len(strategies)}] Fetching {arxiv_id}...", end=" ")

        abstract_data = fetch_abstract(arxiv_id)

        if abstract_data:
            results[strategy_id] = {
                "strategy_id": strategy_id,
                "paper_url": paper_url,
                **abstract_data,
            }
            print(f"OK ({len(abstract_data['abstract'] or '')} chars)")
        else:
            results[strategy_id] = {
                "strategy_id": strategy_id,
                "paper_url": paper_url,
                "arxiv_id": arxiv_id,
                "error": "fetch_failed",
            }
            print("FAILED")

        # Save progress
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Rate limit
        time.sleep(delay)

    # Summary
    successful = sum(1 for r in results.values() if "abstract" in r and r["abstract"])
    failed = len(results) - successful

    print(f"\nDone! Scraped {successful} abstracts, {failed} failed")
    print(f"Saved to {output_path}")

    return results


def main():
    strategies_path = Path("outputs/star_strategies/star_seed_strategies.json")
    output_path = Path("outputs/star_strategies/arxiv_abstracts.json")

    scrape_abstracts(strategies_path, output_path, delay=0.5)


if __name__ == "__main__":
    main()
