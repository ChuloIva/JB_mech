#!/usr/bin/env python3
"""
Scraper for Promptfoo LM Security Database
https://www.promptfoo.dev/lm-security-db

Extracts vulnerability data from Next.js streaming format and saves as JSON/CSV.
"""

import requests
import re
import json
import csv
from pathlib import Path
from datetime import datetime


def fetch_page(url: str) -> str:
    """Fetch the page HTML content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def extract_nextjs_data(html: str) -> list[dict]:
    """Extract vulnerability data from Next.js streaming chunks."""
    vulnerabilities = []

    # Pattern to match self.__next_f.push data chunks
    pattern = r'self\.__next_f\.push\(\[[\d,]+,"(.+?)"\]\)'
    matches = re.findall(pattern, html, re.DOTALL)

    # Also try script tags with __NEXT_DATA__
    next_data_pattern = r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>'
    next_data_match = re.search(next_data_pattern, html, re.DOTALL)

    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            # Navigate through the Next.js data structure
            if "props" in data and "pageProps" in data["props"]:
                page_props = data["props"]["pageProps"]
                if "vulnerabilities" in page_props:
                    return page_props["vulnerabilities"]
        except json.JSONDecodeError:
            pass

    # Parse streaming chunks
    all_text = ""
    for match in matches:
        # Unescape the string
        try:
            unescaped = match.encode().decode('unicode_escape')
            all_text += unescaped
        except:
            all_text += match

    # Try to find JSON arrays or objects in the combined text
    # Look for vulnerability-like objects
    vuln_pattern = r'\{[^{}]*"title"\s*:\s*"[^"]+?"[^{}]*"cveId"\s*:\s*"[^"]*?"[^{}]*\}'
    vuln_matches = re.findall(vuln_pattern, all_text, re.DOTALL)

    for match in vuln_matches:
        try:
            vuln = json.loads(match)
            vulnerabilities.append(vuln)
        except json.JSONDecodeError:
            continue

    # Alternative: look for structured data patterns
    # title, paperTitle, paperUrl, tags, affectedModels, description
    structured_pattern = r'"title":"([^"]+)".*?"paperTitle":"([^"]*)".*?"paperUrl":"([^"]*)".*?"tags":\[([^\]]*)\].*?"affectedModels":\[([^\]]*)\].*?"description":"([^"]*)"'

    struct_matches = re.findall(structured_pattern, all_text, re.DOTALL)
    for m in struct_matches:
        vuln = {
            "title": m[0],
            "paperTitle": m[1],
            "paperUrl": m[2],
            "tags": [t.strip().strip('"') for t in m[3].split(",") if t.strip()],
            "affectedModels": [t.strip().strip('"') for t in m[4].split(",") if t.strip()],
            "description": m[5],
        }
        # Avoid duplicates
        if vuln not in vulnerabilities:
            vulnerabilities.append(vuln)

    return vulnerabilities


def extract_from_rsc_payload(html: str) -> list[dict]:
    """Extract data from React Server Components payload format."""
    vulnerabilities = []

    # RSC payloads are often in script tags or inline
    # Pattern for RSC-style data
    patterns = [
        # Pattern 1: Look for array of vulnerability objects
        r'\["(\$[^"]+)",\s*\{[^}]*"title":\s*"([^"]+)"[^}]*\}',
        # Pattern 2: JSON-like structures
        r'(\{[^{}]*?"title"\s*:\s*"[^"]+?"[^{}]*?"description"\s*:\s*"[^"]*?"[^{}]*\})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, re.DOTALL)
        for match in matches:
            if isinstance(match, tuple):
                match = match[-1] if match[-1].startswith("{") else match[0]
            try:
                if match.startswith("{"):
                    vuln = json.loads(match)
                    if "title" in vuln:
                        vulnerabilities.append(vuln)
            except:
                continue

    return vulnerabilities


def parse_inline_data(html: str) -> list[dict]:
    """Parse vulnerability data from various inline formats."""
    vulnerabilities = []

    # Extract all script content
    script_pattern = r'<script[^>]*>(.*?)</script>'
    scripts = re.findall(script_pattern, html, re.DOTALL)

    combined = " ".join(scripts)

    # Look for patterns that indicate vulnerability entries
    # CVE IDs
    cve_pattern = r'(CVE-\d{4}-\d+|LLM-\d+|VULN-\d+)'
    cve_ids = re.findall(cve_pattern, combined)

    # arXiv links (common source for papers)
    arxiv_pattern = r'https?://arxiv\.org/[^\s"<>]+'
    arxiv_links = re.findall(arxiv_pattern, combined)

    # Titles (heuristic)
    title_pattern = r'"title"\s*:\s*"([^"]+)"'
    titles = re.findall(title_pattern, combined)

    # Build partial records if we can't get full objects
    if titles and not vulnerabilities:
        for i, title in enumerate(titles):
            vuln = {"title": title}
            if i < len(cve_ids):
                vuln["cveId"] = cve_ids[i]
            if i < len(arxiv_links):
                vuln["paperUrl"] = arxiv_links[i]
            vulnerabilities.append(vuln)

    return vulnerabilities


def scrape_lm_security_db(url: str = "https://www.promptfoo.dev/lm-security-db") -> list[dict]:
    """Main scraping function."""
    print(f"Fetching {url}...")
    html = fetch_page(url)

    print("Extracting vulnerability data...")

    # Try multiple extraction methods
    vulnerabilities = extract_nextjs_data(html)

    if not vulnerabilities:
        print("Trying RSC payload extraction...")
        vulnerabilities = extract_from_rsc_payload(html)

    if not vulnerabilities:
        print("Trying inline data parsing...")
        vulnerabilities = parse_inline_data(html)

    # Deduplicate based on title
    seen_titles = set()
    unique_vulns = []
    for v in vulnerabilities:
        title = v.get("title", "")
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_vulns.append(v)

    print(f"Found {len(unique_vulns)} unique vulnerabilities")
    return unique_vulns


def save_json(data: list[dict], output_path: str):
    """Save data as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {output_path}")


def save_csv(data: list[dict], output_path: str):
    """Save data as CSV."""
    if not data:
        print("No data to save")
        return

    # Collect all possible fields
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())

    fieldnames = sorted(all_fields)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            # Convert lists to strings for CSV
            row = {}
            for k, v in item.items():
                if isinstance(v, list):
                    row[k] = "; ".join(str(x) for x in v)
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"Saved CSV to {output_path}")


def main():
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Scrape the data
    vulnerabilities = scrape_lm_security_db()

    if vulnerabilities:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = output_dir / f"lm_security_db_{timestamp}.json"
        save_json(vulnerabilities, str(json_path))

        # Save as CSV
        csv_path = output_dir / f"lm_security_db_{timestamp}.csv"
        save_csv(vulnerabilities, str(csv_path))

        # Also save a latest version
        save_json(vulnerabilities, str(output_dir / "lm_security_db_latest.json"))
        save_csv(vulnerabilities, str(output_dir / "lm_security_db_latest.csv"))

        # Print sample
        print("\n--- Sample Entry ---")
        print(json.dumps(vulnerabilities[0], indent=2))
    else:
        print("No vulnerabilities extracted. The page structure may have changed.")
        print("Saving raw HTML for manual inspection...")
        html = fetch_page("https://www.promptfoo.dev/lm-security-db")
        with open(output_dir / "raw_page.html", "w") as f:
            f.write(html)


if __name__ == "__main__":
    main()
