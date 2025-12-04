#!/usr/bin/env python3
"""
Coda Clout — Wikidata Metadata Enrichment

Enriches existing wikipedia_page rows with:
- sitelinks (number of language Wikipedias)
- birth_year (for age calculation)
- award_count
- occupations (stored in separate table)

Run this after populate_clout.py to add pre-death metadata features.
"""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

DB_PATH = Path(__file__).resolve().parent / "wikipedia_clout.db"
SPARQL_URL = "https://query.wikidata.org/sparql"
SLEEP_S = 0.2  # Polite delay between requests


def get_wikidata_qid_from_title(page_title: str) -> Optional[str]:
    """Given an enwiki page_title, return the Wikidata QID or None."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": page_title,
        "format": "json",
        "redirects": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            pageprops = page.get("pageprops", {})
            qid = pageprops.get("wikibase_item")
            if qid:
                return qid
    except Exception:
        pass
    return None


def fetch_wikidata_features(qid: str) -> Optional[Dict]:
    """
    Given a QID like 'Q26876', return a dict with:
    - sitelinks (int)
    - birth_year (int or None)
    - award_count (int)
    - occupations (list[str])
    """
    url = SPARQL_URL
    headers = {"Accept": "application/json", "User-Agent": "CodaClout/1.0"}

    query = f"""
    SELECT
      ?sitelinks
      ?birthDate
      (COUNT(DISTINCT ?award) AS ?awardCount)
      (GROUP_CONCAT(DISTINCT ?occLabel; separator="|") AS ?occupations)
    WHERE {{
      VALUES ?person {{ wd:{qid} }}

      ?person wikibase:sitelinks ?sitelinks.

      OPTIONAL {{ ?person wdt:P569 ?birthDate. }}
      OPTIONAL {{ ?person wdt:P166 ?award. }}

      OPTIONAL {{
        ?person wdt:P106 ?occ.
        ?occ rdfs:label ?occLabel.
        FILTER (LANG(?occLabel) = "en")
      }}
    }}
    GROUP BY ?sitelinks ?birthDate
    """

    try:
        time.sleep(SLEEP_S)
        resp = requests.get(url, params={"query": query}, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"SPARQL error for {qid}: {resp.status_code}")
            return None

        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        row = bindings[0]

        sitelinks = int(row["sitelinks"]["value"])
        award_count = int(row["awardCount"]["value"]) if "awardCount" in row else 0

        birth_year = None
        if "birthDate" in row:
            birth_str = row["birthDate"]["value"]  # e.g. "1989-12-13T00:00:00Z"
            birth_year = int(birth_str[:4])

        occupations = []
        occ_raw = row.get("occupations", {}).get("value")
        if occ_raw:
            occupations = occ_raw.split("|")

        return {
            "sitelinks": sitelinks,
            "birth_year": birth_year,
            "award_count": award_count,
            "occupations": occupations,
        }
    except Exception as e:
        print(f"Error fetching features for {qid}: {e}")
        return None


def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Get all pages that need enrichment (only ones without sitelinks)
    rows = cur.execute(
        """
        SELECT id, page_title, wikidata_qid
        FROM wikipedia_page
        WHERE sitelinks IS NULL
        ORDER BY id
        """
    ).fetchall()

    print(f"Found {len(rows)} pages to enrich (missing sitelinks)...")

    enriched = 0
    skipped = 0
    errors = 0

    for page_id, page_title, qid in rows:
        # Get QID if missing
        if not qid:
            qid = get_wikidata_qid_from_title(page_title)
            if not qid:
                print(f"No QID for {page_title}")
                skipped += 1
                continue
            cur.execute(
                "UPDATE wikipedia_page SET wikidata_qid = ? WHERE id = ?", (qid, page_id)
            )
            con.commit()

        # Fetch metadata
        feats = fetch_wikidata_features(qid)
        if not feats:
            print(f"No metadata for {qid} ({page_title})")
            errors += 1
            continue

        sitelinks = feats["sitelinks"]
        birth_year = feats["birth_year"]
        award_count = feats["award_count"]
        occupations = feats["occupations"]

        # Update main table
        cur.execute(
            """
            UPDATE wikipedia_page
            SET sitelinks = ?, birth_year = ?, award_count = ?
            WHERE id = ?;
            """,
            (sitelinks, birth_year, award_count, page_id),
        )

        # Clear existing occupations for this page
        cur.execute("DELETE FROM wikipedia_occupation WHERE page_id = ?", (page_id,))

        # Insert occupations
        for occ in occupations:
            if occ.strip():
                cur.execute(
                    """
                    INSERT OR IGNORE INTO wikipedia_occupation (page_id, occupation)
                    VALUES (?, ?);
                    """,
                    (page_id, occ.strip()),
                )

        con.commit()
        enriched += 1
        print(f"Enriched {page_title} ({qid}): {sitelinks} sitelinks, {award_count} awards, {len(occupations)} occupations")

    # Verification: count occupations
    occ_count = cur.execute("SELECT COUNT(*) FROM wikipedia_occupation").fetchone()[0]
    unique_occs = cur.execute("SELECT COUNT(DISTINCT occupation) FROM wikipedia_occupation").fetchone()[0]
    
    con.close()

    print(f"\nDone:")
    print(f"  Enriched: {enriched}")
    print(f"  Skipped (no QID): {skipped}")
    print(f"  Errors: {errors}")
    print(f"\nOccupation table stats:")
    print(f"  Total occupation records: {occ_count}")
    print(f"  Unique occupations: {unique_occs}")


if __name__ == "__main__":
    main()

