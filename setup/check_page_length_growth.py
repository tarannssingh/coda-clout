"""
Quick script to check how much a Wikipedia page grew after death.
Example: Sidhu Moose Wala (died 2022-05-29)
"""

import requests
import time
from datetime import datetime, timedelta

USER_AGENT = "CodaClout/1.0 (example@example.com)"

def get_page_length_at_date(page_title: str, before_date: str) -> int:
    """Get page length at a specific date."""
    url = "https://en.wikipedia.org/w/api.php"
    try:
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        return 0
    
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": rvstart,
        "rvprop": "timestamp|size",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    
    try:
        time.sleep(0.2)
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            if "revisions" in page_data and page_data["revisions"]:
                rev = page_data["revisions"][0]
                return rev.get("size", 0) or 0, rev.get("timestamp", "")
        return 0, ""
    except Exception as e:
        print(f"Error: {e}")
        return 0, ""

def get_current_page_length(page_title: str) -> int:
    """Get current page length."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "info",
        "titles": page_title,
        "inprop": "length",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    
    try:
        time.sleep(0.2)
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            return page_data.get("length", 0) or 0
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Check Sidhu Moose Wala
page_title = "Sidhu_Moose_Wala"
death_date = "2022-05-29"
day_before = "2022-05-28"

print("="*60)
print(f"PAGE LENGTH GROWTH: {page_title}")
print("="*60)
print(f"Death date: {death_date}")
print(f"Checking page length on: {day_before} (day before death)")
print()

# Get historical length
print("Fetching historical page length...")
hist_length, hist_timestamp = get_page_length_at_date(page_title, day_before)
print(f"✓ Historical length (day before death): {hist_length:,} bytes")
if hist_timestamp:
    print(f"  Revision timestamp: {hist_timestamp}")

print()
print("Fetching current page length...")
current_length = get_current_page_length(page_title)
print(f"✓ Current length (today): {current_length:,} bytes")

print()
print("="*60)
print("COMPARISON")
print("="*60)
if hist_length > 0:
    growth = current_length - hist_length
    growth_pct = (growth / hist_length) * 100 if hist_length > 0 else 0
    print(f"Growth: {growth:,} bytes ({growth_pct:+.1f}%)")
    print(f"  Before death: {hist_length:,} bytes")
    print(f"  Today:        {current_length:,} bytes")
    print()
    if growth_pct > 50:
        print("⚠️  SIGNIFICANT GROWTH - This is a serious data leakage issue!")
        print("   The page grew substantially after death, which leaks information.")
    elif growth_pct > 20:
        print("⚠️  MODERATE GROWTH - This could affect model predictions.")
    else:
        print("✓ MINIMAL GROWTH - Less of a concern.")
else:
    print("⚠️  Could not fetch historical length (page may not have existed)")

