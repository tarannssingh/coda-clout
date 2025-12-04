"""
Quick script to update page_len_bytes, num_editors to historical values (DAY BEFORE death).
This fixes the data leakage issue without re-collecting everything.

Usage:
    python fix_historical_page_length.py

This will:
1. Read modeling_data_balanced.csv
2. For each person with date_of_death, fetch historical values the DAY BEFORE death
3. Update page_len_bytes, num_editors columns
4. Note: page_watchers cannot be gotten historically (current metadata only)
5. Save to modeling_data_balanced_fixed.csv
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

# Configuration
SLEEP_S = 0.2  # Polite delay between requests
RETRIES = 3
USER_AGENT = "CodaClout/1.0 (example@example.com)"
CSV_IN = Path(__file__).parent / "modeling_data_upgraded.csv"
CSV_OUT = Path(__file__).parent / "modeling_data_upgraded_fixed.csv"
PROGRESS_FILE = Path(__file__).parent / ".page_length_progress.json"

def get_day_before_death(date_of_death: str) -> str:
    """Get the day before death date."""
    try:
        date_obj = datetime.strptime(date_of_death, "%Y-%m-%d")
        day_before = date_obj - timedelta(days=1)
        return day_before.strftime("%Y-%m-%d")
    except ValueError:
        try:
            date_obj = datetime.strptime(date_of_death.split()[0], "%Y-%m-%d")
            day_before = date_obj - timedelta(days=1)
            return day_before.strftime("%Y-%m-%d")
        except:
            return date_of_death

def get_page_length_at_date(page_title: str, before_date: str) -> tuple:
    """Get page length at a specific date (before_date) by querying revision history.
    
    Returns:
        (page_length, revision_timestamp) or (0, "") if not found
    """
    url = "https://en.wikipedia.org/w/api.php"
    try:
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        try:
            date_obj = datetime.strptime(before_date.split()[0], "%Y-%m-%d")
            rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
        except:
            return (0, "")
    
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvlimit": 1,
        "rvdir": "older",  # Get older revisions (before the date)
        "rvstart": rvstart,  # Start from this timestamp
        "rvprop": "timestamp|size",  # Get size (length) of revision
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(RETRIES):
        try:
            time.sleep(SLEEP_S)
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code in (403, 429, 503):
                wait_time = 2.0 * (attempt + 1)
                print(f"   Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                if "revisions" in page_data and page_data["revisions"]:
                    rev = page_data["revisions"][0]
                    return (rev.get("size", 0) or 0, rev.get("timestamp", ""))
            return (0, "")
        except requests.Timeout:
            if attempt < RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            print(f"   Error: {e}")
    return (0, "")

def get_historical_num_editors(page_title: str, before_date: str) -> int:
    """Get number of unique editors in the year before death date.
    
    Counts unique editors from revisions in the 365 days before the date.
    """
    url = "https://en.wikipedia.org/w/api.php"
    try:
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
        one_year_before = date_obj - timedelta(days=365)
        rvend = one_year_before.strftime("%Y-%m-%dT00:00:00Z")
    except ValueError:
        try:
            date_obj = datetime.strptime(before_date.split()[0], "%Y-%m-%d")
            rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
            one_year_before = date_obj - timedelta(days=365)
            rvend = one_year_before.strftime("%Y-%m-%dT00:00:00Z")
        except:
            return 0
    
    editors = set()
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvlimit": 500,  # Get up to 500 revisions
        "rvdir": "older",  # Get older revisions
        "rvstart": rvstart,  # Start from before_date
        "rvend": rvend,  # End at one year before
        "rvprop": "timestamp|user",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(RETRIES):
        try:
            time.sleep(SLEEP_S)
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code in (403, 429, 503):
                wait_time = 2.0 * (attempt + 1)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                if "revisions" in page_data:
                    for rev in page_data["revisions"]:
                        if "user" in rev and rev["user"]:
                            editors.add(rev["user"])
            return len(editors)
        except requests.Timeout:
            if attempt < RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))
                continue
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            print(f"   Error getting editors: {e}")
    return 0

def load_progress():
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_ids):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_ids), f)

def main():
    print("="*60)
    print("FIXING HISTORICAL VALUES (DAY BEFORE DEATH)")
    print("="*60)
    print(f"\nReading: {CSV_IN}")
    print("Updating: page_len_bytes, num_editors")
    print("Note: page_watchers cannot be gotten historically (current metadata only)")
    print()
    
    if not CSV_IN.exists():
        print(f"ERROR: {CSV_IN} not found!")
        return
    
    df = pd.read_csv(CSV_IN)
    print(f"Loaded {len(df)} rows")
    
    # Filter to rows with date_of_death
    has_death = df['date_of_death'].notna()
    to_update = df[has_death].copy()
    print(f"Rows with date_of_death: {len(to_update)}")
    
    # Load progress (skip already processed)
    processed = load_progress()
    if processed:
        print(f"Resuming: {len(processed)} already processed")
    
    # Update historical values
    updated_length = 0
    updated_editors = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in to_update.iterrows():
        # Skip if already processed
        if idx in processed:
            skipped_count += 1
            if skipped_count % 50 == 0:
                print(f"[{skipped_count} skipped so far...]")
            continue
        
        page_title = row['page_title']
        date_of_death = row['date_of_death']
        day_before = get_day_before_death(date_of_death)
        
        old_length = row['page_len_bytes']
        old_editors = row['num_editors'] if 'num_editors' in row else 0
        
        name = row['name'] if pd.notna(row['name']) and row['name'] else page_title
        
        current_num = updated_length + skipped_count + error_count + 1
        print(f"\n[{current_num}/{len(to_update)}] {name}")
        print(f"   Death: {date_of_death}, Day before: {day_before}")
        print(f"   Current: length={old_length:,} bytes, editors={old_editors}")
        
        # Get historical page length (day before death)
        new_length, rev_timestamp = get_page_length_at_date(page_title, day_before)
        
        if new_length > 0:
            df.at[idx, 'page_len_bytes'] = new_length
            updated_length += 1
            diff = new_length - old_length
            diff_pct = (diff / old_length * 100) if old_length > 0 else 0
            print(f"   ✓ Length: {old_length:,} → {new_length:,} bytes ({diff:+,}, {diff_pct:+.1f}%)")
            if rev_timestamp:
                print(f"      Revision: {rev_timestamp}")
        else:
            error_count += 1
            print(f"   ⚠ No historical revision found for length (keeping {old_length:,})")
        
        # Get historical num_editors (year before death)
        new_editors = get_historical_num_editors(page_title, day_before)
        if new_editors > 0:
            df.at[idx, 'num_editors'] = new_editors
            updated_editors += 1
            print(f"   ✓ Editors: {old_editors} → {new_editors} (year before death)")
        else:
            print(f"   ⚠ Could not get historical editors (keeping {old_editors})")
        
        # Save progress every 10 rows
        processed.add(idx)
        if current_num % 10 == 0:
            save_progress(processed)
            print(f"\n   [Progress saved] Length: {updated_length}, Editors: {updated_editors}, Errors: {error_count}")
    
    # Final save
    save_progress(processed)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Updated page_len_bytes: {updated_length}")
    print(f"Updated num_editors: {updated_editors}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Errors/not found: {error_count}")
    print(f"Total processed: {updated_length + skipped_count + error_count}")
    
    # Save updated CSV
    print(f"\nSaving to: {CSV_OUT}")
    df.to_csv(CSV_OUT, index=False)
    print("✓ Done!")
    print(f"\nNext steps:")
    print(f"1. Review {CSV_OUT}")
    print(f"2. Re-run upgrade_features.py to regenerate log features")
    print(f"3. Update train_baselines.py to remove log_edits_past_year from features")
    print(f"4. Re-train the model")

if __name__ == "__main__":
    main()
