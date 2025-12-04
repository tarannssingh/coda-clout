from datetime import datetime, timedelta, date
from pathlib import Path
from urllib.parse import urlparse, unquote
import sqlite3
import requests
import time

DB_PATH = Path(__file__).resolve().parent / "wikipedia_clout.db"
SLEEP_S = 0.2  # Polite delay between requests (adjust if hitting rate limits)
RETRIES = 3  # Number of retry attempts
USER_AGENT = "CodaClout/1.0 (example@example.com)"  # Update with your email
DEAD_PEOPLE_LIMIT = 2000  # Target sample size after filtering (increased for better coverage)
SKIP_DEAD_PEOPLE = False  # Set to True to skip dead people processing (if already collected)

with sqlite3.connect(DB_PATH) as con:
    cur = con.cursor()
    def date_diff(date_str, diff, future):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        shifted = date_obj + timedelta(days=diff if future else -diff)
        return shifted.strftime("%Y%m%d")

    def date_to_api(dt: date) -> str:
        return dt.strftime("%Y%m%d")

    def extract_title(url: str) -> str:
        """Extract Wikipedia page title from full URL, handling percent encoding."""
        path = urlparse(url).path
        return unquote(path.split("/wiki/")[1])

    def get_page_creation_date(page_title: str) -> str | None:
        """Get the date when a Wikipedia page was first created."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvlimit": 1,
            "rvdir": "newer",
            "rvprop": "timestamp",
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
                    if "revisions" in page_data and page_data["revisions"]:
                        timestamp = page_data["revisions"][0]["timestamp"]
                        return timestamp.split("T")[0]  # Return YYYY-MM-DD
                return None
            except requests.Timeout:
                if attempt < RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                if attempt < RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Failed to get creation date for {page_title}: {e}")
        return None

    def is_disambiguation(page_title: str) -> bool:
        """Check if a page is a disambiguation page (not a real biography)."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "categories",
            "titles": page_title,
            "cllimit": "max",
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
                    if "categories" in page_data:
                        for cat in page_data["categories"]:
                            if "disambiguation" in cat.get("title", "").lower():
                                return True
                return False
            except requests.Timeout:
                if attempt < RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                if attempt < RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Failed to check disambiguation for {page_title}: {e}")
        return False

    def get_page_length_at_date(page_title: str, before_date: str) -> int:
        """Get page length at a specific date (before_date) by querying revision history.
        
        Args:
            page_title: Wikipedia page title
            before_date: Date string in YYYY-MM-DD format. Returns length of revision just before this date.
        
        Returns:
            Page length in bytes, or 0 if not found.
        """
        url = "https://en.wikipedia.org/w/api.php"
        # Convert date to API format: YYYY-MM-DD -> YYYYMMDDT000000Z
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        rvstart = date_obj.strftime("%Y-%m-%dT00:00:00Z")
        
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
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    if "revisions" in page_data and page_data["revisions"]:
                        # Get the size of the revision just before the date
                        return page_data["revisions"][0].get("size", 0) or 0
                # No revision found before this date (page didn't exist yet)
                return 0
            except requests.Timeout:
                if attempt < RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                if attempt < RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Failed to get historical page length for {page_title} at {before_date}: {e}")
        return 0

    def get_page_info(page_title: str, date_of_death: str = None) -> dict:
        """Get page metadata in one call: length, watchers, edits, editors.
        
        Args:
            page_title: Wikipedia page title
            date_of_death: Optional date string (YYYY-MM-DD). If provided, gets page length
                          at time of death instead of current length.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "info|revisions",
            "titles": page_title,
            "inprop": "watchers",
            "rvprop": "user|timestamp",
            "rvlimit": 500,  # Get last 500 edits (most recent by default)
            "format": "json",
        }
        headers = {"User-Agent": USER_AGENT}
        result = {"length": 0, "watchers": 0, "edits_past_year": 0, "num_editors": 0}
        
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
                    # If date_of_death provided, get historical length; otherwise use current
                    if date_of_death:
                        result["length"] = get_page_length_at_date(page_title, date_of_death)
                    else:
                        result["length"] = page_data.get("length", 0) or 0
                    # Watchers may not always be available, default to 0
                    result["watchers"] = page_data.get("watchers", 0) or 0
                    
                    # Count edits in past year and unique editors
                    if "revisions" in page_data:
                        one_year_ago = datetime.now() - timedelta(days=365)
                        editors = set()
                        edits_count = 0
                        for rev in page_data["revisions"]:
                            rev_time = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                            if rev_time >= one_year_ago:
                                edits_count += 1
                                if "user" in rev:
                                    editors.add(rev["user"])
                        result["edits_past_year"] = edits_count
                        result["num_editors"] = len(editors)
                return result
            except requests.Timeout:
                if attempt < RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                if attempt < RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Failed to get page info for {page_title}: {e}")
        return result

    # 2022-01-01    # start
    # 2022-05-29    # end
    def dead_people_from(start, end, limit=DEAD_PEOPLE_LIMIT, min_sitelinks=None):
        # SPARQL query to find random sample of people who died in a date range
        # SELECT: Return these variables (?person = Wikidata entity, ?personLabel = human-readable name,
        #         ?dateOfDeath = when they died, ?article = Wikipedia URL, ?causeOfDeath = cause of death)
        # WHERE: Conditions that must be true
        #   - ?person wdt:P31 wd:Q5: Person must be an instance of "human" (Q5 is the Wikidata ID for human)
        #   - wdt:P570 ?dateOfDeath: Person must have a date of death property (P570)
        #   - OPTIONAL causeOfDeath: Get cause of death (P509) - powerful predictor
        #   - ?article schema:about ?person: Find Wikipedia articles about this person
        #   - schema:isPartOf <https://en.wikipedia.org/>: Only English Wikipedia articles
        # FILTER: Restrict death dates to the specified range (start to end)
        # SERVICE wikibase:label: Automatically fetch human-readable labels (names) in English
        # ORDER BY RAND(): Randomize results for sampling
        # LIMIT: Only return {limit} random people (prevents processing thousands)
        # Require birth_year (P569) for age_at_death calculation - prioritize rich data
        query = f"""
            SELECT ?person ?personLabel ?dateOfDeath ?article ?causeOfDeath ?causeOfDeathLabel WHERE {{
                ?person wdt:P31 wd:Q5;  
                        wdt:P570 ?dateOfDeath;
                        wdt:P569 ?birthDate.  # REQUIRE birth date for age_at_death

                OPTIONAL {{ ?person wdt:P509 ?causeOfDeath. }}

                ?article schema:about ?person;
                        schema:isPartOf <https://en.wikipedia.org/>.

                FILTER(?dateOfDeath >= "{start["year"]}-{start["month"]}-{start["day"]}T00:00:00Z"^^xsd:dateTime &&
                        ?dateOfDeath <= "{end["year"]}-{end["month"]}-{end["day"]}T00:00:00Z"^^xsd:dateTime) 

                SERVICE wikibase:label {{ 
                    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
                }}
            }}
            ORDER BY RAND()
            LIMIT {limit}
        """
        url = 'https://query.wikidata.org/sparql'
        headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT}
        try:
            sparql = requests.get(url, params={'query': query}, headers=headers, timeout=30)
            sparql.raise_for_status()  # Raise error for bad status codes
            try:
                data = sparql.json()
            except requests.exceptions.JSONDecodeError:
                print(f"SPARQL query failed. Response text: {sparql.text[:500]}")
                print(f"Status code: {sparql.status_code}")
                raise
            if "results" not in data:
                print(f"Unexpected SPARQL response: {data}")
                return []
            results = data["results"]["bindings"]
            return results
        except requests.exceptions.RequestException as e:
            print(f"SPARQL request failed: {e}")
            raise


    def get_wiki_view_data(page_title, start, end):
        headers = {"User-Agent": USER_AGENT}
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{page_title}/daily/{start}/{end}"
        for attempt in range(RETRIES):
            try:
                time.sleep(SLEEP_S)
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code in (403, 429, 503):
                    # Rate limited - wait longer and retry
                    wait_time = 2.0 * (attempt + 1)
                    print(f"Rate limited (403/429), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                if response.status_code == 404:
                    # Page doesn't exist or no data available
                    return []
                if response.status_code != 200:
                    if attempt < RETRIES - 1:
                        print(f"Failed to fetch data: {response.status_code}, retrying...")
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    print(f"Failed to fetch data: {response.status_code}")
                    return []
                return response.json().get('items', [])
            except requests.Timeout:
                if attempt < RETRIES - 1:
                    print(f"Timeout fetching views for {page_title}, retrying...")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                print(f"Timeout fetching views for {page_title} after {RETRIES} attempts")
                return []
            except Exception as e:
                if attempt < RETRIES - 1:
                    print(f"Error fetching views for {page_title} (attempt {attempt + 1}): {e}")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                print(f"Error fetching views for {page_title}: {e}")
                return []
        return []

    def living_people_sample(limit, min_sitelinks=3):
        # Better approach: Query by occupation (faster, more reliable than birth year)
        # Each occupation query is small and specific, avoiding timeouts
        import random
        
        all_living = []
        
        # Notable occupations that likely have living people with Wikipedia pages
        # Using QIDs for common notable occupations
        occupations = [
            ("Q33999", "actor"),  # Actor
            ("Q177220", "singer"),  # Singer
            ("Q36180", "writer"),  # Writer
            ("Q2066131", "athlete"),  # Athlete
            ("Q2526255", "film_director"),  # Film director
            ("Q82955", "politician"),  # Politician
            ("Q901", "scientist"),  # Scientist
            ("Q43845", "businessperson"),  # Businessperson
            ("Q639669", "musician"),  # Musician
            ("Q193391", "journalist"),  # Journalist
        ]
        
        random.shuffle(occupations)
        chunk_size = 20  # Small chunks to avoid timeouts
        
        for occ_qid, occ_name in occupations:
            if len(all_living) >= limit:
                break
            
            # Simple query: living people with this occupation who have Wikipedia pages
            query = f"""
                SELECT ?person ?personLabel ?article WHERE {{
                    ?person wdt:P31 wd:Q5.
                    FILTER NOT EXISTS {{ ?person wdt:P570 ?dateOfDeath. }}
                    ?person wdt:P106 wd:{occ_qid}.
                    ?article schema:about ?person;
                            schema:isPartOf <https://en.wikipedia.org/>.
                    SERVICE wikibase:label {{ 
                        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
                    }}
                }}
                ORDER BY RAND()
                LIMIT {chunk_size}
            """
            
            url = 'https://query.wikidata.org/sparql'
            headers = {'Accept': 'application/json', 'User-Agent': USER_AGENT}
            
            for attempt in range(RETRIES):
                try:
                    time.sleep(SLEEP_S * 3)  # Be extra polite
                    response = requests.get(url, params={'query': query}, headers=headers, timeout=45)
                    if response.status_code in (504, 503, 429):
                        wait_time = 3.0 * (attempt + 1)
                        print(f"Rate limited for {occ_name} (attempt {attempt + 1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    if "results" not in data:
                        print(f"Unexpected SPARQL response for {occ_name}: {data}")
                        break
                    chunk_results = data["results"]["bindings"]
                    all_living.extend(chunk_results)
                    print(f"Got {len(chunk_results)} living {occ_name}s (total: {len(all_living)})")
                    break
                except requests.Timeout:
                    if attempt < RETRIES - 1:
                        wait_time = 3.0 * (attempt + 1)
                        print(f"Timeout for {occ_name} (attempt {attempt + 1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    print(f"Timeout after {RETRIES} attempts for {occ_name}, skipping...")
                    break
                except Exception as e:
                    if attempt < RETRIES - 1:
                        print(f"Error fetching {occ_name} (attempt {attempt + 1}): {e}")
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    print(f"Failed to fetch {occ_name} after {RETRIES} attempts: {e}")
                    break
        
        # If we still need more, try a simpler query without occupation filter
        if len(all_living) < limit:
            print(f"Only got {len(all_living)}/{limit}, trying simpler query...")
            simple_query = """
                SELECT ?person ?personLabel ?article WHERE {
                    ?person wdt:P31 wd:Q5.
                    FILTER NOT EXISTS { ?person wdt:P570 ?dateOfDeath. }
                    ?article schema:about ?person;
                            schema:isPartOf <https://en.wikipedia.org/>.
                    SERVICE wikibase:label { 
                        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
                    }
                }
                ORDER BY RAND()
                LIMIT 50
            """
            
            for attempt in range(RETRIES):
                try:
                    time.sleep(SLEEP_S * 3)
                    response = requests.get(url, params={'query': simple_query}, headers=headers, timeout=45)
                    if response.status_code in (504, 503, 429):
                        wait_time = 5.0 * (attempt + 1)
                        print(f"Rate limited on simple query (attempt {attempt + 1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    response.raise_for_status()
                    data = response.json()
                    if "results" in data:
                        chunk_results = data["results"]["bindings"]
                        all_living.extend(chunk_results)
                        print(f"Got {len(chunk_results)} more from simple query (total: {len(all_living)})")
                    break
                except Exception as e:
                    if attempt < RETRIES - 1:
                        print(f"Simple query error (attempt {attempt + 1}): {e}")
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    print(f"Simple query failed after {RETRIES} attempts")
                    break
        
        # Shuffle and deduplicate by article URL
        seen_articles = set()
        unique_living = []
        for person in all_living:
            article = person.get("article", {}).get("value", "")
            if article and article not in seen_articles:
                seen_articles.add(article)
                unique_living.append(person)
        
        random.shuffle(unique_living)
        return unique_living[:limit]


    def add_person_to_tables(name, page_title, qid, date_of_death=None, cause_of_death=None):
        # QID is already known from SPARQL results, no API call needed
        wikidata_qid = qid
        
        # CRITICAL VALIDITY FILTERS: Ensure scientific rigor
        # Filter 0: Skip disambiguation pages (not real biographies)
        if is_disambiguation(page_title):
            print(f"Skipping {name}: disambiguation page")
            return
        
        page_created = get_page_creation_date(page_title)
        if not page_created:
            print(f"Skipping {name}: could not determine page creation date")
            return
        
        if date_of_death:
            # Filter 1: Page must exist before death
            if page_created > date_of_death:
                print(f"Skipping {name}: page created {page_created} after death {date_of_death}")
                return
            
            # Filter 2: Page must have existed ≥90 days before death (avoid illness-stub pages)
            death_dt = datetime.strptime(date_of_death, "%Y-%m-%d")
            created_dt = datetime.strptime(page_created, "%Y-%m-%d")
            days_page_existed = (death_dt - created_dt).days
            if days_page_existed < 90:
                print(f"Skipping {name}: page existed only {days_page_existed} days before death (need ≥90)")
                return
        
        # Get page metadata (length, watchers, edits, editors) - GOD-TIER features
        # Pass date_of_death to get historical page length at time of death (not current)
        page_info = get_page_info(page_title, date_of_death=date_of_death)
        page_len_bytes = page_info.get("length", 0)
        if page_len_bytes is None:
            page_len_bytes = 0  # Handle None case
        page_watchers = page_info.get("watchers", 0) or 0
        edits_past_year = page_info.get("edits_past_year", 0) or 0
        num_editors = page_info.get("num_editors", 0) or 0
        
        if date_of_death:
            start = date_diff(date_of_death, 11, False)
            end = date_diff(date_of_death, 1, False)
        else:
            anchor = date.today()
            start = date_to_api(anchor - timedelta(days=10))
            end = date_to_api(anchor - timedelta(days=1))

        views = get_wiki_view_data(page_title, start, end)
        
        # Defensive check: skip if no pre-death views (corrupted/protected/nonexistent pages)
        if date_of_death and (not views or len(views) == 0):
            print(f"Skipping {name}: no pre-death views (likely corrupted/protected/nonexistent page)")
            return
        
        avg_views_pre_death_10d = (
            sum(view["views"] for view in views) // len(views) if views and date_of_death else None
        )
        
        # Fetch full 365-day pre-death window for feature engineering
        # For living people, use today as anchor (symmetric with dead people)
        if date_of_death:
            start_365 = date_diff(date_of_death, 365, False)
            end_365 = date_diff(date_of_death, 1, False)
            views_pre_year = get_wiki_view_data(page_title, start_365, end_365)
        else:
            # Living people: use last 365 days as "pre-death" window
            anchor = date.today()
            start_365 = date_to_api(anchor - timedelta(days=365))
            end_365 = date_to_api(anchor - timedelta(days=1))
            views_pre_year = get_wiki_view_data(page_title, start_365, end_365)
        
        # add person to person table with all metadata (including GOD-TIER features)
        try:
            id = (cur.execute("""
                INSERT INTO wikipedia_page (name, page_title, date_of_death, avg_views_pre_death_10d, wikidata_qid, page_created, page_len_bytes, page_watchers, edits_past_year, num_editors, cause_of_death)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;
            """, (name, page_title, date_of_death, avg_views_pre_death_10d, wikidata_qid, page_created, page_len_bytes, page_watchers, edits_past_year, num_editors, cause_of_death))).fetchone()[0]
        except sqlite3.IntegrityError as e:
            id = (cur.execute("""
                SELECT 
                    id 
                FROM
                    wikipedia_page
                WHERE 
                    page_title = ?
            """, (page_title,))).fetchone()[0]
            print(f"duplicate value for {page_title}")
        except sqlite3.OperationalError as e:
            print("Malformed data, so skipped")
            return

        # Store the 365-day pre-death window views (for feature engineering)
        # Works for both dead and living people (living uses NULL for day_since_death)
        if views_pre_year:
            for view_object in views_pre_year:
                view = view_object["views"]
                date_of_view = view_object["timestamp"]
                date_of_view = f"{date_of_view[:4]}-{date_of_view[4:6]}-{date_of_view[6:8]}"
                
                # Calculate days_since_death (negative for pre-death, NULL for living)
                if date_of_death:
                    days_since_death = (datetime.strptime(str(date_of_view), '%Y-%m-%d') - datetime.strptime(str(date_of_death), '%Y-%m-%d')).days
                else:
                    days_since_death = None  # Living people
                
                cur.execute("""
                    INSERT OR IGNORE INTO wikipedia_daily_clout (page_id, views, date, day_since_death)
                    VALUES (?, ?, ?, ?);
                """, (id, view, date_of_view, days_since_death))
        
        # then get long-tail view data (2 years before to 3 years after death)
        if date_of_death:
            start = date_diff(date_of_death, 365 * 2, False) # 2 years before
            end = date_diff(date_of_death, 365 * 3, True) # 3 years after
        else:
            anchor = date.today()
            start = date_to_api(anchor - timedelta(days=365 * 3))
            end = date_to_api(anchor)

        views = get_wiki_view_data(page_title, start, end)
        
        # Store all views, but you could filter here if needed:
        # - Keep daily for critical windows: -30 to +365 days
        # - Sample every 3 days for long tail: outside that window
        for view_object in views:
            view = view_object["views"]
            date_of_view = view_object["timestamp"]
            date_of_view = f"{date_of_view[:4]}-{date_of_view[4:6]}-{date_of_view[6:8]}"

            days_since_death = (datetime.strptime(str(date_of_view), '%Y-%m-%d') - datetime.strptime(str(date_of_death), '%Y-%m-%d')).days if date_of_death else None
            
            # Smart sampling: Keep daily for critical windows, sample long tail
            # Critical: -365 to +365 (full pre-death year + post-death year for sustain metrics)
            # Long tail: before -365 or after +365 (just for context, can sample)
            if date_of_death and days_since_death is not None:
                if days_since_death < -365 or days_since_death > 365:
                    # Long tail: only store every 3rd day (reduces storage, minimal impact)
                    if days_since_death % 3 != 0:
                        continue
            
            # Insert for both dead and living people
            cur.execute("""
                INSERT OR IGNORE INTO wikipedia_daily_clout (page_id, views, date, day_since_death)
                VALUES (?, ?, ?, ?);
            """, (id, view, date_of_view, days_since_death))
            # Duplicates are expected (365-day window overlaps with long-tail) - silently skip

    # Expanded date range: 2018-2025 for better generalizability and more recent data
    # Strategy: Collect from ALL years first, then sample evenly across years
    # This ensures temporal diversity for both EDA and modeling
    all_dead_people = []
    
    if not SKIP_DEAD_PEOPLE:
        start_year = 2017  # Extended back to 2017 for more data
        end_year = 2025  # Extended to 2025 for more recent data
        
        # First pass: Collect candidates from all years (smaller chunks per year)
        # Target ~100-150 per year to ensure diversity
        # Prioritize recent years (2023-2025) which have more legends
        per_year_target = max(100, DEAD_PEOPLE_LIMIT // (end_year - start_year + 1))
        recent_year_target = int(per_year_target * 1.5)  # Get more from 2023-2025
        
        print(f"Collecting dead people from {start_year}-{end_year}...")
        print(f"Target: ~{per_year_target} per year for temporal diversity")
        
        # Collect more from recent years (2023-2025) which have higher legend rates
        for year in range(start_year, end_year + 1):
            # Use higher target for recent years (2023-2025) which have more legends
            year_target = recent_year_target if year >= 2023 else per_year_target
            # Split each year into quarters for better distribution
            quarters = [
                ({"year": str(year), "month": "1", "day": "1"}, {"year": str(year), "month": "3", "day": "31"}),
                ({"year": str(year), "month": "4", "day": "1"}, {"year": str(year), "month": "6", "day": "30"}),
                ({"year": str(year), "month": "7", "day": "1"}, {"year": str(year), "month": "9", "day": "30"}),
                ({"year": str(year), "month": "10", "day": "1"}, {"year": str(year), "month": "12", "day": "31"}),
            ]
            
            year_people = []
            for q_start, q_end in quarters:
                if len(year_people) >= per_year_target:
                    break
                print(f"  Fetching {year} Q{quarters.index((q_start, q_end)) + 1} ({q_start['month']}/{q_start['day']} to {q_end['month']}/{q_end['day']})...")
                try:
                    chunk_results = dead_people_from(q_start, q_end, limit=year_target // 2)
                    year_people.extend(chunk_results)
                    print(f"    Got {len(chunk_results)} people (year total: {len(year_people)})")
                    time.sleep(1.0)  # Be extra polite between queries
                except Exception as e:
                    print(f"    Error fetching quarter: {e}, continuing...")
                    continue
            
            # Sample evenly from this year
            import random
            if len(year_people) > per_year_target:
                random.shuffle(year_people)
                year_people = year_people[:per_year_target]
            
            all_dead_people.extend(year_people)
            print(f"  Year {year}: {len(year_people)} people collected (total: {len(all_dead_people)})")
        
        # Final sampling: Ensure we have good distribution but hit our target
        import random
        if len(all_dead_people) > DEAD_PEOPLE_LIMIT:
            # Group by year to maintain distribution
            by_year = {}
            for person in all_dead_people:
                try:
                    year = person["dateOfDeath"]["value"].split("T")[0][:4]
                    if year not in by_year:
                        by_year[year] = []
                    by_year[year].append(person)
                except:
                    continue
            
            # Sample proportionally from each year
            final_sample = []
            target_per_year = DEAD_PEOPLE_LIMIT // len(by_year)
            for year, people in by_year.items():
                random.shuffle(people)
                final_sample.extend(people[:min(target_per_year, len(people))])
            
            # Fill remaining slots randomly if needed
            if len(final_sample) < DEAD_PEOPLE_LIMIT:
                remaining = [p for p in all_dead_people if p not in final_sample]
                random.shuffle(remaining)
                final_sample.extend(remaining[:DEAD_PEOPLE_LIMIT - len(final_sample)])
            
            all_dead_people = final_sample[:DEAD_PEOPLE_LIMIT]
            print(f"Final sample: {len(all_dead_people)} people across {len(by_year)} years")
    
    if not SKIP_DEAD_PEOPLE:
        print(f"Processing {len(all_dead_people)} dead people...")
        for i, person in enumerate(all_dead_people):
            try:
                name = person["personLabel"]["value"]
                qid = person["person"]["value"].split("/")[-1]  # Extract QID from "http://www.wikidata.org/entity/Q12345"
                date_of_death = person["dateOfDeath"]["value"].split("T")[0]
                page_title = extract_title(person["article"]["value"])
                # Extract cause of death if available (powerful predictor)
                cause_of_death = person.get("causeOfDeathLabel", {}).get("value") if "causeOfDeathLabel" in person else None
                print(f"[{i+1}] Processing: {name} ({date_of_death})")
                add_person_to_tables(name, page_title, qid, date_of_death=date_of_death, cause_of_death=cause_of_death)
                con.commit()
                print(f"[{i+1}] Completed: {name}")
                # Extra delay every 10 people to be extra polite
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1} people, taking a short break...")
                    time.sleep(1.0)
            except Exception as e:
                print(f"Error processing person: {e}")
                con.rollback()
                continue
    else:
        print("Skipping dead people processing (SKIP_DEAD_PEOPLE=True)")
    
    # Collect living people for comparison (control group)
    print("\n" + "="*60)
    print("Collecting living people sample (control group)...")
    print("="*60)
    for i, person in enumerate(living_people_sample(limit=150, min_sitelinks=3)):
        try:
            name = person["personLabel"]["value"]
            qid = person["person"]["value"].split("/")[-1]  # Extract QID from "http://www.wikidata.org/entity/Q12345"
            page_title = extract_title(person["article"]["value"])
            print(f"[Living {i+1}/150] Processing: {name}")
            add_person_to_tables(name, page_title, qid, date_of_death=None)
            con.commit()
            print(f"[Living {i+1}/150] Completed: {name}")
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} living people, taking a short break...")
                time.sleep(1.0)
        except Exception as e:
            print(f"Error processing living person: {e}")
            con.rollback()
            continue
