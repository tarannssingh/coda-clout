"""
Populate database with curated list of known legends (2018-2023)
High-profile deaths that are obvious legends or culturally significant tragic deaths
"""

import sqlite3
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "wikipedia_clout.db"

# Curated list of obvious legends and high-profile tragic deaths (2018-2023)
KNOWN_LEGENDS = [
    # Music - Hip Hop/Rap Deaths
    'Mac_Miller',              # 2018 - Overdose, massive posthumous attention
    'XXXTentacion',            # 2018 - Shot, huge Gen-Z icon
    'Juice_Wrld',              # 2019 - Overdose, massive streaming numbers posthumously
    'Nipsey_Hussle',           # 2019 - Shot, community icon
    'Pop_Smoke',               # 2020 - Shot, posthumous album success
    'DMX',                     # 2021 - Overdose, hip-hop legend
    'Young_Dolph',             # 2021 - Shot, Memphis legend
    'PnB_Rock',                # 2022 - Shot, tragic restaurant killing
    'Takeoff_(rapper)',        # 2022 - Shot, Migos member
    'Coolio',                  # 2022 - Hip-hop icon (Gangsta's Paradise)

    # International Hip Hop
    'Sidhu_Moose_Wala',        # 2022 - Punjabi rapper, massive global impact

    # Electronic/DJ
    'Avicii',                  # 2018 - Suicide, EDM legend
    'Keith_Flint',             # 2019 - Suicide, The Prodigy frontman

    # Rock/Alternative
    'Taylor_Hawkins',          # 2022 - Foo Fighters drummer, overdose

    # Pop/R&B
    'Aaron_Carter',            # 2022 - Child star turned musician

    # Sports Icons
    'Kobe_Bryant',             # 2020 - Helicopter crash, global mourning
    'Emiliano_Sala',           # 2019 - Plane crash, footballer

    # Actors/TV
    'Cameron_Boyce',           # 2019 - Disney star, epilepsy
    'Naya_Rivera',             # 2020 - Glee star, drowning
    'Anne_Heche',              # 2022 - Car crash, actress
    'Matthew_Perry',           # 2023 - Friends star, drowning
    'Michael_K._Williams',     # 2021 - The Wire star, overdose
    'Angus_Cloud',             # 2023 - Euphoria star, overdose

    # Celebrity Chefs/TV Personalities
    'Anthony_Bourdain',        # 2018 - Suicide, beloved chef/host

    # Fashion/Models
    'Kate_Spade',              # 2018 - Suicide, fashion designer
    'Stella_Tennant',          # 2020 - Suicide, supermodel

    # K-pop/Asian Entertainment
    'Goo_Hara',                # 2019 - Suicide, K-pop star
    'Yuko_Takeuchi',           # 2020 - Suicide, Japanese actress
    'Sulli',                   # 2019 - Suicide, K-pop star
    'Moonbin',                 # 2023 - Suicide, K-pop star (Astro)
    'Lee_Sun-kyun',            # 2023 - Suicide, Parasite actor

    # Bollywood
    'Sushant_Singh_Rajput',    # 2020 - Suicide, Bollywood actor, massive controversy

    # TV/Reality/Social Media
    'Caroline_Flack',          # 2020 - Suicide, Love Island host
    'Stephen_Boss',            # 2022 - Suicide, DJ tWitch
    'Naomi_Judd',              # 2022 - Suicide, country music legend
    'Cheslie_Kryst',           # 2022 - Suicide, Miss USA 2019

    # Film Directors
    'Jean-Luc_Godard',         # 2022 - Assisted suicide, French New Wave legend

    # TV/Wrestling
    'Jason_David_Frank',       # 2022 - Suicide, Power Rangers icon

    # Character Actors
    'Verne_Troyer',            # 2018 - Alcohol, Mini-Me from Austin Powers
    'Margot_Kidder',           # 2018 - Suicide, Lois Lane actress
    'Mark_Salling',            # 2018 - Suicide, Glee actor

    # Additional High-Profile Deaths (2016-2023)
    'Chadwick_Boseman',        # 2020 - Cancer, Black Panther
    'John_Lewis',              # 2020 - Cancer, Civil Rights icon
    'Ruth_Bader_Ginsburg',     # 2020 - Cancer, Supreme Court Justice
    'George_Floyd',            # 2020 - Murder, sparked global protests
    'Diego_Maradona',          # 2020 - Heart attack, soccer legend
    'Paul_Walker',             # 2013 - Car crash (before range but legendary)
    'Robin_Williams',          # 2014 - Suicide (before range but legendary)
    'Prince',                  # 2016 - Overdose (before range but legendary)
    'David_Bowie',             # 2016 - Cancer (before range but legendary)
    'Alan_Rickman',            # 2016 - Cancer, Harry Potter/Die Hard
    'Carrie_Fisher',           # 2016 - Heart attack, Princess Leia
    'Stan_Lee',                # 2018 - Natural causes, Marvel legend
    'Stephen_Hawking',         # 2018 - Natural causes, physicist icon
    'George_H._W._Bush',       # 2018 - Natural causes, US President
    'John_McCain',             # 2018 - Cancer, US Senator/war hero
    'Aretha_Franklin',         # 2018 - Cancer, Queen of Soul
    'Luke_Perry',              # 2019 - Stroke, Beverly Hills 90210
    'Doris_Day',               # 2019 - Natural causes, Hollywood icon
    'Peter_Mayhew',            # 2019 - Natural causes, Chewbacca
    'Peggy_Lipton',            # 2019 - Cancer, Twin Peaks actress
    'Bill_Paxton',             # 2017 - Surgery complications, character actor
    'Tom_Petty',               # 2017 - Overdose, rock legend
    'Chris_Cornell',           # 2017 - Suicide, Soundgarden/Audioslave
    'Chester_Bennington',      # 2017 - Suicide, Linkin Park
    'Kirstie_Alley',           # 2022 - Cancer, Cheers actress
    'Bob_Saget',               # 2022 - Head injury, Full House
    'Ray_Liotta',              # 2022 - Heart issues, Goodfellas
    'Olivia_Newton-John',      # 2022 - Cancer, Grease star
    'Angela_Lansbury',         # 2022 - Natural causes, Murder She Wrote
    'Irene_Cara',              # 2022 - Natural causes, Flashdance
    'Lisa_Marie_Presley',      # 2023 - Cardiac arrest, Elvis daughter
    'Tina_Turner',             # 2023 - Natural causes, music legend
    'Sinead_O\'Connor',        # 2023 - Natural causes, Nothing Compares 2 U
    'Tony_Bennett',            # 2023 - Natural causes, jazz legend
    'Harry_Belafonte',         # 2023 - Natural causes, singer/activist
    'Paul_Reubens',            # 2023 - Cancer, Pee-wee Herman
    'Raquel_Welch',            # 2023 - Cardiac arrest, actress icon

    # More Hip Hop/Rap
    'Lil_Peep',                # 2017 - Overdose, emo rap pioneer
    'Fredo_Santana',           # 2018 - Seizure, Chicago drill rapper
    'Jimmy_Wopo',              # 2018 - Shot, Pittsburgh rapper
    'Smoke_Dawg',              # 2018 - Shot, Toronto rapper
    'Lil_Marlo',               # 2020 - Shot, Atlanta rapper
    'King_Von',                # 2020 - Shot, Chicago drill rapper
    'MF_DOOM',                 # 2020 - Natural causes, underground legend
    'Biz_Markie',              # 2021 - Diabetes, hip-hop icon
    'Shock_G',                 # 2021 - Overdose, Digital Underground
    'Drakeo_the_Ruler',        # 2021 - Stabbed, LA rapper
    'Gangsta_Boo',             # 2023 - Overdose, Three 6 Mafia

    # More Rock/Alternative
    'Dolores_O\'Riordan',      # 2018 - Drowning, The Cranberries
    'Scott_Weiland',           # 2015 - Overdose, Stone Temple Pilots
    'Eddie_Van_Halen',         # 2020 - Cancer, Van Halen guitarist
    'Neil_Peart',              # 2020 - Cancer, Rush drummer
    'Dusty_Hill',              # 2021 - Natural causes, ZZ Top
    'Charlie_Watts',           # 2021 - Natural causes, Rolling Stones drummer
    'Jeff_Beck',               # 2023 - Meningitis, guitar legend
    'David_Crosby',            # 2023 - Natural causes, Crosby Stills Nash

    # More Actors/Directors
    'Brittany_Murphy',         # 2009 - Pneumonia (before range but notable)
    'Heath_Ledger',            # 2008 - Overdose (before range but legendary)
    'Philip_Seymour_Hoffman',  # 2014 - Overdose (before range but legendary)
    'Anton_Yelchin',           # 2016 - Car accident, Star Trek
    'Burt_Reynolds',           # 2018 - Cardiac arrest, Smokey and the Bandit
    'Penny_Marshall',          # 2018 - Diabetes, Laverne & Shirley director
    'Albert_Finney',           # 2019 - Cancer, British actor
    'Rip_Torn',                # 2019 - Natural causes, character actor
    'Peter_Fonda',             # 2019 - Lung failure, Easy Rider
    'Rutger_Hauer',            # 2019 - Natural causes, Blade Runner
    'Kirk_Douglas',            # 2020 - Natural causes, Hollywood legend (103)
    'Sean_Connery',            # 2020 - Natural causes, James Bond
    'James_Caan',              # 2022 - Heart attack, The Godfather
    'Meat_Loaf',               # 2022 - COVID, singer/actor
    'Nichelle_Nichols',        # 2022 - Natural causes, Star Trek Uhura
    'James_Earl_Jones',        # 2024 - Natural causes, Darth Vader voice (if applicable)

    # Comedy Legends
    'John_Witherspoon',        # 2019 - Heart attack, Friday movies
    'Norm_Macdonald',          # 2021 - Cancer (kept secret), SNL legend
    'Gilbert_Gottfried',       # 2022 - Heart disease, comedian/voice actor
    'Louie_Anderson',          # 2022 - Cancer, comedian

    # Sports (Additional)
    'Muhammad_Ali',            # 2016 - Parkinson's, boxing legend
    'Arnold_Palmer',           # 2016 - Heart disease, golf legend
    'Jose_Fernandez',          # 2016 - Boat crash, MLB pitcher
    'Roy_Halladay',            # 2017 - Plane crash, MLB pitcher
    'Don_Shula',               # 2020 - Natural causes, NFL coach legend
    'Tommy_Lasorda',           # 2021 - Heart attack, Dodgers legend
    'Hank_Aaron',              # 2021 - Natural causes, baseball home run king
    'Bob_Dole',                # 2021 - Natural causes, WWII vet/politician
    'John_Madden',             # 2021 - Natural causes, NFL coach/broadcaster
    'Franco_Harris',           # 2022 - Natural causes, NFL legend
    'Pele',                    # 2022 - Cancer, soccer god
    'Gaylord_Perry',           # 2022 - Natural causes, MLB Hall of Famer

    # International Icons
    'Karl_Lagerfeld',          # 2019 - Natural causes, fashion designer
    'Niki_Lauda',              # 2019 - Kidney failure, F1 legend
    'Ennio_Morricone',         # 2020 - Complications, film composer
    'Nobby_Stiles',            # 2020 - Natural causes, 1966 World Cup winner
    'Sabine_Schmitz',          # 2021 - Cancer, Top Gear Nürburgring Queen
    'Jean-Paul_Belmondo',      # 2021 - Natural causes, French actor icon
    'Desmond_Tutu',            # 2021 - Cancer, South African archbishop
    'Mikhail_Gorbachev',       # 2022 - Natural causes, former Soviet leader
    'Jiang_Zemin',             # 2022 - Leukemia, former Chinese president
    'Pope_Benedict_XVI',       # 2022 - Natural causes, former Pope
    'Pelé',                    # 2022 - Cancer, soccer legend (duplicate but important)
    'Silvio_Berlusconi',       # 2023 - Leukemia, Italian PM/media mogul
    'Gina_Lollobrigida',       # 2023 - Natural causes, Italian actress icon

    # Reality TV / Internet Culture
    'Billy_Mays',              # 2009 - Heart disease, infomercial legend
    'August_Ames',             # 2017 - Suicide, adult film actress
    'Etika',                   # 2019 - Suicide, gaming YouTuber
    'Grant_Imahara',           # 2020 - Brain aneurysm, Mythbusters
    'Brent_Rivera',            # N/A - (skip if not applicable)
]

def get_wikidata_qid(page_title):
    """Get Wikidata QID from Wikipedia page title"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": page_title,
        "ppprop": "wikibase_item",
        "format": "json"
    }

    try:
        headers = {"User-Agent": "LegendCollector/1.0 (CSC466 Project)"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        if not pages:
            return None

        page_id = list(pages.keys())[0]
        if page_id == "-1":
            return None

        pageprops = pages[page_id].get("pageprops", {})
        return pageprops.get("wikibase_item")
    except Exception as e:
        print(f"  ⚠️  Could not get QID for {page_title}: {e}")
        return None

def get_person_data(qid):
    """Get person data from Wikidata"""
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?person ?personLabel ?dateOfDeath ?dateOfBirth
    WHERE {{
        wd:{qid} wdt:P31 wd:Q5.
        OPTIONAL {{ wd:{qid} wdt:P570 ?dateOfDeath. }}
        OPTIONAL {{ wd:{qid} wdt:P569 ?dateOfBirth. }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1
    """

    try:
        headers = {"Accept": "application/json", "User-Agent": "LegendCollector/1.0 (CSC466 Project)"}
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=15)
        response.raise_for_status()

        results = response.json().get("results", {}).get("bindings", [])
        if results:
            result = results[0]
            return {
                "name": result.get("personLabel", {}).get("value", ""),
                "date_of_death": result.get("dateOfDeath", {}).get("value", "").split("T")[0] if result.get("dateOfDeath") else None,
                "date_of_birth": result.get("dateOfBirth", {}).get("value", "").split("T")[0] if result.get("dateOfBirth") else None,
            }
        return None
    except Exception as e:
        print(f"  ⚠️  Could not get person data for {qid}: {e}")
        return None

def person_exists(conn, qid):
    """Check if person already exists in database"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM wikipedia_page WHERE wikidata_qid = ?", (qid,))
    count = cursor.fetchone()[0]
    return count > 0

def add_person_to_db(conn, page_title, qid, person_data):
    """Add person to database"""
    cursor = conn.cursor()

    try:
        # Extract birth year from date_of_birth
        birth_year = None
        if person_data.get("date_of_birth"):
            try:
                birth_year = int(person_data["date_of_birth"][:4])
            except:
                pass

        cursor.execute("""
            INSERT INTO wikipedia_page (
                wikidata_qid,
                page_title,
                name,
                date_of_death,
                birth_year
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            qid,
            page_title,
            person_data.get("name", page_title.replace("_", " ")),
            person_data.get("date_of_death"),
            birth_year
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"  ⚠️  Person already exists: {page_title}")
        return False
    except Exception as e:
        print(f"  ❌ Error inserting {page_title}: {e}")
        conn.rollback()
        return False

def main():
    print("=" * 80)
    print("CURATED LEGEND COLLECTION")
    print("Adding known legends and high-profile tragic deaths (2018-2023)")
    print("=" * 80)
    print()

    # Connect to database
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        print("Please run create_clout.py first")
        return

    conn = sqlite3.connect(DB_PATH)

    added = 0
    skipped = 0
    errors = 0

    print(f"Processing {len(KNOWN_LEGENDS)} curated legends...\n")

    for i, page_title in enumerate(KNOWN_LEGENDS, 1):
        print(f"[{i}/{len(KNOWN_LEGENDS)}] {page_title}")

        # Get QID
        qid = get_wikidata_qid(page_title)
        if not qid:
            print(f"  ❌ No QID found (page may not exist)")
            errors += 1
            time.sleep(1)
            continue

        # Check if already exists
        if person_exists(conn, qid):
            print(f"  ⏭️  Already in database")
            skipped += 1
            time.sleep(0.5)
            continue

        # Get person data
        person_data = get_person_data(qid)
        if not person_data:
            print(f"  ❌ Could not fetch person data")
            errors += 1
            time.sleep(1)
            continue

        # Check if they actually died (some might be alive)
        if not person_data.get("date_of_death"):
            print(f"  ⚠️  No date of death found (may still be alive)")
            errors += 1
            time.sleep(1)
            continue

        # Add to database
        if add_person_to_db(conn, page_title, qid, person_data):
            death_year = person_data.get("date_of_death", "")[:4]
            print(f"  ✅ Added: {person_data.get('name')} (died {death_year})")
            added += 1
        else:
            errors += 1

        # Rate limiting
        time.sleep(1.5)

    conn.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Added:   {added}")
    print(f"⏭️  Skipped: {skipped} (already in database)")
    print(f"❌ Errors:  {errors}")
    print(f"📊 Total:   {len(KNOWN_LEGENDS)}")
    print()
    print("Next steps:")
    print("1. Run: python enrich_wikidata.py")
    print("2. Run: python upgrade_features.py")
    print("3. Run: python create_balanced_dataset.py")
    print("4. Retrain models: cd ../final-report && python train_baselines.py")
    print()

if __name__ == "__main__":
    main()
