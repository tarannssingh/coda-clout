import sqlite3

con = sqlite3.connect("./wikipedia_clout.db")
cur = con.cursor()


def index_exists(index_name: str) -> bool:
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    )
    return cur.fetchone() is not None


def create_tables() -> None:
    """Bootstraps the minimal schema for early exploratory work."""
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wikipedia_page (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            page_title TEXT UNIQUE,
            date_of_death DATE,
            avg_views_pre_death_10d INTEGER,
            wikidata_qid TEXT,
            sitelinks INTEGER,
            birth_year INTEGER,
            award_count INTEGER,
            page_created DATE,
            page_len_bytes INTEGER,
            page_watchers INTEGER,
            edits_past_year INTEGER,
            num_editors INTEGER,
            cause_of_death TEXT
        );
        """
    )
    con.commit()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wikipedia_occupation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER,
            occupation TEXT,
            FOREIGN KEY(page_id) REFERENCES wikipedia_page(id) ON DELETE CASCADE,
            UNIQUE(page_id, occupation)
        );
        """
    )
    con.commit()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wikipedia_daily_clout (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER,
            views INTEGER,
            date DATE,
            day_since_death INTEGER,
            FOREIGN KEY(page_id) REFERENCES wikipedia_page(id) ON DELETE CASCADE,
            UNIQUE(page_id, date)
        );
        """
    )
    con.commit()


create_tables()

if not index_exists("idx_page_id"):
    cur.execute("CREATE INDEX idx_page_id ON wikipedia_daily_clout(page_id);")
if not index_exists("idx_date"):
    cur.execute("CREATE INDEX idx_date ON wikipedia_daily_clout(date);")
if not index_exists("idx_page_occ"):
    cur.execute("CREATE INDEX idx_page_occ ON wikipedia_occupation(page_id);")

con.commit()
con.close()
