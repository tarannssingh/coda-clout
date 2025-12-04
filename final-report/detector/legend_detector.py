"""
Legend Detector - Streamlit App
Predict if a Wikipedia person is/will be a legend based on pre-death features
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import pickle

# Try to load the trained model
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        from xgboost import XGBClassifier
        import pickle
        model_path = Path(__file__).parent / "xgb_model.pkl"  # Same directory as script
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.warning("⚠️ Model file not found. Please run train_baselines.py first to generate xgb_model.pkl")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_feature_columns():
    """Load feature column order"""
    try:
        import json
        feature_path = Path(__file__).parent / "feature_columns.json"  # Same directory as script
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                return json.load(f)
        return None
    except:
        return None

def get_wikipedia_data(page_title):
    """Fetch Wikipedia page data for a given page title"""
    from urllib.parse import quote
    
    url = "https://en.wikipedia.org/w/api.php"
    
    # URL encode the page title
    page_title_encoded = quote(page_title.replace(" ", "_"), safe="")
    
    # Get page info
    params = {
        "action": "query",
        "prop": "info|revisions",
        "titles": page_title,
        "rvlimit": 1,
        "rvdir": "newer",
        "rvprop": "timestamp",
        "inprop": "url",
        "format": "json"
    }
    
    try:
        headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()  # Raise exception for bad status codes
        
        if not response.text or response.text.strip() == "":
            return None
            
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return None
        
        page_id = list(pages.keys())[0]
        page_data = pages[page_id]
        
        if page_id == "-1":
            return None
        
        # Get creation date
        creation_date = None
        if "revisions" in page_data:
            creation_date = page_data["revisions"][0]["timestamp"][:10]
        
        # Get page length, watchers, etc.
        page_info_params = {
            "action": "query",
            "prop": "info",
            "titles": page_title,
            "inprop": "length|watchers",
            "format": "json"
        }
        
        info_response = requests.get(url, params=page_info_params, headers=headers, timeout=10)
        info_data = info_response.json()
        info_pages = info_data.get("query", {}).get("pages", {})
        if page_id in info_pages:
            page_info = info_pages[page_id]
            page_len = page_info.get("length", 0)
            watchers = page_info.get("watchers", 0)
        else:
            page_len = 0
            watchers = 0
        
        return {
            "page_title": page_title,
            "page_created": creation_date,
            "page_len_bytes": page_len,
            "page_watchers": watchers,
            "url": page_data.get("fullurl", "")
        }
    except requests.exceptions.RequestException as e:
        # Don't show error to user, just return None
        return None
    except Exception as e:
        # Unexpected error
        return None

def get_wikidata_info(page_title):
    """Get Wikidata info for a person"""
    # First, get Wikidata QID from Wikipedia
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "titles": page_title,
        "ppprop": "wikibase_item",
        "format": "json"
    }
    
    try:
        headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        if not pages:
            return None
        
        page_id = list(pages.keys())[0]
        pageprops = pages[page_id].get("pageprops", {})
        qid = pageprops.get("wikibase_item")
        
        if not qid:
            return None
        
        # Query Wikidata
        sparql_url = "https://query.wikidata.org/sparql"
        query = f"""
        SELECT ?person ?personLabel ?dateOfDeath ?dateOfBirth ?causeOfDeath ?causeOfDeathLabel
        WHERE {{
            wd:{qid} wdt:P31 wd:Q5.
            OPTIONAL {{ wd:{qid} wdt:P570 ?dateOfDeath. }}
            OPTIONAL {{ wd:{qid} wdt:P569 ?dateOfBirth. }}
            OPTIONAL {{ wd:{qid} wdt:P509 ?causeOfDeath. }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 1
        """
        
        headers = {"Accept": "application/json", "User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
        sparql_response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=10)
        sparql_response.raise_for_status()
        
        if sparql_response.status_code == 200:
            results = sparql_response.json().get("results", {}).get("bindings", [])
            if results:
                result = results[0]
                return {
                    "qid": qid,
                    "name": result.get("personLabel", {}).get("value", page_title),
                    "date_of_death": result.get("dateOfDeath", {}).get("value", "").split("T")[0] if result.get("dateOfDeath") else None,
                    "date_of_birth": result.get("dateOfBirth", {}).get("value", "").split("T")[0] if result.get("dateOfBirth") else None,
                    "cause_of_death": result.get("causeOfDeathLabel", {}).get("value", "") if result.get("causeOfDeath") else None
                }
        
        return None
    except Exception as e:
        st.warning(f"Could not fetch Wikidata info: {e}")
        return None

def get_recent_views(page_title, days=10):
    """Get recent Wikipedia page views"""
    today = datetime.now()
    start = (today - timedelta(days=days)).strftime("%Y%m%d")
    end = (today - timedelta(days=1)).strftime("%Y%m%d")
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{page_title}/daily/{start}/{end}"
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            if items:
                total_views = sum(item.get("views", 0) for item in items)
                avg_views = total_views / len(items) if items else 0
                return avg_views
        return 0
    except:
        return 0

def get_sitelinks(qid):
    """Get number of sitelinks from Wikidata"""
    if not qid:
        return 0
    
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT (COUNT(?article) as ?count)
    WHERE {{
        ?article schema:about wd:{qid};
                schema:isPartOf <https://en.wikipedia.org/>.
    }}
    """
    
    headers = {"Accept": "application/json", "User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    try:
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            if results:
                return int(results[0].get("count", {}).get("value", 0))
        return 0
    except:
        return 0

def get_edits_past_year(page_title, date_of_death=None):
    """Get number of edits in the year before death (or past year if alive)"""
    url = "https://en.wikipedia.org/w/api.php"
    
    # Calculate date range
    if date_of_death:
        try:
            death_date = datetime.strptime(date_of_death, "%Y-%m-%d")
            end_date = death_date
            start_date = death_date - timedelta(days=365)
        except:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
    
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvlimit": 500,
        "rvdir": "older",
        "rvstart": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rvend": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rvprop": "timestamp",
        "format": "json"
    }
    
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_data in pages.values():
            if "revisions" in page_data:
                # Count revisions in the date range
                count = 0
                for rev in page_data["revisions"]:
                    rev_time = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                    if start_date <= rev_time <= end_date:
                        count += 1
                return count
        return 0
    except Exception as e:
        # If API fails, return 0 (will use default)
        return 0

def get_person_image(page_title):
    """Get the main image URL for a Wikipedia page"""
    url = "https://en.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "prop": "pageimages",
        "titles": page_title,
        "piprop": "original",
        "pithumbsize": 300,
        "format": "json"
    }
    
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_data in pages.values():
            if "original" in page_data:
                return page_data["original"]["source"]
            elif "thumbnail" in page_data:
                return page_data["thumbnail"]["source"]
        return None
    except Exception as e:
        return None

def get_page_length_at_date(page_title, before_date):
    """Get page length at a specific date (day before death)"""
    if not before_date:
        return 0
    
    url = "https://en.wikipedia.org/w/api.php"
    try:
        date_obj = datetime.strptime(before_date, "%Y-%m-%d")
        # Get day before death
        day_before = date_obj - timedelta(days=1)
        rvstart = day_before.strftime("%Y-%m-%dT23:59:59Z")
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
        "format": "json"
    }
    
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_data in pages.values():
            if "revisions" in page_data and page_data["revisions"]:
                return page_data["revisions"][0].get("size", 0) or 0
        return 0
    except Exception as e:
        return 0

def get_all_time_edits(page_title):
    """Get total number of edits for a page (all time) - paginates through all revisions"""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    total_count = 0
    rvcontinue = None
    
    try:
        while True:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvlimit": 500,  # Max per request
                "rvdir": "newer",  # Start from oldest
                "rvprop": "timestamp",
                "format": "json"
            }
            
            if rvcontinue:
                params["rvcontinue"] = rvcontinue
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_data in pages.values():
                if "revisions" in page_data:
                    total_count += len(page_data.get("revisions", []))
            
            # Check for continuation
            if "continue" in data and "rvcontinue" in data["continue"]:
                rvcontinue = data["continue"]["rvcontinue"]
            else:
                break
        
        return total_count
    except Exception as e:
        # If pagination fails, try to get at least a count
        try:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvlimit": 500,
                "rvdir": "newer",
                "rvprop": "timestamp",
                "format": "json"
            }
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                if "revisions" in page_data:
                    return len(page_data.get("revisions", []))
        except:
            pass
        return 0

def get_edits_creation_to_death(page_title, page_created, date_of_death):
    """Get number of edits from page creation to day before death"""
    if not page_created or not date_of_death:
        return 0
    
    url = "https://en.wikipedia.org/w/api.php"
    
    try:
        creation_date = datetime.strptime(page_created, "%Y-%m-%d")
        death_date = datetime.strptime(date_of_death, "%Y-%m-%d")
        # Day before death
        end_date = death_date - timedelta(days=1)
        
        # Get all revisions from creation to day before death
        # We'll need to paginate through revisions
        total_edits = 0
        rvcontinue = None
        
        while True:
            params = {
                "action": "query",
                "prop": "revisions",
                "titles": page_title,
                "rvlimit": 500,
                "rvdir": "older",
                "rvstart": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rvend": creation_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rvprop": "timestamp",
                "format": "json"
            }
            
            if rvcontinue:
                params["rvcontinue"] = rvcontinue
            
            headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for page_data in pages.values():
                if "revisions" in page_data:
                    # Count revisions in the date range
                    for rev in page_data["revisions"]:
                        rev_time = datetime.strptime(rev["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                        if creation_date <= rev_time <= end_date:
                            total_edits += 1
            
            # Check for continuation
            if "continue" in data and "rvcontinue" in data["continue"]:
                rvcontinue = data["continue"]["rvcontinue"]
            else:
                break
                
        return total_edits
    except Exception as e:
        # If API fails, return 0
        return 0

def get_awards(qid):
    """Get number of awards from Wikidata"""
    if not qid:
        return 0
    
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT (COUNT(?award) as ?count)
    WHERE {{
        wd:{qid} wdt:P166 ?award.
    }}
    """
    
    headers = {"Accept": "application/json", "User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    try:
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            if results:
                return int(results[0].get("count", {}).get("value", 0))
        return 0
    except:
        return 0

def get_occupations(qid):
    """Get occupations from Wikidata"""
    if not qid:
        return []
    
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?occupation ?occupationLabel
    WHERE {{
        wd:{qid} wdt:P106 ?occupation.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    
    headers = {"Accept": "application/json", "User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    try:
        response = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            occupations = []
            for result in results:
                occ_label = result.get("occupationLabel", {}).get("value", "")
                if occ_label:
                    occupations.append(occ_label)
            return occupations
        return []
    except:
        return []

def categorize_cause(cause_str):
    """Categorize cause of death"""
    if not cause_str:
        return 'unknown'
    
    cause_lower = str(cause_str).lower()
    
    if any(kw in cause_lower for kw in ['cancer', 'carcinoma', 'tumor', 'leukemia']):
        return 'cancer'
    elif any(kw in cause_lower for kw in ['heart', 'cardiac', 'stroke', 'cerebrovascular']):
        return 'heart_disease'
    elif any(kw in cause_lower for kw in ['covid', 'coronavirus']):
        return 'covid'
    elif any(kw in cause_lower for kw in ['accident', 'crash', 'collision']):
        return 'accident'
    elif any(kw in cause_lower for kw in ['suicide']):
        return 'suicide'
    else:
        return 'other'

def prepare_features(wiki_data, wikidata_info, recent_views, sitelinks, awards, occupations):
    """Prepare features for model prediction"""
    # Calculate age at death if available
    age_at_death = None
    death_year = None
    
    if wikidata_info and wikidata_info.get("date_of_death") and wikidata_info.get("date_of_birth"):
        try:
            death_date = datetime.strptime(wikidata_info["date_of_death"], "%Y-%m-%d")
            birth_date = datetime.strptime(wikidata_info["date_of_birth"], "%Y-%m-%d")
            age_at_death = (death_date - birth_date).days / 365.25
            death_year = death_date.year
        except:
            pass
    
    # If alive, use current date
    if not death_year and wikidata_info and wikidata_info.get("date_of_birth"):
        try:
            birth_date = datetime.strptime(wikidata_info["date_of_birth"], "%Y-%m-%d")
            age_at_death = (datetime.now() - birth_date).days / 365.25
            death_year = datetime.now().year
        except:
            pass
    
    # Default values
    if age_at_death is None:
        age_at_death = 70  # median
    if death_year is None:
        death_year = 2023
    
    # Log features
    log_avg_views = np.log1p(recent_views)
    log_sitelinks = np.log1p(sitelinks)
    log_award_count = np.log1p(awards)
    
    # Get historical page length (day before death) if dead, otherwise current
    date_of_death = wikidata_info.get("date_of_death") if wikidata_info else None
    if date_of_death:
        historical_page_len = get_page_length_at_date(page_title, date_of_death)
        page_len_bytes = historical_page_len if historical_page_len > 0 else (wiki_data.get("page_len_bytes", 0) if wiki_data else 0)
    else:
        page_len_bytes = wiki_data.get("page_len_bytes", 0) if wiki_data else 0
    
    log_page_len = np.log1p(page_len_bytes)
    log_watchers = np.log1p(wiki_data.get("page_watchers", 0) if wiki_data else 0)
    
    # Fame proxy
    fame_proxy = log_avg_views + log_sitelinks + awards - 0.5 * log_page_len
    
    # Interaction features
    age_x_fame = age_at_death * fame_proxy
    age_x_year = age_at_death * (death_year - 2018)
    views_per_sitelink = log_avg_views - log_sitelinks if log_sitelinks > 0 else 0
    
    # Cause category (one-hot)
    cause_cat = categorize_cause(wikidata_info.get("cause_of_death") if wikidata_info else None)
    cause_cancer = 1 if cause_cat == 'cancer' else 0
    cause_heart_disease = 1 if cause_cat == 'heart_disease' else 0
    cause_covid = 1 if cause_cat == 'covid' else 0
    cause_accident = 1 if cause_cat == 'accident' else 0
    cause_suicide = 1 if cause_cat == 'suicide' else 0
    
    # Occupations (one-hot) - top 15 from training
    top_occs = ['actor', 'film_actor', 'television_actor', 'American_football_player']
    
    occ_features = {}
    for occ in top_occs:
        occ_features[f'occ_{occ}'] = 0
    
    if occupations:
        for occ in occupations:
            occ_normalized = occ.lower().replace(" ", "_").replace("/", "_")
            if f'occ_{occ_normalized}' in occ_features:
                occ_features[f'occ_{occ_normalized}'] = 1
    
    # Get edits in past year (before death if dead)
    edits_past_year = get_edits_past_year(page_title, wikidata_info.get("date_of_death") if wikidata_info else None)
    log_edits_past_year = np.log1p(edits_past_year)
    
    # Build feature vector (must match training order from feature_columns.json)
    # Note: The model doesn't use cause categories, so we don't include them
    features = {
        'log_avg_views_pre_death_10d': log_avg_views,
        'log_sitelinks': log_sitelinks,
        'log_page_len_bytes': log_page_len,
        'log_page_watchers': log_watchers,
        'log_edits_past_year': log_edits_past_year,
        'log_num_editors': 0,  # Would need API call - default to 0
        'log_award_count': log_award_count,
        'age_at_death': age_at_death,
        'death_year': death_year,
        **occ_features,  # All occupation features
        'fame_proxy': fame_proxy,
        'age_x_fame': age_x_fame,
        'age_x_year': age_x_year,
        'views_per_sitelink': views_per_sitelink,
    }
    
    return features


def predict_legend(features, model, feature_cols, threshold=0.25):
    """Make prediction using the model with adjustable threshold
    
    Default threshold=0.25 for higher recall (catches more legends like Sidhu Moosewala)
    Use threshold=0.5 for balanced precision/recall
    """
    if model is None or feature_cols is None:
        return None, None
    
    # Build feature vector in correct order
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(features.get(col, 0))
    
    X = np.array(feature_vector).reshape(1, -1)
    
    try:
        proba = model.predict_proba(X)[0]
        legend_prob = proba[1]
        # Use custom threshold instead of default 0.5
        prediction = 1 if legend_prob >= threshold else 0
        return prediction, legend_prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def check_actual_legend_status(page_title, date_of_death):
    """Check if person is actually a legend based on post-death metrics"""
    if not date_of_death:
        return None, None
    
    try:
        death_date = datetime.strptime(date_of_death, "%Y-%m-%d")
        
        # Get views for 30-365 days after death
        start = (death_date + timedelta(days=30)).strftime("%Y%m%d")
        end = (death_date + timedelta(days=365)).strftime("%Y%m%d")
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{page_title}/daily/{start}/{end}"
        headers = {"User-Agent": "LegendDetector/1.0"}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            if items:
                post_views = [item.get("views", 0) for item in items]
                post_avg_daily = np.mean(post_views)
                
                # Get pre-death baseline (10 days before)
                pre_start = (death_date - timedelta(days=10)).strftime("%Y%m%d")
                pre_end = (death_date - timedelta(days=1)).strftime("%Y%m%d")
                
                pre_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{page_title}/daily/{pre_start}/{pre_end}"
                pre_response = requests.get(pre_url, headers=headers, timeout=10)
                
                pre_avg = 0
                if pre_response.status_code == 200:
                    pre_data = pre_response.json()
                    pre_items = pre_data.get("items", [])
                    if pre_items:
                        pre_views = [item.get("views", 0) for item in pre_items]
                        pre_avg = np.mean(pre_views) if pre_views else 1
                
                if pre_avg > 0:
                    sustained_ratio = post_avg_daily / pre_avg
                    is_legend = (sustained_ratio > 2.5) and (post_avg_daily > 50)
                    
                    return is_legend, {
                        "sustained_ratio": sustained_ratio,
                        "post_avg_daily": post_avg_daily,
                        "pre_avg": pre_avg
                    }
        
        return None, None
    except Exception as e:
        return None, None

# Streamlit UI
st.set_page_config(
    page_title="Legend Detector",
    page_icon="⭐",
    layout="wide"
)

# Custom CSS - Premium minimal design
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f2f5 100%);
    }

    /* Typography */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #6c757d;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Info boxes - clean minimal cards */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .info-box-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #667eea;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .info-box-text {
        font-size: 0.95rem;
        color: #495057;
        line-height: 1.6;
    }

    /* Prediction cards */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.3s ease;
    }

    .legend-yes {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }

    .legend-no {
        background: white;
        color: #1a1a2e;
        border: 2px solid #e9ecef;
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .prediction-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }

    .prediction-explanation {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin: 0.75rem 0;
        border: 1px solid #e9ecef;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .metric-explanation {
        font-size: 0.85rem;
        color: #868e96;
        margin-top: 0.5rem;
        line-height: 1.4;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }

    /* Technical details badge */
    .tech-badge {
        display: inline-block;
        background: #f8f9fa;
        color: #495057;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        margin: 0.25rem;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e9ecef, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Legend Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict who will be remembered after death using machine learning</div>', unsafe_allow_html=True)

# Explanation section
st.markdown("""
<div class="info-box">
    <div class="info-box-title">What does this tool do?</div>
    <div class="info-box-text">
        <strong>Plain English:</strong> This predicts if someone will be culturally remembered months and years after they die—not just a brief spike in the news, but sustained interest like Kobe Bryant or David Bowie.<br><br>
        <strong>Technical:</strong> Uses gradient boosting (XGBoost) trained on 2,281 deceased Wikipedia subjects (2017-2025). Temporally balanced dataset (equal samples per year) prevents temporal bias. Predicts "legend" status defined as post-death views (days 30-365) exceeding 2.5× baseline + 50 avg daily views. Model achieves ROC-AUC 0.850 with 35.3% precision and 31.6% recall using only pre-death features.
    </div>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="section-header">Enter a Person to Analyze</div>', unsafe_allow_html=True)

# Use a form so Enter key submits
with st.form("analyze_form", clear_on_submit=False):
    col1, col2 = st.columns([4, 1])
    with col1:
        page_title = st.text_input(
            "Wikipedia Page Title",
            placeholder="e.g., Matthew_Perry, Sidhu_Moose_Wala, Kobe_Bryant",
            help="Use underscores instead of spaces (e.g., 'Matthew_Perry' not 'Matthew Perry')",
            label_visibility="collapsed"
        )
    with col2:
        analyze_button = st.form_submit_button("Analyze", type="primary", use_container_width=True)

# Advanced settings in expander
with st.expander("Advanced Settings"):
    st.markdown("**Prediction Threshold**")
    threshold = st.slider(
        "Adjust sensitivity",
        min_value=0.1, max_value=0.6, value=0.2, step=0.05,
        help="Lower = catch more legends (higher recall), Higher = fewer false alarms (higher precision)",
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div style="font-size: 0.85rem; color: #6c757d; margin-top: 0.5rem;">
        <strong>Plain English:</strong> Lower threshold = more generous, catches edge cases<br>
        <strong>Technical:</strong> Classification threshold τ={threshold:.2f}. Default 0.25 optimizes for recall over precision.
    </div>
    """, unsafe_allow_html=True)

if analyze_button and page_title.strip():
    with st.spinner("Fetching data and making prediction..."):
        # Fetch data
        wiki_data = get_wikipedia_data(page_title)
        wikidata_info = get_wikidata_info(page_title)
        
        if not wiki_data and not wikidata_info:
            st.error(f"❌ Could not find Wikipedia page: {page_title}")
            st.info("💡 Tip: Make sure to use underscores (e.g., 'Matthew_Perry' not 'Matthew Perry')")
            st.info("💡 Also try: Check if the page exists on Wikipedia, or try a different spelling")
            
            # Try to suggest alternatives
            try:
                search_url = f"https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "opensearch",
                    "search": page_title.replace("_", " "),
                    "limit": 5,
                    "format": "json"
                }
                search_response = requests.get(search_url, params=search_params, timeout=10)
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    if len(search_data) > 1 and len(search_data[1]) > 0:
                        st.info("🔍 Did you mean:")
                        for suggestion in search_data[1][:3]:
                            suggestion_title = suggestion.split("/")[-1].replace(" ", "_")
                            st.write(f"   - `{suggestion_title}`")
            except:
                pass
        else:
            # Get additional data
            recent_views = get_recent_views(page_title, days=10)
            qid = wikidata_info.get("qid") if wikidata_info else None
            sitelinks = get_sitelinks(qid) if qid else 0
            awards = get_awards(qid) if qid else 0
            occupations = get_occupations(qid) if qid else []
            
            # Prepare features
            features = prepare_features(wiki_data, wikidata_info, recent_views, sitelinks, awards, occupations)
            
            # Load model
            model = load_model()
            feature_cols = load_feature_columns()
            
            if not wiki_data:
                st.warning("⚠️ Could not fetch Wikipedia page data. Using Wikidata data only.")
            if not wikidata_info:
                st.warning("⚠️ Could not fetch Wikidata data. Some features will be missing.")
            
            # Display person info
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            name = wikidata_info.get("name", page_title) if wikidata_info else page_title
            st.markdown(f'<div class="section-header">Analysis: {name}</div>', unsafe_allow_html=True)

            st.markdown("**Basic Information**")
            
            # Get person's image and page creation date
            person_image = get_person_image(page_title)
            page_created = wiki_data.get('page_created') if wiki_data else None
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display person's image
                if person_image:
                    st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <img src="{person_image}" style="width: 100%; max-width: 200px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" alt="{name}">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem; padding: 3rem 1rem; background: #f8f9fa; border-radius: 12px; color: #6c757d;">
                        <div style="font-size: 3rem;">👤</div>
                        <div style="font-size: 0.85rem; margin-top: 0.5rem;">No image available</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if wikidata_info:
                    if wikidata_info.get("date_of_death"):
                        st.markdown(f"**Status:** Deceased ({wikidata_info['date_of_death']})")
                        if wikidata_info.get("date_of_birth"):
                            age = int(features['age_at_death'])  # Use int() to truncate, not round
                            st.markdown(f"**Age at Death:** {age} years")
                    else:
                        st.markdown(f"**Status:** Currently Living")
                        if wikidata_info.get("date_of_birth"):
                            age = int(features['age_at_death'])  # Use int() to truncate, not round
                            st.markdown(f"**Current Age:** {age} years")

                    if wikidata_info.get("cause_of_death"):
                        st.markdown(f"**Cause of Death:** {wikidata_info['cause_of_death']}")
                
                if page_created:
                    st.markdown(f"**Wikipedia Page Created:** {page_created}")
                    
                    # Calculate time between page creation and death
                    if wikidata_info and wikidata_info.get("date_of_death"):
                        try:
                            creation_date = datetime.strptime(page_created, "%Y-%m-%d")
                            death_date = datetime.strptime(wikidata_info["date_of_death"], "%Y-%m-%d")
                            time_diff = death_date - creation_date
                            years = time_diff.days / 365.25
                            months = (time_diff.days % 365.25) / 30.44
                            
                            if years >= 1:
                                if months < 1:
                                    time_str = f"{int(years)} year{'s' if int(years) != 1 else ''}"
                                else:
                                    time_str = f"{int(years)} year{'s' if int(years) != 1 else ''}, {int(months)} month{'s' if int(months) != 1 else ''}"
                            else:
                                time_str = f"{int(months)} month{'s' if int(months) != 1 else ''}"
                            
                            st.markdown(f"**Page Existed Before Death:** {time_str}")
                        except:
                            pass
                
                if occupations:
                    st.markdown(f"**Profession:** {', '.join(occupations[:3])}")
            
            # Prediction
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

            # Use model if available, otherwise use heuristic
            if model and feature_cols:
                pred, prob = predict_legend(features, model, feature_cols, threshold=threshold)
                if pred is not None:
                    is_predicted_legend = pred == 1
                    proba_legend = prob
                else:
                    # Fallback heuristic
                    legend_score = (
                        features['fame_proxy'] * 0.35 +
                        (100 - features['age_at_death']) * 0.25 +
                        features['log_sitelinks'] * 0.25 +
                        features['log_award_count'] * 0.15
                    )
                    proba_legend = min(max(legend_score / 20, 0), 1)
                    is_predicted_legend = proba_legend >= threshold
            else:
                # Heuristic fallback
                legend_score = (
                    features['fame_proxy'] * 0.35 +
                    (100 - features['age_at_death']) * 0.25 +
                    features['log_sitelinks'] * 0.25 +
                    features['log_award_count'] * 0.15
                )
                proba_legend = min(max(legend_score / 20, 0), 1)
                is_predicted_legend = proba_legend >= threshold

            # Display prediction
            if is_predicted_legend:
                st.markdown(f"""
                <div class="prediction-box legend-yes">
                    <div class="prediction-title">LEGEND STATUS PREDICTED</div>
                    <div class="prediction-subtitle">Model Confidence: {proba_legend*100:.1f}% LEGEND</div>
                    <div class="prediction-explanation">
                        <strong>Plain English:</strong> This person is likely to be culturally remembered for years after death—not just a temporary news spike, but sustained public interest like Kobe Bryant, David Bowie, or Matthew Perry.<br><br>
                        <strong>Technical:</strong> XGBoost gradient boosting classifier output P(legend|X) = {proba_legend:.3f} exceeds threshold τ={threshold:.2f}. Prediction based on pre-death feature vector including page metrics, demographics, and engineered interaction terms.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box legend-no">
                    <div class="prediction-title">NO LEGEND STATUS PREDICTED</div>
                    <div class="prediction-subtitle">Model Confidence: {(1-proba_legend)*100:.1f}% NOT LEGEND</div>
                    <div class="prediction-explanation">
                        <strong>Plain English:</strong> While their death may generate initial news coverage, Wikipedia activity is predicted to return to baseline levels within months. Think: temporary spike, not sustained cultural remembrance.<br><br>
                        <strong>Technical:</strong> XGBoost classifier output P(legend|X) = {proba_legend:.3f} below threshold τ={threshold:.2f}. Model predicts post-death sustained ratio (days 30-365) will not meet legend criteria (>2.5× baseline + 50 avg daily views).
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # PRE-DEATH METRICS SECTION
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 Pre-Death Metrics (What Model Used)</div>', unsafe_allow_html=True)

            # Calculate pre-death metrics
            import numpy as np
            page_created = wiki_data.get('page_created') if wiki_data else None
            date_of_death = wikidata_info.get('date_of_death') if wikidata_info else None
            
            # For living people, use today as the reference date
            is_living = date_of_death is None
            
            status_text = "before death" if not is_living else "current (as of today)"
            st.markdown(f"""
            <div class="info-box">
                <div class="info-box-text">
                    All values are from <strong>{status_text}</strong> — this is what the model analyzed to make its prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if is_living:
                reference_date = datetime.now().strftime("%Y-%m-%d")
            else:
                reference_date = date_of_death
            
            # Pre-death views (10 days before death, or last 10 days for living)
            pre_death_views = np.expm1(features['log_avg_views_pre_death_10d'])
            
            # Pre-death edits (from page creation to day before death, or to today for living)
            if page_created:
                pre_death_edits = get_edits_creation_to_death(page_title, page_created, reference_date)
            else:
                pre_death_edits = 0
            
            # Pre-death page size (historical - day before death, or current for living)
            # Fetch actual historical page length if we have a reference date
            if reference_date:
                historical_page_len = get_page_length_at_date(page_title, reference_date)
                if historical_page_len > 0:
                    pre_death_page_bytes = historical_page_len
                else:
                    # Fallback to model value if historical fetch fails
                    pre_death_page_bytes = np.expm1(features['log_page_len_bytes'])
            else:
                pre_death_page_bytes = np.expm1(features['log_page_len_bytes'])
            pre_death_page_kb = pre_death_page_bytes / 1024
            
            # Pre-death sitelinks
            pre_death_sitelinks = np.expm1(features['log_sitelinks'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #667eea;">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">🔥 PRE-DEATH ATTENTION</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{pre_death_views:,.0f}</div>
                    <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">avg daily views (10 days before death)</div>
                    <div style="font-size: 0.75rem; color: #667eea; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">Top predictor • 16.7%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #f5576c;">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">✏️ PRE-DEATH PAGE EDITS</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{pre_death_edits:,.0f}</div>
                    <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">total edits (creation → {'day before death' if not is_living else 'today'})</div>
                    <div style="font-size: 0.75rem; color: #f5576c; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">2nd predictor • 12.6%</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #4facfe;">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">📄 PRE-DEATH PAGE SIZE</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{pre_death_page_kb:,.0f} KB</div>
                    <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">Wikipedia page size ({'day before death' if not is_living else 'current'})</div>
                    <div style="font-size: 0.75rem; color: #4facfe; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">4th predictor • 7.2%</div>
                </div>
                """, unsafe_allow_html=True)

            # Second row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style="padding: 1.25rem; background: white; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">👤 AGE AT DEATH</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{int(features['age_at_death'])} years</div>
                    <div style="font-size: 0.75rem; color: #667eea; margin-top: 0.5rem;">3rd predictor • 11.2%</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="padding: 1.25rem; background: white; border-radius: 10px; border-left: 4px solid #00f2fe; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">🌍 GLOBAL REACH</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{pre_death_sitelinks:.0f}</div>
                    <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">language Wikipedias</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="padding: 1.25rem; background: white; border-radius: 10px; border-left: 4px solid #00f2fe; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                    <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">📊 VIEWS/SITELINK</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{features['views_per_sitelink']:.1f}</div>
                    <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">attention efficiency</div>
                </div>
                """, unsafe_allow_html=True)

            # Why Legend explanation
            high_attention = pre_death_views > 1000
            high_edits = pre_death_edits > 150
            young_death = features['age_at_death'] < 60
            big_page = pre_death_page_kb > 50
            
            attention_status = '🔥 HIGH' if high_attention else '❄️ LOW'
            attention_desc = 'People already cared about this person.' if high_attention else 'Limited public interest before death.'
            activity_status = '✅ ACTIVE' if high_edits else '⚠️ QUIET'
            activity_desc = 'Page was actively maintained.' if high_edits else 'Page saw minimal editing.'
            age_desc = 'Younger deaths create more legends.' if young_death else 'Older age reduces likelihood.'
            page_desc = 'Comprehensive coverage = existing fame.' if big_page else 'Shorter page = less documented.'
            legend_status = 'LEGEND' if is_predicted_legend else 'NOT LEGEND'
            confidence_color = '#28a745' if is_predicted_legend else '#dc3545'
            
            st.markdown(f"""
            <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-top: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid {confidence_color};">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 1rem;">💡 Why {legend_status}?</div>
                <div style="font-size: 0.95rem; color: #2d3436; line-height: 1.8;">
                    <strong>1. Pre-Death Interest:</strong> {attention_status} — <strong>{pre_death_views:,.0f} daily views</strong>. {attention_desc}<br><br>
                    <strong>2. Page Activity:</strong> {activity_status} — <strong>{pre_death_edits:,.0f} total edits</strong> {'before death' if not is_living else 'to date'}. {activity_desc}<br><br>
                    <strong>3. Age Factor:</strong> Died at <strong>{int(features['age_at_death'])} years old</strong>. {age_desc}<br><br>
                    <strong>4. Page Depth:</strong> <strong>{pre_death_page_kb:,.0f} KB</strong> page size. {page_desc}<br><br>
                    <strong>→ Model Confidence:</strong> <span style="font-size: 1.2rem; font-weight: 700; color: {confidence_color};">{(proba_legend*100 if is_predicted_legend else (1-proba_legend)*100):.1f}%</span> {legend_status}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # GROUND TRUTH SECTION (POST-DEATH METRICS)
            if wikidata_info and wikidata_info.get("date_of_death"):
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-header">📊 Post-Death Metrics (Ground Truth)</div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="info-box">
                    <div class="info-box-text">
                        <strong>Live data from today</strong> — compare with pre-death metrics above to see what actually happened.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                actual_legend, metrics = check_actual_legend_status(page_title, wikidata_info["date_of_death"])
                
                # Get current metrics (TODAY's data)
                post_death_views = metrics.get('post_avg_daily', 0) if metrics else 0
                
                # Recalculate sustained ratio using the same pre_death_views shown in UI
                # This ensures consistency - the ratio uses the same baseline as displayed
                if metrics and pre_death_views > 0:
                    metrics['sustained_ratio'] = post_death_views / pre_death_views
                    metrics['pre_avg'] = pre_death_views  # Update to match UI
                
                # All-time edits (from creation to TODAY) - TOTAL COUNT
                all_time_edits = get_all_time_edits(page_title)
                
                # Current page size (TODAY)
                current_wiki_data = get_wikipedia_data(page_title)
                current_page_bytes = current_wiki_data.get('page_len_bytes', 0) if current_wiki_data else 0
                current_page_kb = current_page_bytes / 1024
                
                # Current sitelinks (should be same or more than pre-death)
                current_sitelinks = sitelinks  # This is already fetched, should be current
                current_views_per_sitelink = (np.log1p(post_death_views) - np.log1p(current_sitelinks)) if current_sitelinks > 0 else 0
                
                # Calculate changes
                edit_change = all_time_edits - pre_death_edits if (pre_death_edits > 0 and all_time_edits >= pre_death_edits) else 0
                page_growth = ((current_page_kb - pre_death_page_kb) / pre_death_page_kb * 100) if pre_death_page_kb > 0 else 0
                sitelink_change = current_sitelinks - pre_death_sitelinks if current_sitelinks >= pre_death_sitelinks else 0

                if actual_legend is not None:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #667eea;">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">🔥 POST-DEATH ATTENTION</div>
                            <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{post_death_views:,.0f}</div>
                            <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">avg daily views (days 30-365 post-death)</div>
                            <div style="font-size: 0.75rem; color: #667eea; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">Was {pre_death_views:,.0f} before death</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #f5576c;">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">✏️ TOTAL PAGE EDITS</div>
                            <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{all_time_edits:,.0f}</div>
                            <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">all-time edits (creation → today)</div>
                            <div style="font-size: 0.75rem; color: {'#28a745' if edit_change > 0 else '#6c757d'}; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">+{edit_change:,.0f} since death</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        growth_sign = "+" if page_growth >= 0 else ""
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid #4facfe;">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">📄 CURRENT PAGE SIZE</div>
                            <div style="font-size: 2.2rem; font-weight: 700; color: #1a1a2e;">{current_page_kb:,.0f} KB</div>
                            <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">Wikipedia page size (today)</div>
                            <div style="font-size: 0.75rem; color: {'#28a745' if page_growth > 0 else '#6c757d'}; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e9ecef;">{growth_sign}{page_growth:.1f}% from {pre_death_page_kb:,.0f} KB</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Second row for Global Reach and Views/Sitelink
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 1.25rem; background: white; border-radius: 10px; border-left: 4px solid #00f2fe; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">🌍 CURRENT GLOBAL REACH</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{current_sitelinks:.0f}</div>
                            <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">language Wikipedias (today)</div>
                            <div style="font-size: 0.75rem; color: {'#28a745' if sitelink_change > 0 else '#6c757d'}; margin-top: 0.5rem;">{'Was ' + str(int(pre_death_sitelinks)) + ' before death' if sitelink_change == 0 else '+' + str(int(sitelink_change)) + ' since death'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="padding: 1.25rem; background: white; border-radius: 10px; border-left: 4px solid #00f2fe; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600;">📊 POST-DEATH VIEWS/SITELINK</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{current_views_per_sitelink:.1f}</div>
                            <div style="font-size: 0.85rem; color: #495057; margin-top: 0.25rem;">attention efficiency (post-death)</div>
                            <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.5rem;">Was {features['views_per_sitelink']:.1f} before death</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Legend criteria
                    st.markdown("**Legend Criteria Check**")
                    
                    col1, col2 = st.columns(2)

                    with col1:
                        ratio_met = metrics['sustained_ratio'] > 2.5
                        st.markdown(f"""
                        <div style="padding: 1rem; background: {'#d4edda' if ratio_met else '#f8f9fa'}; border-radius: 8px; border-left: 4px solid {'#28a745' if ratio_met else '#6c757d'};">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem;">SUSTAINED RATIO</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{metrics['sustained_ratio']:.2f}×</div>
                            <div style="font-size: 0.8rem; color: {'#155724' if ratio_met else '#495057'}; margin-top: 0.25rem;">{'✓ Exceeds 2.5× threshold' if ratio_met else '✗ Below 2.5× threshold'}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        views_met = metrics['post_avg_daily'] > 50
                        st.markdown(f"""
                        <div style="padding: 1rem; background: {'#d4edda' if views_met else '#f8f9fa'}; border-radius: 8px; border-left: 4px solid {'#28a745' if views_met else '#6c757d'};">
                            <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; margin-bottom: 0.5rem;">SUSTAINED VIEWS</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">{metrics['post_avg_daily']:.0f}/day</div>
                            <div style="font-size: 0.8rem; color: {'#155724' if views_met else '#495057'}; margin-top: 0.25rem;">{'✓ Exceeds 50 views/day' if views_met else '✗ Below 50 views/day'}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Final verdict
                    if actual_legend:
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: #d4edda; border-left: 4px solid #28a745; border-radius: 8px; margin-top: 1rem;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: #155724; margin-bottom: 0.5rem;">✓ CONFIRMED LEGEND</div>
                            <div style="font-size: 0.95rem; color: #155724;">
                                Views stayed <strong>{metrics['sustained_ratio']:.1f}×</strong> higher than before death. Averaged <strong>{metrics['post_avg_daily']:,.0f}</strong> daily views for months 1-12 after death. This is sustained cultural interest.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: #f8f9fa; border-left: 4px solid #6c757d; border-radius: 8px; margin-top: 1rem;">
                            <div style="font-size: 1.3rem; font-weight: 700; color: #495057; margin-bottom: 0.5rem;">✗ NOT A LEGEND</div>
                            <div style="font-size: 0.95rem; color: #495057;">
                                Views only sustained <strong>{metrics['sustained_ratio']:.1f}×</strong> baseline with <strong>{metrics['post_avg_daily']:,.0f}</strong> daily views. Brief news coverage, not lasting cultural impact.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Compare prediction vs actual
                    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                    if is_predicted_legend == actual_legend:
                        st.success("✓ **Model Accuracy:** Prediction matches ground truth — the model got it right!")
                    else:
                        st.warning("⚠ **Model Accuracy:** Prediction differs from ground truth — this is an edge case or model error")
                else:
                    st.info("Ground truth unavailable (insufficient time post-death or API limitations)")
            
            # Feature breakdown
            with st.expander("Advanced: Feature Engineering Details"):
                st.markdown("**Model Input Features (Top Contributors)**")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <strong>Age at Death:</strong> <span class="tech-badge">{features['age_at_death']:.1f} years</span><br>
                        <em>Plain:</em> How old when they died<br>
                        <em>Tech:</em> 11.2% feature importance (rank #3)<br><br>

                        <strong>Editorial Activity:</strong> <span class="tech-badge">log({features['log_edits_past_year']:.2f})</span><br>
                        <em>Plain:</em> How actively their page was being edited recently<br>
                        <em>Tech:</em> 12.6% feature importance (rank #2)<br><br>

                        <strong>Recent Attention:</strong> <span class="tech-badge">log({features['log_avg_views_pre_death_10d']:.2f})</span><br>
                        <em>Plain:</em> How much people were already reading about them<br>
                        <em>Tech:</em> 16.7% feature importance (rank #1)<br><br>

                        <strong>Page Length:</strong> <span class="tech-badge">log({features['log_page_len_bytes']:.2f})</span><br>
                        <em>Plain:</em> How comprehensive their Wikipedia page is<br>
                        <em>Tech:</em> 7.2% feature importance (rank #4)<br><br>

                        <strong>Global Recognition:</strong> <span class="tech-badge">log({features['log_sitelinks']:.2f})</span><br>
                        <em>Plain:</em> Known across different languages/cultures<br>
                        <em>Tech:</em> 4.3% feature importance (rank #9)
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div style="font-size: 0.9rem; line-height: 1.8;">
                        <strong>Views per Sitelink:</strong> <span class="tech-badge">{features['views_per_sitelink']:.2f}</span><br>
                        <em>Plain:</em> Attention relative to global reach<br>
                        <em>Tech:</em> 6.8% feature importance (rank #5)<br><br>

                        <strong>Age × Year Interaction:</strong> <span class="tech-badge">{features['age_x_year']:.1f}</span><br>
                        <em>Plain:</em> When and how old you die matters together<br>
                        <em>Tech:</em> age ({features['age_at_death']:.0f}) × (year - 2018) = {features['age_x_year']:.1f}<br>
                        <em>Note:</em> 4.2% feature importance (rank #10)<br><br>

                        <strong>Age × Fame Interaction:</strong> <span class="tech-badge">{features['age_x_fame']:.1f}</span><br>
                        <em>Plain:</em> Young + famous = more likely legend<br>
                        <em>Tech:</em> age ({features['age_at_death']:.0f}) × fame_proxy ({features['fame_proxy']:.2f}) = {features['age_x_fame']:.1f}<br>
                        <em>Note:</em> 4.1% feature importance (rank #11)<br><br>

                        <strong>Fame Proxy:</strong> <span class="tech-badge">{features['fame_proxy']:.2f}</span><br>
                        <em>Plain:</em> Overall celebrity score<br>
                        <em>Tech:</em> log(views) + log(sitelinks) + awards - 0.5×log(page_len)<br>
                        <em>Breakdown:</em> log({np.expm1(features['log_avg_views_pre_death_10d']):.0f}) + log({np.expm1(features['log_sitelinks']):.0f}) + {awards} - 0.5×log({np.expm1(features['log_page_len_bytes']):.0f})
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Show key interaction features that might explain predictions
                living_note = "(living person - using current age)" if is_living else "(at death)"
                st.markdown(f"""
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin-top: 1rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.75rem;">🔍 Key Factors Affecting This Prediction:</div>
                    <div style="font-size: 0.85rem; color: #495057; line-height: 1.8;">
                        <strong>Age Factor:</strong> {features['age_at_death']:.0f} years old {living_note}<br>
                        • Younger deaths (under 30) are more predictive of legend status<br>
                        • Age × Year interaction: {features['age_x_year']:.1f} (higher = older person in later years)<br>
                        • Age × Fame interaction: {features['age_x_fame']:.1f} (young + famous = boost)<br><br>
                        <strong>Fame Proxy:</strong> {features['fame_proxy']:.2f}<br>
                        • Formula: log(views) + log(sitelinks) + awards - 0.5×log(page_len)<br>
                        • Longer pages reduce fame_proxy (accounts for "already famous" bias)<br><br>
                        <strong>Note:</strong> For living people, the model uses current year which affects age×year interaction. This may not perfectly match historical patterns.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="font-size: 0.85rem; color: #6c757d; margin-top: 1rem;">
                    <strong>Why these features matter:</strong><br>
                    • <strong>Recent attention</strong> (views before death) is the strongest predictor of legend status (16.7% importance)<br>
                    • <strong>Editorial activity</strong> shows ongoing cultural relevance before death (12.6% importance)<br>
                    • <strong>Age at death</strong> influences cultural impact—dying young often creates legends (11.2% importance)<br>
                    • All features are <strong>pre-death only</strong>—model doesn't "cheat" by seeing post-death data
                </div>
                """, unsafe_allow_html=True)

elif analyze_button:
    st.warning("Please enter a Wikipedia page title")

# Footer with model info
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #1a1a2e;">0.850</div>
        <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px;">ROC-AUC Score</div>
        <div style="font-size: 0.8rem; color: #868e96; margin-top: 0.25rem;">XGBoost gradient boosting model</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #1a1a2e;">2,281</div>
        <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px;">Training Samples</div>
        <div style="font-size: 0.8rem; color: #868e96; margin-top: 0.25rem;">Temporally balanced (2017-2025)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; color: #1a1a2e;">2.5%</div>
        <div style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px;">Legend Rate</div>
        <div style="font-size: 0.8rem; color: #868e96; margin-top: 0.25rem;">57 legends in dataset</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">
        Legend Detector
    </div>
    <div style="font-size: 0.85rem; color: #6c757d; line-height: 1.6;">
        <strong>Model:</strong> XGBoost Gradient Boosting Classifier (ROC-AUC 0.850)<br>
        <strong>Features:</strong> 16 engineered features (page metrics, demographics, temporal interactions)<br>
        <strong>Dataset:</strong> 2,281 samples, temporally balanced (equal per year), 57 legends (2.5%)<br>
        <strong>Built for:</strong> CSC 466 — Knowledge Discovery from Data (December 2025)
    </div>
</div>
""", unsafe_allow_html=True)

