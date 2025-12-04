"""
Generate view graphs for specific people (Pop Smoke and Jimmy Carter)
Shows Wikipedia pageviews over time, especially around death
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

def get_wikipedia_views(page_title, start_date, end_date):
    """Fetch Wikipedia pageviews for a date range"""
    url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{}/daily/{}/{}"
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    full_url = url.format(page_title, start_str, end_str)
    headers = {"User-Agent": "LegendDetector/1.0 (https://github.com/yourusername/coda-clout)"}
    
    try:
        response = requests.get(full_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            
            views_data = []
            for item in items:
                views_data.append({
                    'date': datetime.strptime(item['timestamp'], '%Y%m%d00'),
                    'views': item['views']
                })
            
            return pd.DataFrame(views_data)
        else:
            print(f"   API returned status {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"   Error fetching views: {e}")
        return pd.DataFrame()

def create_view_graph(page_title, name, death_date, birth_date=None):
    """Create a view graph for a person"""
    print(f"\nFetching data for {name} ({page_title})...")
    
    death_dt = datetime.strptime(death_date, "%Y-%m-%d")
    
    # Get 1 year before to 1 year after death
    start_date = death_dt - timedelta(days=365)
    end_date = death_dt + timedelta(days=365)
    
    df = get_wikipedia_views(page_title, start_date, end_date)
    
    if len(df) == 0:
        print(f"   No data available for {name}")
        return None
    
    df['days_since_death'] = (df['date'] - death_dt).dt.days
    df = df.sort_values('date')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot views
    ax.plot(df['days_since_death'], df['views'], linewidth=2, color='#667eea', alpha=0.8)
    ax.fill_between(df['days_since_death'], df['views'], alpha=0.3, color='#667eea')
    
    # Add death date line
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Death Date', alpha=0.7)
    
    # Add day 30 and 365 markers (legend calculation window)
    ax.axvline(x=30, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label='Day 30 (legend window start)')
    ax.axvline(x=365, color='orange', linestyle=':', linewidth=1.5, alpha=0.6, label='Day 365 (legend window end)')
    
    # Calculate and show legend metrics
    post_window = df[(df['days_since_death'] >= 30) & (df['days_since_death'] <= 365)]
    pre_window = df[(df['days_since_death'] >= -365) & (df['days_since_death'] < 0)]
    
    if len(post_window) > 0 and len(pre_window) > 0:
        post_avg = post_window['views'].mean()
        pre_avg = pre_window['views'].mean()
        sustained_ratio = post_avg / pre_avg if pre_avg > 0 else 0
        is_legend = (sustained_ratio > 2.5) and (post_avg > 50)
        
        # Add text box with metrics
        legend_text = f"Legend Window (Days 30-365):\n"
        legend_text += f"Post-death avg: {post_avg:.0f} views/day\n"
        legend_text += f"Pre-death avg: {pre_avg:.0f} views/day\n"
        legend_text += f"Sustained ratio: {sustained_ratio:.2f}×\n"
        legend_text += f"Status: {'LEGEND ✓' if is_legend else 'Not Legend'}"
        
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                fontweight='bold')
    
    # Styling
    ax.set_xlabel('Days Since Death', fontweight='bold', fontsize=12)
    ax.set_ylabel('Daily Wikipedia Views', fontweight='bold', fontsize=12)
    ax.set_title(f'{name} - Wikipedia Pageviews Over Time', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    # Add age at death if available
    if birth_date:
        birth_dt = datetime.strptime(birth_date, "%Y-%m-%d")
        age = (death_dt - birth_dt).days / 365.25
        ax.text(0.98, 0.02, f'Age at Death: {age:.0f} years', 
                transform=ax.transAxes, fontsize=10,
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    filename = f"figures/view_graph_{page_title.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {filename}")
    
    plt.close()
    return df

# Load dataset to get a random person
print("Loading dataset to find a random person...")
df = pd.read_csv("../setup/modeling_data_balanced.csv")
df = df[df['date_of_death'].notna()].copy()

# Get a random person (not Pop Smoke or Jimmy Carter)
df_random = df[~df['page_title'].isin(['Pop_Smoke', 'Jimmy_Carter'])].copy()
if len(df_random) > 0:
    random_person = df_random.sample(1).iloc[0]
    random_name = random_person.get('name', random_person.get('page_title', 'Unknown'))
    random_death = random_person['date_of_death']
    random_birth = random_person.get('date_of_birth', None)
    print(f"   Selected random person: {random_name} (died {random_death})")
else:
    random_person = None

# People to visualize
people = [
    {
        'page_title': 'Pop_Smoke',
        'name': 'Pop Smoke',
        'death_date': '2020-02-19',
        'birth_date': '1999-07-20'
    },
    {
        'page_title': 'Jimmy_Carter',
        'name': 'Jimmy Carter',
        'death_date': '2024-11-18',  # Actually still alive, but let's check
        'birth_date': '1924-10-01'
    }
]

# Add random person if available
if random_person is not None:
    people.append({
        'page_title': random_person['page_title'],
        'name': random_name,
        'death_date': random_death,
        'birth_date': random_birth
    })

print("="*60)
print("GENERATING VIEW GRAPHS")
print("="*60)

for person in people:
    create_view_graph(
        person['page_title'],
        person['name'],
        person['death_date'],
        person['birth_date']
    )

print("\n" + "="*60)
print("DONE!")
print("="*60)

