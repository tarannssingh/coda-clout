# Predicting Posthumous Wikipedia Legacy: A Machine Learning Approach

**Author:** Taran Singh  
**Course:** CSC 466 — Knowledge Discovery from Data  
**Date:** December 2025

---

## Abstract

Can I predict who will become a cultural legend after death? This paper presents a machine learning framework for predicting sustained posthumous attention on Wikipedia using only pre-death features. I collected data on 2,281 deceased individuals (2017–2025) with temporally balanced sampling and engineered 16 features capturing pre-death fame, demographics, and cultural footprint. After fixing critical data leakage (historical page length and num_editors) and systematic feature selection, I achieved **ROC-AUC 0.850** with **35.3% precision** and **31.6% recall** using XGBoost, demonstrating that legacy is predictable from past signals. My analysis reveals that **pre-death attention (16.7%), edit activity (12.6%), and age at death (11.2%) are the strongest predictors**; pre-death cultural footprint combined with younger age generates the most sustained cultural interest.

---

## 1. Introduction

Some deaths spark brief media attention; others create lasting cultural immortality. Understanding what distinguishes transient interest from sustained legacy has implications for sociology, media studies, and content curation. I formalize this question: _Can pre-death features predict posthumous Wikipedia attention?_

I define a **"legend"** as someone who achieves sustained cultural attention months after death—characterized by word-of-mouth discovery and new fan following rather than transient news coverage. This conceptual framework captures the "larger than life" quality of legends: their stories continue to resonate and attract new audiences long after initial media attention fades.

To operationalize this, I use objective, time-invariant criteria:

1. **Sustained ratio > 2.5×**: Average daily views from days 30–365 after death exceed pre-death baseline by 2.5×
2. **Volume threshold > 50**: Average daily views post-death exceed 50

The 2.5× threshold reflects sustained word-of-mouth: if post-death attention consistently exceeds pre-death levels by 2.5× over 11 months, this indicates genuine cultural resonance rather than a brief news spike. The 30-day delay excludes immediate death coverage, focusing on sustained discovery. The volume threshold (>50 views/day) ensures meaningful cultural impact beyond niche interest.

This definition deliberately avoids traditional fame metrics (awards, career achievements, occupation) that would bias toward athletes, musicians, or politicians. Instead, it captures anyone whose story generates sustained curiosity—whether through tragic circumstances, cultural impact, or unexpected relevance. From 2,281 deceased individuals with temporally balanced sampling (equal representation per year), **57 (2.5%)** meet these criteria. The balanced sampling prevents temporal bias where the model could learn "recent deaths = more views" rather than true legend characteristics.

---

## 2. Data & Methods

### 2.1 Data Collection

I collected Wikipedia page data for 2,281 deceased individuals via:

- **Wikidata SPARQL**: Birth dates, death dates (2017–2025), occupations, awards, cause of death
- **Wikipedia Pageview API**: Daily view counts (2 years pre-death to 1 year post-death)
- **Wikipedia API**: Page metadata (length, watchers, editors)

**Temporally balanced sampling** addressed systematic bias: I ensured equal representation across years (2017-2025) to prevent the model from learning temporal patterns instead of true legend characteristics. Without this, the model could exploit "recent deaths get more views" rather than identifying genuine cultural impact signals.

**Data leakage fix**: I identified and corrected a critical data leakage issue where `page_len_bytes` and `num_editors` were measured at data collection time (2024) rather than at time of death. This leaked information because legends receive significantly more post-death edits (e.g., Sidhu Moose Wala's page grew 184% after death). I fixed this by querying Wikipedia's revision history API to retrieve historical values: `page_len_bytes` from the day before death and `num_editors` from the year before death. This ensures all features reflect pre-death status only, eliminating leakage and producing more honest model performance.

### 2.2 Feature Engineering

I engineered **16 features** across five categories:

**Log-transformed features (7)**: Normalized right-skewed metrics using log1p transformation

1. `log_avg_views_pre_death_10d` — Average daily views in 10 days before death
2. `log_sitelinks` — Number of cross-language Wikipedia page links
3. `log_page_len_bytes` — Wikipedia page length in bytes (historical: measured day before death via revision history API)
4. `log_page_watchers` — Number of users watching the page
5. `log_edits_past_year` — Number of edits in the past year
6. `log_num_editors` — Number of unique editors
7. `log_award_count` — Number of awards/recognitions

**Demographics (2)**: Temporal and age-based features

1. `age_at_death` — Age when person died
2. `death_year` — Year of death (numeric, captures temporal cultural context)

**Occupations (3)**: One-hot encoded top occupations from Wikidata (removed 12 with <1% importance)

1. `occ_actor`
2. `occ_film_actor`
3. `occ_politician`

**Interactions (3)**: Non-linear relationships capturing complex patterns that simple features miss

1. `age_x_fame` — Age at death × fame_proxy (captures "tragic young death with established fame" pattern)
2. `age_x_year` — Age at death × (death_year - 2018) (temporal-age interaction, 4.2% importance)
3. `views_per_sitelink` — log(views) - log(sitelinks) (normalizes attention by international presence)

**Composite (1)**: Pre-death fame signal

1. `fame_proxy = log1p(avg_views_pre_death_10d) + log1p(sitelinks) + award_count − 0.5×log1p(page_len_bytes)` — Composite measure of pre-death cultural reach that subtracts page length to account for "already famous" bias (longer pages indicate established fame but leave less room for post-death spike). Uses `log1p` (log(1+x)) to handle zero values gracefully. **Note**: `page_len_bytes` uses historical values (day before death) to prevent data leakage from post-death edits.

### 2.3 Feature Selection

I initially engineered 29 features, but systematic removal of low-importance features improved model performance. Using XGBoost feature importance, I removed 13 features with <1% importance:

**Removed features:**

- `young_death` (0.0%) — Binary threshold redundant with continuous `age_at_death`
- `occ_singer`, `occ_writer` (0.0%) — No predictive signal
- `occ_musician`, `occ_film_director`, `occ_composer`, `occ_lawyer`, `occ_journalist`, `occ_American_football_player`, `occ_screenwriter`, `occ_association_football_player`, `occ_university_teacher`, `occ_television_actor` (<1%) — Low importance

Removing noise features allowed the model to focus on the 16 most predictive features, resulting in a cleaner, more interpretable model. The final model uses only features with >2.5% importance, ensuring every feature contributes meaningfully.

### 2.4 Models & Evaluation

I trained four models with **temporal split** (train: 2017–2022; validation: 2023–2025) to prevent leakage:

1. **Logistic Regression** — linear baseline
2. **Random Forest** — 500 trees, unlimited depth
3. **XGBoost** — gradient boosting, early stopping (best overall performance)
4. **HistGradientBoosting** — scikit-learn gradient boosting variant

**Model selection rationale**: XGBoost achieved the best overall performance with ROC-AUC 0.850 on the temporally balanced dataset. The gradient boosting approach handles non-linear interactions between pre-death attention, demographics, and temporal patterns effectively.

All tree-based models used **temporally balanced sampling** to prevent temporal bias. By ensuring equal representation across years (2017-2025), the model cannot exploit patterns like "recent deaths get more views" and must instead learn genuine legend characteristics. This prevents the model from overfitting to temporal patterns rather than cultural impact signals.

---

## 3. Results

### 3.1 Model Performance

| Model                | ROC-AUC   | Precision | Recall    | F1        | PR-AUC    |
| -------------------- | --------- | --------- | --------- | --------- | --------- |
| **XGBoost** ⭐       | **0.850** | **0.353** | **0.316** | **0.333** | **0.239** |
| HistGradientBoosting | 0.802     | 0.222     | 0.316     | 0.261     | 0.170     |
| Logistic Regression  | 0.784     | 0.149     | 0.579     | 0.237     | 0.176     |
| Random Forest        | 0.635     | 0.212     | 0.368     | 0.269     | 0.178     |

**XGBoost achieves the best overall performance with ROC-AUC 0.850**, demonstrating strong discriminative ability. The model achieves 35.3% precision and 31.6% recall, reflecting honest predictive performance after fixing data leakage and using temporally balanced sampling. The temporal split (2017-2022 train, 2023-2025 validation) ensures realistic performance on future data.

### 3.2 Feature Importance

XGBoost feature importance reveals the strongest predictors (using historical pre-death values):

| Rank | Feature                       | Importance | Interpretation                      |
| ---- | ----------------------------- | ---------- | ----------------------------------- |
| 1    | `log_avg_views_pre_death_10d` | 16.7%      | Pre-death attention signal          |
| 2    | `log_edits_past_year`         | 12.6%      | Recent editorial activity           |
| 3    | `age_at_death`                | 11.2%      | Younger deaths → more legends       |
| 4    | `log_page_len_bytes`          | 7.2%       | Historical pre-death fame indicator |
| 5    | `views_per_sitelink`          | 6.8%       | Views normalized by fame            |
| 6    | `occ_actor`                   | 6.2%       | Occupation signal                   |
| 7    | `death_year`                  | 6.0%       | Temporal context (cultural shifts)  |
| 8    | `occ_politician`              | 4.3%       | Occupation signal                   |
| 9    | `log_sitelinks`               | 4.3%       | Cross-language Wikipedia links      |
| 10   | `age_x_year`                  | 4.2%       | Age-temporal interaction            |

**Key insight**: Pre-death views (16.7%), edit activity (12.6%), and age at death (11.2%) are the top predictors—combined 40.5% importance. This confirms that _pre-death cultural footprint, combined with younger age, matters most_.

---

## 4. Discussion

### 4.1 Key Findings

1. **Legacy is predictable**: ROC-AUC 0.850 demonstrates pre-death features contain substantial signal about posthumous attention.

2. **Pre-death attention, edit activity, and age dominate**: Top predictors are `log_avg_views_pre_death_10d` (16.7%), `log_edits_past_year` (12.6%), and `age_at_death` (11.2%)—combined 40.5% importance.

3. **Data leakage fix was critical**: Fixing historical page length and num_editors eliminated leakage from post-death edits, producing honest model performance.

4. **Interaction features matter**: `age_x_fame` and `age_x_year` capture complex patterns like "young tragic death with high pre-death fame."

### 4.2 Limitations

- **Recall (31.6%)**: The model misses 68% of actual legends, often "surprise" legends with unexpected cultural relevance (e.g., regional fame not captured by English Wikipedia metrics).
- **Western/English bias**: Training data favors Western celebrities with longer English Wikipedia pages.
- **Missing features**: Social media, news coverage, and cultural context are not captured.
- **Temporal shift**: 2023–2025 deaths may have different patterns (e.g., COVID-19 effects).

**Mitigation**: The detector app now includes an adjustable threshold (default 0.25) to catch more legends at the cost of precision.

### 4.3 Implications

**For cultural studies**: Pre-death fame and demographics predict legacy—legacy follows patterns, not randomness.

**For Wikipedia**: The model could identify pages needing post-death maintenance or predict editing activity spikes.

**For ML practitioners**: Data leakage detection and fixing is critical. XGBoost achieved best performance (ROC-AUC 0.850) through effective handling of non-linear interactions and proper feature selection.

---

## 5. Conclusion

I demonstrated that posthumous Wikipedia legacy can be predicted from pre-death features, achieving **ROC-AUC 0.850** with **35.3% precision** and **31.6% recall** using XGBoost. Key findings: (1) Data leakage fix was critical—fixed historical page length and num_editors to eliminate post-death edit leakage; (2) Pre-death attention (16.7%), edit activity (12.6%), and age (11.2%) dominate—combined 40.5% importance; (3) Feature selection improved model by removing 13 low-importance features; (4) Interaction features capture complexity that simple features miss.

Future work should explore additional features (social media, news coverage), advanced class imbalance techniques (SMOTE, focal loss), and interpretability methods (SHAP values) to understand individual predictions.

**Code & Data**: Available at `https://github.com/taransingh/coda-clout`

---

## References

- **Dataset**: 2,281 deceased individuals (2017–2025) from Wikipedia/Wikidata with temporally balanced sampling
- **Models**: XGBoost 2.0, Random Forest, Logistic Regression, HistGradientBoosting (scikit-learn 1.3)
- **Evaluation**: Temporal split (2017–2022 train, 2023–2025 validation) with balanced sampling to prevent temporal bias
- **Feature Engineering**: 16 features across 5 categories (13 features removed with <1% importance)
- **Data Leakage Fix**: Historical `page_len_bytes` (day before death) and `num_editors` (year before death) via Wikipedia revision history API
