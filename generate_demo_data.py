"""
Generate synthetic demonstration files for the public Streamlit portfolio demo.

IMPORTANT:
- All values produced by this script are artificial.
- They are not derived from, sampled from, or intended to reproduce company/customer data.
- Use these files only to demonstrate the interface and code publicly.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 1000
OUT = Path("demo_data")
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# 1) Synthetic customer-level dataset expected by streamlit_app.py
# ---------------------------------------------------------------------

customer_id = [f"DEMO_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]

tenure = np.clip(rng.gamma(shape=2.6, scale=10, size=N_CUSTOMERS), 1, 72)
frequency = np.clip(
    rng.poisson(lam=np.maximum(1.2, 2.0 + tenure / 8)),
    1,
    None,
)
avg_purchase = np.clip(rng.lognormal(mean=2.3, sigma=0.55, size=N_CUSTOMERS), 2, 120)
monetary = frequency * avg_purchase
recency = np.clip(rng.gamma(shape=1.8, scale=16, size=N_CUSTOMERS), 0, 150)

life_time = np.clip(tenure + rng.normal(0, 2.0, N_CUSTOMERS), 1, 72)
nb_offers = rng.choice([1, 2, 3, 4], size=N_CUSTOMERS, p=[0.56, 0.28, 0.12, 0.04])

offers = np.array(["Demo Start", "Demo Plus", "Demo Max", "Demo Flex", "Demo Connect"])
main_offer = rng.choice(offers, size=N_CUSTOMERS, p=[0.24, 0.25, 0.18, 0.18, 0.15])

frequency_per_month = frequency / np.maximum(tenure, 1)
monetary_per_month = monetary / np.maximum(tenure, 1)
avg_amount = monetary / frequency

# Quintile scores.
# Recency is reversed: lower recency -> better R score.
def quintile_score(values, reverse=False):
    ranks = pd.Series(values).rank(method="first")
    q = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    if reverse:
        q = 6 - q
    return q.to_numpy()

r_score = quintile_score(recency, reverse=True)
f_score = quintile_score(frequency)
m_score = quintile_score(monetary)

rfm_total = r_score + f_score + m_score
rfm_weighted = 0.2 * r_score + 0.3 * f_score + 0.5 * m_score
rfm_score = (
    pd.Series(r_score).astype(str)
    + pd.Series(f_score).astype(str)
    + pd.Series(m_score).astype(str)
)

# Simple RFM segment
segment = np.select(
    [
        (r_score >= 4) & (f_score >= 4) & (m_score >= 4),
        (r_score >= 3) & (f_score >= 3),
        (r_score <= 2) & (f_score <= 2),
    ],
    ["Champion", "Client fidèle", "Client à risque"],
    default="Client moyen",
)

# More detailed behavioural segment
segment_advanced = np.select(
    [
        tenure <= 3,
        (r_score >= 4) & (f_score >= 4),
        (r_score >= 3) & (f_score >= 4),
        (r_score >= 4) & (f_score <= 2),
        (r_score <= 2) & (f_score >= 4),
        (r_score <= 2) & (f_score <= 2),
        (r_score == 3) & (f_score <= 3),
    ],
    [
        "Nouveaux clients",
        "Champions",
        "Clients fidèles",
        "Clients prometteurs",
        "Ne doit pas les perdre",
        "Clients hibernants",
        "Clients à développer",
    ],
    default="Clients à risque",
)

value_label = pd.qcut(
    monetary,
    3,
    labels=["Faible valeur", "Valeur moyenne", "Forte valeur"],
)

# Synthetic churn mechanism used only for the DEMO.
# It deliberately does NOT reproduce any real company rule.
logit = (
    -2.0
    + 0.026 * recency
    - 0.17 * np.log1p(frequency)
    - 0.010 * tenure
    + rng.normal(0, 0.45, N_CUSTOMERS)
)
churn_proba = 1 / (1 + np.exp(-logit))
churned = rng.binomial(1, np.clip(churn_proba, 0.02, 0.98))

# Synthetic cluster labels, chosen from behavioural dimensions only for visual demo.
cluster = np.select(
    [
        (monetary >= np.quantile(monetary, 0.70)) & (recency <= np.quantile(recency, 0.45)),
        recency >= np.quantile(recency, 0.72),
        frequency >= np.quantile(frequency, 0.70),
    ],
    [0, 1, 2],
    default=3,
)

# Demo PCA-like coordinates for visualization.
pc1 = (
    0.7 * (np.log1p(monetary) - np.log1p(monetary).mean()) / np.log1p(monetary).std()
    + 0.5 * (frequency - frequency.mean()) / frequency.std()
    - 0.4 * (recency - recency.mean()) / recency.std()
    + rng.normal(0, 0.35, N_CUSTOMERS)
)
pc2 = (
    0.7 * (tenure - tenure.mean()) / tenure.std()
    - 0.3 * (frequency - frequency.mean()) / frequency.std()
    + rng.normal(0, 0.45, N_CUSTOMERS)
)

clients = pd.DataFrame(
    {
        "msisdn_crypte": customer_id,
        "Recency": np.round(recency, 0).astype(int),
        "Frequency": frequency.astype(int),
        "Monetary": np.round(monetary, 2),
        "Tenure_Months": np.round(tenure, 1),
        "Nb_Offers_Distinctes": nb_offers,
        "Main_Offer": main_offer,
        "R_score": r_score,
        "F_score": f_score,
        "M_score": m_score,
        "RFM_Score": rfm_score,
        "RFM_Total": rfm_total,
        "RFM_Weighted": np.round(rfm_weighted, 2),
        "Segment": segment,
        "Segment_Avance": segment_advanced,
        "Valeur": value_label.astype(str),
        "Life_Time": np.round(life_time, 1),
        "Montant_Moyen_Par_Achat": np.round(avg_amount, 2),
        "Frequency_Par_Mois": np.round(frequency_per_month, 3),
        "Monetary_Par_Mois": np.round(monetary_per_month, 2),
        "Cluster_KMeans": cluster.astype(int),
        "Churned": churned.astype(int),
        "Churn_Proba": np.round(churn_proba, 4),
        "PC1": np.round(pc1, 4),
        "PC2": np.round(pc2, 4),
    }
)

clients.to_csv(OUT / "clients_export.csv", index=False)

# ---------------------------------------------------------------------
# 2) Synthetic model-comparison table
# These values are DEMO values, not internship results.
# ---------------------------------------------------------------------

model_comparison = pd.DataFrame(
    {
        "Modele": ["Random Forest", "XGBoost", "LightGBM"],
        "Accuracy": [0.82, 0.84, 0.83],
        "F1-score (classe 1)": [0.74, 0.78, 0.76],
        "AUC-ROC": [0.86, 0.89, 0.88],
        "Temps GridSearchCV (s)": [12.8, 18.6, 10.4],
        "Temps prédiction (ms)": [8.4, 6.7, 5.9],
        "Temps prédiction (µs/client)": [8.4, 6.7, 5.9],
    }
)
model_comparison.to_csv(OUT / "model_comparison.csv", index=False)

# ---------------------------------------------------------------------
# 3) Synthetic feature importance table
# ---------------------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "Feature": [
            "Frequency_Par_Mois",
            "Monetary_Par_Mois",
            "Tenure_Months",
            "Frequency",
            "Monetary",
            "Life_Time",
            "Nb_Offers_Distinctes",
            "Montant_Moyen_Par_Achat",
        ],
        "Importance": [0.23, 0.19, 0.16, 0.14, 0.11, 0.08, 0.05, 0.04],
        "Modele": ["XGBoost"] * 8,
    }
)
feature_importance.to_csv(OUT / "feature_importance.csv", index=False)

print("Synthetic demo files created:")
for path in sorted(OUT.glob("*.csv")):
    print(f" - {path} ({path.stat().st_size:,} bytes)")
print("\nAll values are synthetic and are not derived from company/customer data.")
