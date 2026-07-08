"""
NLP / text mining over support tickets: TF-IDF vectorization, unsupervised
theme clustering (KMeans), and per-ticket sentiment scoring (VADER).

The synthetic ticket generator (python/etl/generate_synthetic_data.py) tags
each ticket with a `category` label, but that label is intentionally NOT fed
into the clustering step -- the whole point of this module is to discover
themes purely from the free-text subject/body, the way you'd have to on real
unlabeled support data. The known `category` is only used afterwards, to
sanity-check how well the unsupervised clusters line up with the labels a
human would have assigned (see `dominant_category` / `category_purity` in
the output).

Outputs:
    outputs/ticket_themes.csv          - one row per theme (cluster)
    outputs/ticket_sentiment.csv       - one row per ticket (theme + sentiment)

Run:
    python python/nlp/analyze_tickets.py
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DB_PATH = "warehouse/transaction_analytics.duckdb"
THEMES_OUTPUT_PATH = "outputs/ticket_themes.csv"
TICKETS_OUTPUT_PATH = "outputs/ticket_sentiment.csv"

RANDOM_STATE = 42
K_CANDIDATES = range(5, 11)   # search for the best number of theme clusters
TOP_TERMS_PER_THEME = 8


def load_tickets() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("SELECT * FROM main_staging.stg_support_tickets").df()
    con.close()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["text"] = (df["subject"].fillna("") + ". " + df["body_text"].fillna("")).str.strip()
    return df


def vectorize(texts: pd.Series) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,       # ignore terms that appear in fewer than 5 tickets
        max_df=0.5,     # ignore terms that appear in more than half the tickets (too generic)
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def choose_k(matrix, candidates) -> tuple[int, KMeans]:
    """Pick the cluster count with the best silhouette score."""
    best_k, best_score, best_model = None, -1.0, None
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels, sample_size=2000, random_state=RANDOM_STATE)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, model
    print(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k, best_model


def top_terms_for_cluster(model: KMeans, vectorizer: TfidfVectorizer, cluster_id: int, n: int) -> list[str]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    center = model.cluster_centers_[cluster_id]
    top_idx = center.argsort()[::-1][:n]
    return list(feature_names[top_idx])


def trend_slope(dates: pd.Series) -> tuple[float, str]:
    """
    Linear trend in weekly ticket counts for a theme: fit count ~ week_index
    and classify the slope as rising / falling / stable relative to the
    theme's average weekly volume.
    """
    weekly = dates.dt.to_period("W").value_counts().sort_index()
    if len(weekly) < 3:
        return 0.0, "stable"
    x = np.arange(len(weekly))
    y = weekly.values
    slope = np.polyfit(x, y, 1)[0]
    avg = y.mean()
    if avg == 0:
        return 0.0, "stable"
    relative_slope = slope / avg
    if relative_slope > 0.03:
        label = "rising"
    elif relative_slope < -0.03:
        label = "falling"
    else:
        label = "stable"
    return round(float(slope), 3), label


def main() -> None:
    tickets = load_tickets()
    print(f"Loaded {len(tickets):,} tickets")

    vectorizer, matrix = vectorize(tickets["text"])
    print(f"TF-IDF matrix: {matrix.shape[0]:,} tickets x {matrix.shape[1]:,} terms")

    print("Searching for best cluster count via silhouette score...")
    best_k, model = choose_k(matrix, K_CANDIDATES)
    tickets["cluster"] = model.labels_

    # ---- sentiment scoring ----
    analyzer = SentimentIntensityAnalyzer()
    tickets["sentiment"] = tickets["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])

    # ---- per-theme summary ----
    theme_rows = []
    for cluster_id in range(best_k):
        cluster_tickets = tickets[tickets["cluster"] == cluster_id]
        top_terms = top_terms_for_cluster(model, vectorizer, cluster_id, TOP_TERMS_PER_THEME)
        theme_label = " / ".join(top_terms[:4])

        dominant_category = cluster_tickets["category"].value_counts().idxmax()
        category_purity = (
            cluster_tickets["category"].value_counts().iloc[0] / len(cluster_tickets)
        )

        slope, trend = trend_slope(cluster_tickets["created_at"])

        theme_rows.append({
            "cluster_id": cluster_id,
            "theme_label": theme_label,
            "top_terms": ", ".join(top_terms),
            "ticket_count": len(cluster_tickets),
            "avg_sentiment": round(cluster_tickets["sentiment"].mean(), 3),
            "dominant_category": dominant_category,
            "category_purity_pct": round(category_purity * 100, 1),
            "trend_slope_per_week": slope,
            "trend": trend,
        })

    themes_df = pd.DataFrame(theme_rows).sort_values("ticket_count", ascending=False)
    themes_df.to_csv(THEMES_OUTPUT_PATH, index=False)
    print(f"\nWrote {len(themes_df)} themes -> {THEMES_OUTPUT_PATH}")
    print(themes_df[["cluster_id", "theme_label", "ticket_count", "avg_sentiment", "dominant_category", "trend"]])

    # ---- per-ticket output (theme + sentiment, for dashboard drill-down) ----
    theme_label_map = themes_df.set_index("cluster_id")["theme_label"].to_dict()
    tickets["theme_label"] = tickets["cluster"].map(theme_label_map)
    ticket_out = tickets[[
        "ticket_id", "customer_id", "created_at", "category", "cluster", "theme_label", "sentiment",
    ]].rename(columns={"cluster": "cluster_id"})
    ticket_out.to_csv(TICKETS_OUTPUT_PATH, index=False)
    print(f"Wrote {len(ticket_out)} ticket-level rows -> {TICKETS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
