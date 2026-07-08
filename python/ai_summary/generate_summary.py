"""
Optional AI executive-summary layer.

Takes the flagged anomalies (outputs/anomalies.csv) and support ticket
themes (outputs/ticket_themes.csv) and asks the Anthropic API for a short,
plain-English executive summary suitable for a leadership readout.

This is a clearly secondary/optional module: the core pipeline (dbt models,
anomaly detection, NLP theme clustering, dashboard) works completely without
it. The API key is read from the ANTHROPIC_API_KEY environment variable --
it is never hardcoded and never committed to git (see .gitignore). If the
key isn't set, this script prints a clear message and exits cleanly so the
rest of the project still runs standalone.

Run:
    python python/ai_summary/generate_summary.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ANOMALIES_PATH = "outputs/anomalies.csv"
THEMES_PATH = "outputs/ticket_themes.csv"
SUMMARY_OUTPUT_PATH = "outputs/executive_summary.txt"

MODEL = "claude-sonnet-4-6"
TOP_N_ANOMALIES = 8


def build_prompt(anomalies: pd.DataFrame, themes: pd.DataFrame) -> str:
    top_anomalies = anomalies.sort_values("zscore", key=lambda s: s.abs(), ascending=False).head(TOP_N_ANOMALIES)

    anomaly_lines = "\n".join(
        f"- [{row.severity.upper()}] {row.description}" for row in top_anomalies.itertuples()
    )

    themes_sorted = themes.sort_values("ticket_count", ascending=False)
    theme_lines = "\n".join(
        f"- {row.theme_label}: {row.ticket_count} tickets, avg sentiment {row.avg_sentiment:+.2f}, "
        f"trend {row.trend}, dominant category '{row.dominant_category}'"
        for row in themes_sorted.itertuples()
    )

    return f"""You are a data analyst preparing a short executive summary for a fintech
company's leadership team, based on an automated transaction-monitoring and
support-ticket-analysis pipeline.

Top flagged transaction anomalies (from time-series monitoring):
{anomaly_lines}

Support ticket themes (from unsupervised clustering of ticket text):
{theme_lines}

Write a 3-5 sentence plain-English executive summary. Call out the most
business-relevant anomalies (e.g. suspected fraud, service outages, demand
surges) and the most significant support ticket themes (especially negative-
sentiment ones), and connect them where relevant (e.g. a spike in failed-
transaction tickets around the same time as a processing outage). Do not use
bullet points -- write flowing prose suitable for a leadership readout."""


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set -- skipping the optional AI executive-summary "
            "layer. This is expected if you haven't configured an API key; the rest of "
            "the pipeline (dbt models, anomaly detection, NLP, dashboard) is unaffected."
        )
        return

    if not os.path.exists(ANOMALIES_PATH) or not os.path.exists(THEMES_PATH):
        print(
            f"Missing {ANOMALIES_PATH} or {THEMES_PATH} -- run the anomaly detection and "
            "NLP steps first. Skipping AI summary."
        )
        return

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package isn't installed -- skipping the optional AI summary layer.")
        return

    anomalies = pd.read_csv(ANOMALIES_PATH)
    themes = pd.read_csv(THEMES_PATH)
    prompt = build_prompt(anomalies, themes)

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - any API failure should degrade gracefully
        print(f"Anthropic API call failed ({exc}) -- skipping AI summary. Rest of pipeline unaffected.")
        return

    summary_text = "".join(block.text for block in response.content if block.type == "text").strip()

    os.makedirs(os.path.dirname(SUMMARY_OUTPUT_PATH), exist_ok=True)
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"Wrote AI executive summary -> {SUMMARY_OUTPUT_PATH}\n")
    print(summary_text)


if __name__ == "__main__":
    sys.exit(main() or 0)
