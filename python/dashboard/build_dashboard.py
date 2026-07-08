"""
Build a single, standalone dashboard.html from the dbt marts + the
anomaly detection / NLP outputs. Plotly.js is embedded inline so the file
renders fully offline -- no dev server, no CDN, no internet connection
required. Just open dashboard/dashboard.html in a browser.

In production this exact modeled data (main_marts.* in the warehouse) could
be plugged directly into Power BI or Tableau instead of this script -- see
the README "Porting to production" section.

Run:
    python python/dashboard/build_dashboard.py
"""

from __future__ import annotations

import os

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

DB_PATH = "warehouse/transaction_analytics.duckdb"
ANOMALIES_PATH = "outputs/anomalies.csv"
THEMES_PATH = "outputs/ticket_themes.csv"
SUMMARY_PATH = "outputs/executive_summary.txt"
OUTPUT_PATH = "dashboard/dashboard.html"

SEVERITY_COLOR = {"medium": "#f5b942", "high": "#f57c42", "critical": "#d64545"}
BRAND = {
    "bg": "#0f1420",
    "panel": "#161d2e",
    "text": "#e8ecf4",
    "muted": "#93a0b8",
    "accent": "#4c8bf5",
    "grid": "#232c40",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    daily_kpis = con.execute("SELECT * FROM main_marts.mart_daily_kpis").df()
    fct = con.execute(
        "SELECT transaction_date, region, channel, amount, status, customer_id FROM main_marts.fct_transactions"
    ).df()
    con.close()

    daily_kpis["transaction_date"] = pd.to_datetime(daily_kpis["transaction_date"])

    anomalies = pd.read_csv(ANOMALIES_PATH)
    anomalies["date"] = pd.to_datetime(anomalies["date"])
    anomalies["end_date"] = pd.to_datetime(anomalies["end_date"])

    themes = pd.read_csv(THEMES_PATH).sort_values("ticket_count", ascending=False)

    summary_text = None
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary_text = f.read().strip()

    return daily_kpis, fct, anomalies, themes, summary_text


# --------------------------------------------------------------------------
# KPI header
# --------------------------------------------------------------------------

def compute_kpis(fct: pd.DataFrame, anomalies: pd.DataFrame) -> dict:
    completed = fct[fct["status"] == "completed"]
    return {
        "total_volume": completed["amount"].sum(),
        "total_transactions": len(fct),
        "active_customers": fct["customer_id"].nunique(),
        "flagged_anomalies": len(anomalies),
        "critical_anomalies": (anomalies["severity"] == "critical").sum(),
        "date_min": fct["transaction_date"].min(),
        "date_max": fct["transaction_date"].max(),
    }


def kpi_cards_html(kpis: dict) -> str:
    cards = [
        ("Total Transaction Volume", f"${kpis['total_volume']:,.0f}", "completed transactions"),
        ("Total Transactions", f"{kpis['total_transactions']:,}", "all statuses"),
        ("Active Customers", f"{kpis['active_customers']:,}", "made >=1 transaction"),
        ("Flagged Anomalies", f"{kpis['flagged_anomalies']}", f"{kpis['critical_anomalies']} critical"),
    ]
    cards_html = "".join(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """
        for label, value, sub in cards
    )
    return f'<div class="kpi-row">{cards_html}</div>'


# --------------------------------------------------------------------------
# Merge overlapping anomaly windows across metrics into display bands
# --------------------------------------------------------------------------

def merge_anomaly_bands(anomalies: pd.DataFrame) -> pd.DataFrame:
    intervals = anomalies[["date", "end_date", "severity", "description"]].sort_values("date")
    severity_rank = {"medium": 1, "high": 2, "critical": 3}

    bands = []
    for row in intervals.itertuples(index=False):
        if bands and row.date <= bands[-1]["end"] + pd.Timedelta(days=2):
            bands[-1]["end"] = max(bands[-1]["end"], row.end_date)
            bands[-1]["severities"].append(row.severity)
            bands[-1]["descriptions"].append(row.description)
        else:
            bands.append({
                "start": row.date, "end": row.end_date,
                "severities": [row.severity], "descriptions": [row.description],
            })

    for b in bands:
        b["severity"] = max(b["severities"], key=lambda s: severity_rank[s])
        b["label"] = f"{len(b['descriptions'])} anomaly signal(s) flagged " \
                      f"{b['start'].date()} to {b['end'].date()}"

    return pd.DataFrame(bands)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def chart_time_series(daily_kpis: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    system_daily = daily_kpis.groupby("transaction_date", as_index=False).agg(
        txn_count=("txn_count", "sum"), total_amount=("total_amount", "sum")
    )
    bands = merge_anomaly_bands(anomalies)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=system_daily["transaction_date"], y=system_daily["txn_count"],
            name="Daily Transaction Count", line=dict(color=BRAND["accent"], width=1.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=system_daily["transaction_date"], y=system_daily["total_amount"],
            name="Daily Transaction Amount ($)", line=dict(color="#9b6cf0", width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )

    for b in bands.itertuples(index=False):
        fig.add_vrect(
            x0=b.start, x1=b.end + pd.Timedelta(days=1),
            fillcolor=SEVERITY_COLOR[b.severity], opacity=0.18, line_width=0,
        )
        fig.add_trace(
            go.Scatter(
                x=[b.start + (b.end - b.start) / 2], y=[system_daily["txn_count"].max() * 1.05],
                mode="markers", marker=dict(size=1, color=SEVERITY_COLOR[b.severity]),
                hovertext=b.label, hoverinfo="text", showlegend=False,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="Daily Transaction Volume & Amount (shaded = flagged anomaly window)",
        template="plotly_dark", paper_bgcolor=BRAND["panel"], plot_bgcolor=BRAND["panel"],
        font=dict(color=BRAND["text"]), legend=dict(orientation="h", y=1.12),
        margin=dict(t=70, l=60, r=60, b=40), height=440,
    )
    fig.update_xaxes(gridcolor=BRAND["grid"])
    fig.update_yaxes(title_text="Transaction Count", gridcolor=BRAND["grid"], secondary_y=False)
    fig.update_yaxes(title_text="Transaction Amount ($)", gridcolor=BRAND["grid"], secondary_y=True)
    return fig


def chart_region_channel(fct: pd.DataFrame) -> go.Figure:
    region_agg = fct.groupby("region", as_index=False).agg(amount=("amount", "sum"), count=("amount", "size"))
    channel_agg = fct.groupby("channel", as_index=False).agg(amount=("amount", "sum"), count=("amount", "size"))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("By Region", "By Channel"))
    fig.add_trace(
        go.Bar(x=region_agg["region"], y=region_agg["amount"], name="Amount by Region",
               marker_color=BRAND["accent"]),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=channel_agg["channel"], y=channel_agg["amount"], name="Amount by Channel",
               marker_color="#9b6cf0"),
        row=1, col=2,
    )
    fig.update_layout(
        title="Total Transaction Amount by Region and Channel",
        template="plotly_dark", paper_bgcolor=BRAND["panel"], plot_bgcolor=BRAND["panel"],
        font=dict(color=BRAND["text"]), showlegend=False,
        margin=dict(t=70, l=60, r=40, b=40), height=400,
    )
    fig.update_yaxes(title_text="Total Amount ($)", gridcolor=BRAND["grid"])
    fig.update_xaxes(gridcolor=BRAND["grid"])
    return fig


def chart_ticket_themes(themes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=themes["theme_label"], y=themes["ticket_count"], name="Ticket Count",
            marker=dict(
                color=themes["avg_sentiment"], colorscale="RdYlGn", cmin=-1, cmax=1,
                colorbar=dict(title="Avg<br>Sentiment"),
            ),
            text=themes["avg_sentiment"].apply(lambda s: f"sentiment {s:+.2f}"),
            textposition="outside",
            customdata=themes[["dominant_category", "trend", "category_purity_pct"]],
            hovertemplate=(
                "<b>%{x}</b><br>Tickets: %{y}<br>Dominant category: %{customdata[0]}"
                "<br>Trend: %{customdata[1]}<br>Category purity: %{customdata[2]}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Support Ticket Themes (bar height = volume, color = avg sentiment)",
        template="plotly_dark", paper_bgcolor=BRAND["panel"], plot_bgcolor=BRAND["panel"],
        font=dict(color=BRAND["text"]),
        margin=dict(t=70, l=60, r=40, b=120), height=460,
        xaxis=dict(tickangle=-25, gridcolor=BRAND["grid"]),
        yaxis=dict(title="Ticket Count", gridcolor=BRAND["grid"]),
    )
    return fig


# --------------------------------------------------------------------------
# Anomaly table (plain HTML, sorted by severity)
# --------------------------------------------------------------------------

def anomaly_table_html(anomalies: pd.DataFrame) -> str:
    severity_rank = {"critical": 3, "high": 2, "medium": 1}
    top = anomalies.copy()
    top["rank"] = top["severity"].map(severity_rank)
    top = top.sort_values(["rank", "zscore"], key=lambda s: s.abs() if s.name == "zscore" else s, ascending=False).head(12)

    rows = "".join(
        f"""<tr>
            <td>{r.date.date()}{' to ' + str(r.end_date.date()) if r.date != r.end_date else ''}</td>
            <td>{r.metric.replace('_', ' ')}</td>
            <td>{r.region if pd.notna(r.region) else '—'}</td>
            <td><span class="badge badge-{r.severity}">{r.severity}</span></td>
            <td>{r.description}</td>
        </tr>"""
        for r in top.itertuples(index=False)
    )
    return f"""
    <table class="anomaly-table">
        <thead><tr><th>Date</th><th>Metric</th><th>Region/Channel</th><th>Severity</th><th>Description</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


# --------------------------------------------------------------------------
# Assemble HTML
# --------------------------------------------------------------------------

CSS = f"""
* {{ box-sizing: border-box; }}
body {{
    background: {BRAND['bg']}; color: {BRAND['text']};
    font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 32px 40px 60px;
}}
.page {{ max-width: 1180px; margin: 0 auto; }}
h1 {{ font-size: 26px; margin-bottom: 4px; }}
.subtitle {{ color: {BRAND['muted']}; margin-top: 0; margin-bottom: 28px; font-size: 14px; }}
.kpi-row {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
.kpi-card {{
    background: {BRAND['panel']}; border: 1px solid {BRAND['grid']}; border-radius: 10px;
    padding: 18px 22px; flex: 1; min-width: 200px;
}}
.kpi-label {{ color: {BRAND['muted']}; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
.kpi-value {{ font-size: 30px; font-weight: 700; margin: 6px 0 2px; }}
.kpi-sub {{ color: {BRAND['muted']}; font-size: 12px; }}
.panel {{
    background: {BRAND['panel']}; border: 1px solid {BRAND['grid']}; border-radius: 10px;
    padding: 8px; margin-bottom: 28px;
}}
.summary-panel {{
    background: {BRAND['panel']}; border: 1px solid {BRAND['accent']}; border-radius: 10px;
    padding: 22px 26px; margin-bottom: 28px; line-height: 1.6; font-size: 15px;
}}
.summary-panel h2 {{ margin-top: 0; font-size: 16px; color: {BRAND['accent']}; }}
.summary-panel .missing {{ color: {BRAND['muted']}; font-style: italic; }}
.grid-2 {{ display: grid; grid-template-columns: 1fr; gap: 28px; }}
.anomaly-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.anomaly-table th {{
    text-align: left; padding: 10px 12px; color: {BRAND['muted']}; border-bottom: 1px solid {BRAND['grid']};
    text-transform: uppercase; font-size: 11px; letter-spacing: .04em;
}}
.anomaly-table td {{ padding: 10px 12px; border-bottom: 1px solid {BRAND['grid']}; vertical-align: top; }}
.anomaly-table td:first-child {{ white-space: nowrap; }}
.badge {{ padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; text-transform: uppercase; }}
.badge-critical {{ background: #d6454522; color: #ff8f8f; }}
.badge-high {{ background: #f57c4222; color: #ffb27a; }}
.badge-medium {{ background: #f5b94222; color: #ffd98a; }}
footer {{ color: {BRAND['muted']}; font-size: 12px; margin-top: 40px; }}
"""


def main() -> None:
    daily_kpis, fct, anomalies, themes, summary_text = load_data()
    kpis = compute_kpis(fct, anomalies)

    plotly_js = pyo.get_plotlyjs()

    fig_ts = chart_time_series(daily_kpis, anomalies)
    fig_region = chart_region_channel(fct)
    fig_themes = chart_ticket_themes(themes)

    div_ts = fig_ts.to_html(full_html=False, include_plotlyjs=False)
    div_region = fig_region.to_html(full_html=False, include_plotlyjs=False)
    div_themes = fig_themes.to_html(full_html=False, include_plotlyjs=False)

    if summary_text:
        summary_html = f"<h2>AI Executive Summary</h2><p>{summary_text}</p>"
    else:
        summary_html = (
            "<h2>AI Executive Summary</h2>"
            "<p class='missing'>Not generated -- this optional layer calls the Anthropic API and "
            "requires an <code>ANTHROPIC_API_KEY</code> environment variable. Run "
            "<code>python python/ai_summary/generate_summary.py</code> with a key configured to "
            "populate this section. Every other part of this dashboard runs standalone without it.</p>"
        )

    date_range = f"{kpis['date_min'].date()} to {kpis['date_max'].date()}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Transaction Analytics & Anomaly Detection Pipeline</title>
<style>{CSS}</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="page">
<h1>Transaction Analytics &amp; Anomaly Detection Pipeline</h1>
<p class="subtitle">Fintech transaction &amp; support ticket analytics &middot; {date_range} &middot;
data modeled with dbt + DuckDB, anomalies flagged with a seasonal rolling z-score detector,
ticket themes discovered with TF-IDF + KMeans</p>

{kpi_cards_html(kpis)}

<div class="summary-panel">{summary_html}</div>

<div class="panel">{div_ts}</div>

<div class="panel">{div_region}</div>

<div class="panel">{div_themes}</div>

<div class="panel" style="padding: 20px 24px;">
<h2 style="margin-top:0; font-size:16px;">Top Flagged Anomalies</h2>
{anomaly_table_html(anomalies)}
</div>

<footer>
Generated by python/dashboard/build_dashboard.py from the dbt-modeled DuckDB warehouse.
This static file embeds Plotly.js inline and requires no server -- open it directly in a browser.
</footer>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Wrote dashboard -> {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
