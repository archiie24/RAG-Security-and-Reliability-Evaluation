import streamlit as st
import sqlite3
import pandas as pd
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "rag_evaluation.db"

st.set_page_config(
    page_title="Secure RAG Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# DATABASE
# ============================================================

@st.cache_data(ttl=2)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    try:
        queries = pd.read_sql_query(
            """
            SELECT
                id,
                run_id,
                query,
                config,
                latency_ms,
                evidence_score,
                abstained,
                groundedness,
                unsupported_claim_rate,
                confidence
            FROM query_logs
            ORDER BY id DESC
            """,
            conn
        )
    except Exception:
        queries = pd.read_sql_query(
            """
            SELECT
                id,
                run_id,
                query,
                config,
                latency_ms,
                evidence_score,
                abstained,
                groundedness,
                unsupported_claim_rate
            FROM query_logs
            ORDER BY id DESC
            """,
            conn
        )
        queries["confidence"] = queries["evidence_score"]

    experiments = pd.read_sql_query(
        """
        SELECT
            run_id,
            timestamp,
            config,
            benchmark,
            metrics_json
        FROM experiments
        ORDER BY timestamp DESC
        """,
        conn
    )

    conn.close()

    return queries, experiments


queries, experiments = load_data()

# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Secure RAG Security & Reliability Monitor")

st.caption(
    "Evaluation dashboard for retrieval quality, grounding, "
    "abstention behavior and query performance."
)

st.divider()

# ============================================================
# KPI SECTION
# ============================================================

if len(queries) > 0:

    total_queries = len(queries)

    mean_groundedness = queries["groundedness"].mean()

    mean_evidence = queries["evidence_score"].mean()

    abstention_rate = queries["abstained"].mean()

    mean_latency = queries["latency_ms"].mean()

    unsupported_rate = queries["unsupported_claim_rate"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Logged Queries",
        f"{total_queries}"
    )

    col2.metric(
        "Mean Groundedness",
        f"{mean_groundedness:.2f}"
    )

    col3.metric(
        "Mean Evidence Score",
        f"{mean_evidence:.2f}"
    )

    col4.metric(
        "Abstention Rate",
        f"{abstention_rate:.0%}"
    )

    col5.metric(
        "Mean Latency",
        f"{mean_latency:,.0f} ms"
    )

else:

    st.warning("No query logs found yet.")

# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.subheader("System reliability overview")

if len(queries) > 0:

    a, b, c = st.columns(3)

    a.metric(
        "Unsupported Claim Rate",
        f"{unsupported_rate:.2f}"
    )

    b.metric(
        "Successful Evidence Retrieval",
        f"{(queries['evidence_score'] >= 0.5).mean():.0%}"
    )

    c.metric(
        "Abstained Queries",
        f"{queries['abstained'].sum()}"
    )

# ============================================================
# EXPERIMENTS
# ============================================================

st.subheader("Evaluation experiments")

if len(experiments) > 0:

    exp_display = experiments.copy()

    exp_display["timestamp"] = pd.to_datetime(
        exp_display["timestamp"]
    ).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        exp_display[
            [
                "run_id",
                "timestamp",
                "config",
                "benchmark",
                "metrics_json"
            ]
        ],
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "No benchmark experiment has been recorded yet. "
        "Run the local evaluation summary script to create one."
    )

# ============================================================
# QUERY RELIABILITY
# ============================================================

st.subheader("Query reliability")

if len(queries) > 0:

    display = queries.copy()

    display["abstained"] = display["abstained"].map(
        {0: "No", 1: "Yes"}
    )

    display["latency_ms"] = display["latency_ms"].round(1)

    for col in [
        "evidence_score",
        "groundedness",
        "unsupported_claim_rate",
        "confidence"
    ]:
        display[col] = display[col].round(4)

    st.dataframe(
        display[
            [
                "id",
                "query",
                "config",
                "latency_ms",
                "evidence_score",
                "abstained",
                "groundedness",
                "unsupported_claim_rate",
                "confidence"
            ]
        ],
        width="stretch",
        height=330,
        hide_index=True
    )

# ============================================================
# CHART 1 — GROUNDEDNESS
# ============================================================

if len(queries) > 0:

    st.subheader("Groundedness over logged queries")

    chart_data = (
        queries
        .sort_values("id")
        [["id", "groundedness"]]
        .set_index("id")
    )

    st.line_chart(
        chart_data,
        width="stretch"
    )

# ============================================================
# CHART 2 — RETRIEVAL / GROUNDING
# ============================================================

if len(queries) > 0:

    st.subheader("Retrieval and grounding quality")

    quality = (
        queries
        .sort_values("id")
        [["id", "evidence_score", "groundedness"]]
        .set_index("id")
    )

    st.line_chart(
        quality,
        width="stretch"
    )

# ============================================================
# CHART 3 — LATENCY
# ============================================================

if len(queries) > 0:

    st.subheader("Query latency")

    latency = (
        queries
        .sort_values("id")
        [["id", "latency_ms"]]
        .set_index("id")
    )

    st.line_chart(
        latency,
        width="stretch"
    )

# ============================================================
# CHART 4 — ABSTENTION / UNSUPPORTED CLAIMS
# ============================================================

if len(queries) > 0:

    st.subheader("Safety behavior")

    safety = queries.sort_values("id")[
        [
            "id",
            "unsupported_claim_rate"
        ]
    ].set_index("id")

    st.line_chart(
        safety,
        width="stretch"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Database: {DB_PATH.name}  •  "
    f"Configuration: hardened  •  "
    f"Logged observations: {len(queries)}"
)