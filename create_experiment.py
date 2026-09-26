import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "rag_evaluation.db"

conn = sqlite3.connect(DB_PATH)

df = conn.execute("""
    SELECT
        latency_ms,
        evidence_score,
        abstained,
        groundedness,
        unsupported_claim_rate
    FROM query_logs
""").fetchall()

if not df:
    print("No query logs found.")
    conn.close()
    raise SystemExit

latencies = [float(x[0] or 0) for x in df]
evidence = [float(x[1] or 0) for x in df]
abstained = [int(x[2] or 0) for x in df]
groundedness = [float(x[3] or 0) for x in df]
unsupported = [float(x[4] or 0) for x in df]

metrics = {
    "queries_evaluated": len(df),
    "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
    "mean_evidence_score": round(sum(evidence) / len(evidence), 4),
    "mean_groundedness": round(sum(groundedness) / len(groundedness), 4),
    "abstention_rate": round(sum(abstained) / len(abstained), 4),
    "mean_unsupported_claim_rate": round(
        sum(unsupported) / len(unsupported), 4
    )
}

run_id = str(uuid.uuid4())

conn.execute("""
    INSERT INTO experiments
    (run_id, timestamp, config, benchmark, metrics_json)
    VALUES (?, ?, ?, ?, ?)
""", (
    run_id,
    datetime.now(timezone.utc).isoformat(),
    "hardened",
    "logged_query_evaluation",
    json.dumps(metrics)
))

conn.commit()
conn.close()

print("\nEvaluation experiment created successfully.")
print("---------------------------------------------")
print(f"Queries evaluated : {metrics['queries_evaluated']}")
print(f"Mean groundedness : {metrics['mean_groundedness']}")
print(f"Mean evidence     : {metrics['mean_evidence_score']}")
print(f"Abstention rate   : {metrics['abstention_rate']}")
print(f"Mean latency      : {metrics['mean_latency_ms']} ms")
print("---------------------------------------------")