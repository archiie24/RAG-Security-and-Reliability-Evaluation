from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag_core import SecureRAG

import uuid
import numpy as np


app = FastAPI(
    title="Secure RAG Evaluation API",
    version="1.0"
)


rag = None


class QueryRequest(BaseModel):

    query: str

    top_k: int = 8


# ---------------------------------------------------------
# Convert NumPy values into normal Python JSON values
# ---------------------------------------------------------

def make_json_safe(obj):

    if isinstance(obj, dict):

        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }


    if isinstance(obj, list):

        return [
            make_json_safe(v)
            for v in obj
        ]


    if isinstance(obj, tuple):

        return [
            make_json_safe(v)
            for v in obj
        ]


    if isinstance(obj, np.bool_):

        return bool(obj)


    if isinstance(obj, np.integer):

        return int(obj)


    if isinstance(obj, np.floating):

        return float(obj)


    if isinstance(obj, np.ndarray):

        return obj.tolist()


    return obj


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup():

    global rag

    rag = SecureRAG()


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "loaded": rag is not None
    }


# ---------------------------------------------------------
# Query
# ---------------------------------------------------------

@app.post("/query")
def query(req: QueryRequest):

    if rag is None:

        raise HTTPException(
            status_code=503,
            detail="RAG not initialized"
        )


    result = rag.query(
        req.query,
        top_k=max(
            1,
            min(
                req.top_k,
                20
            )
        )
    )


    # Generate unique ID for dashboard logging

    run_id = str(
        uuid.uuid4()
    )


    # Log query

    rag.log_query(
        run_id=run_id,
        result=result,
        config="hardened"
    )


    # Don't expose full document text through API

    result["retrieved"] = [
        {
            k: v
            for k, v in x.items()
            if k != "text"
        }
        for x in result.get(
            "retrieved",
            []
        )
    ]


    result["run_id"] = run_id


    # Convert NumPy values to JSON-safe Python values

    result = make_json_safe(
        result
    )


    return result