from app.rag_core import SecureRAG
import pandas as pd

CASES=[
('REL01','What is SQL injection?',['sql','inject']),
('REL02','What are the types of cross-site scripting?',['stored','dom']),
('REL03','How do you test for session fixation?',['session','fixation']),
('REL04','What is clickjacking?',['clickjack']),
('REL05','How should an incident response team handle an incident?',['incident','contain']),
('REL06','What is the current stock price of Apple?',[]),
('REL07','What is the weather in Bengaluru today?',[]),
]

def run():
    rag=SecureRAG(); rows=[]
    for rid,q,topics in CASES:
        r=rag.query(q); text=' '.join(c['text'].lower() for c in r['retrieved'])
        passed=(any(x in text for x in topics) if topics else r['abstained'])
        rows.append({'id':rid,'question':q,'pass':int(passed),'abstained':int(r['abstained']),'evidence_score':r['evidence_score'],'groundedness':r['groundedness'],'unsupported_claim_rate':r['unsupported_claim_rate'],'latency_ms':r.get('latency_ms',0)})
    return pd.DataFrame(rows)

if __name__=='__main__': print(run().to_string(index=False))
