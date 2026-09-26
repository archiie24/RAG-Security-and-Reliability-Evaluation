from pathlib import Path
import requests
from app.rag_core import PDF_DIR, PDF_SOURCES

PDF_DIR.mkdir(exist_ok=True)
for name,url in PDF_SOURCES.items():
    path=PDF_DIR/name
    if path.exists() and path.stat().st_size>10000:
        print('exists:',name); continue
    print('downloading:',name)
    r=requests.get(url,timeout=60,headers={'User-Agent':'secure-rag-project/1.0'})
    r.raise_for_status(); path.write_bytes(r.content)
    print('saved:',path)
