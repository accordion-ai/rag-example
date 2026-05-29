# ~/rag-chatbot/query.py
# ingest.py 로 인덱싱한 뒤 검색이 잘 되는지 확인하는 예제
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
 
client = QdrantClient(host="qdrant", port=6333)
embedder = SentenceTransformer("BAAI/bge-m3")
 
QUESTIONS = [
    "원두는 어디에 보관하는 게 좋아?",
    "커피가 너무 쓴데 어떻게 해결해?",
    "핸드드립 물 온도는 몇 도가 적당해?",
]
 
for q in QUESTIONS:
    vec = embedder.encode(q).tolist()
    hits = client.search(collection_name="rag_docs", query_vector=vec, limit=2)
    print(f"\n[질문] {q}")
    for h in hits:
        print(f"  score={h.score:.3f} | {h.payload['source']}")
        print(f"  {h.payload['text'][:120]}...")