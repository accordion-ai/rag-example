from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os, glob, uuid

QDRANT_HOST = "qdrant"      # 같은 네임스페이스 서비스명
COLLECTION  = "rag_docs"
EMBED_MODEL = "BAAI/bge-m3"  # 다국어 임베딩 모델

client = QdrantClient(host=QDRANT_HOST, port=6333)
embedder = SentenceTransformer(EMBED_MODEL)

# 컬렉션 생성 (이미 있으면 skip)
if COLLECTION not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )

# 문서 청크 분할 및 임베딩
points = []
for filepath in glob.glob("docs/**/*", recursive=True):
    text = open(filepath).read() if filepath.endswith(".txt") else extract_pdf(filepath)
    chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # 500자, 100자 오버랩
    for chunk in chunks:
        vec = embedder.encode(chunk).tolist()
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vec,
                                        payload={"text": chunk, "source": filepath}))

client.upsert(collection_name=COLLECTION, points=points)
print(f"총 {len(points)}개 청크 인덱싱 완료")
