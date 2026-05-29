from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
 
# ── 설정 ──────────────────────────────────────────────
VLLM_BASE_URL = "http://vllm:8000/v1"   # vLLM 서비스명:포트
QDRANT_HOST   = "qdrant"                 # Qdrant 서비스명
QDRANT_PORT   = 6333
COLLECTION    = "rag_docs"
EMBED_MODEL   = "BAAI/bge-m3"
TOP_K         = 3                        # 검색할 관련 문서 수
SCORE_THRESHOLD = 0.4                    # 이 점수 미만이면 관련 문서 없음으로 간주
 
SYSTEM_PROMPT = (
    "당신은 주어진 참고 문서를 바탕으로 정확하게 답하는 AI 어시스턴트입니다. "
    "문서에 없는 내용은 추측하지 말고 모른다고 답하세요."
)
 
# ── 클라이언트 초기화 (지연 초기화) ──────────────────
_llm = None
_qdrant = None
_embedder = None
_model_name = None
 
 
def init_clients():
    """클라이언트와 모델명을 최초 1회만 초기화한다."""
    global _llm, _qdrant, _embedder, _model_name
    if _llm is not None:
        return
    _llm = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")
    _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"[init] 임베딩 모델 로딩 중: {EMBED_MODEL} ...")
    _embedder = SentenceTransformer(EMBED_MODEL)
    _model_name = _llm.models.list().data[0].id
    print(f"[init] 완료. LLM 모델: {_model_name}\n")
 
 
def retrieve(query: str) -> list[dict]:
    """질문과 유사한 문서 청크 TOP_K개 검색 (점수/출처 포함)"""
    vec = _embedder.encode(query, normalize_embeddings=True).tolist()
    result = _qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=TOP_K,
        with_payload=True,
    )
    hits = []
    for p in result.points:
        if p.score < SCORE_THRESHOLD:
            continue
        hits.append({
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", "unknown"),
            "score": p.score,
        })
    return hits
 
 
def build_prompt(query: str, contexts: list[dict]) -> str:
    """검색된 문서를 컨텍스트로 프롬프트 구성 (출처 표기 포함)"""
    blocks = [f"[출처: {c['source']}]\n{c['text']}" for c in contexts]
    ctx = "\n\n---\n\n".join(blocks)
    return f"""다음 참고 문서를 바탕으로 질문에 답하세요.
 
참고 문서:
{ctx}
 
질문: {query}
답변:"""
 
 
def rag_answer(query: str, history: list) -> str:
    """RAG 파이프라인: 검색 → 프롬프트 구성 → LLM 생성"""
    contexts = retrieve(query)
    if not contexts:
        return "참고 문서에서 관련 내용을 찾을 수 없습니다."
 
    prompt = build_prompt(query, contexts)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)                       # 이전 대화 맥락 유지
    messages.append({"role": "user", "content": prompt})
 
    response = _llm.chat.completions.create(
        model=_model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=False,
    )
    answer = response.choices[0].message.content
 
    # 답변 하단에 참고한 출처 표시
    sources = ", ".join(sorted({c["source"] for c in contexts}))
    return f"{answer}\n\n(참고: {sources})"
 
 
def main():
    init_clients()
    print("=" * 50)
    print(" RAG 챗봇 (종료: exit / quit / 빈 줄)")
    print("=" * 50)
 
    history = []  # LLM에 넘길 대화 기록 (원본 질문/답변 기준)
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
 
        if query.lower() in ("exit", "quit", ""):
            print("종료합니다.")
            break
 
        try:
            answer = rag_answer(query, history)
        except Exception as e:
            print(f"[오류] {e}")
            continue
 
        print(f"\n답변> {answer}")
 
        # 다음 턴을 위해 맥락 누적 (프롬프트가 아닌 원 질문/답변으로 저장)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
 
 
if __name__ == "__main__":
    main()