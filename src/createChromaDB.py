import json
import os
import uuid  # ✅ ID 중복 완벽 방지를 위한 라이브러리
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from google.colab import drive

# ── 0. 구글 드라이브 마운트 (저장을 위해 미리 연결) ──
print("구글 드라이브를 연결합니다...")
drive.mount('/content/drive')

# ── 1. 원천데이터 로드 ──
base = '/content/dermatology-rag-chatbot/data/raw'
target_folders = [
    'TS_국문_의학 교과서',
    'TS_국문_온라인 의료 정보 제공 사이트',
    'TS_국문_학회 가이드라인',
]

all_contents = []
for folder in target_folders:
    path = os.path.join(base, folder)
    if not os.path.exists(path):
        print(f"❌ 없음: {folder}")
        continue
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    for f in files:
        try:
            with open(os.path.join(path, f), 'r', encoding='utf-8-sig') as fp:
                data = json.load(fp)
                content = data.get('content', '').strip()
                if len(content) > 50:
                    all_contents.append({
                        'content': content,
                        'source': folder,
                        'c_id': data.get('c_id', ''),
                    })
        except:
            pass

print(f"\n✅ 로드 완료: {len(all_contents)}개 문서")

# ── 2. Chunking 및 안전한 ID 생성 ──
chunks = []
chunk_ids = []
chunk_metas = []

for doc in all_contents:
    text = doc['content']
    # 500자씩 자르고 50자 오버랩 (i가 450씩 증가)
    for i in range(0, len(text), 450):
        chunk = text[i:i+500]
        if len(chunk) > 50:
            # ✅ 수정됨: 기존 f"{doc['c_id']}_{i}" 대신 무조건 겹치지 않는 uuid 사용
            chunk_id = str(uuid.uuid4()) 
            
            chunks.append(chunk)
            chunk_ids.append(chunk_id)
            chunk_metas.append({
                'source': doc['source'],
                'c_id': doc['c_id'],
            })

print(f"✅ Chunking 완료: {len(chunks)}개 조각")

# ── 3. 임베딩 모델 로드 ──
print("\n임베딩 모델 로드 중...")
embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# ── 4. ChromaDB 생성 ──
print("ChromaDB 생성 중...")
local_db_path = '/content/dermatology-rag-chatbot/vectordb/chroma'
client = chromadb.PersistentClient(path=local_db_path)

# 기존에 같은 이름의 컬렉션이 있으면 지우고 새로 만듦 (초기화)
try:
    client.delete_collection("medical_knowledge")
except:
    pass

collection = client.create_collection(
    name="medical_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# ── 5. 배치로 임베딩 + 저장 ──
print("\n데이터 임베딩 및 DB 저장 시작 (시간이 조금 걸립니다)...")
batch_size = 500
for i in range(0, len(chunks), batch_size):
    end = min(i + batch_size, len(chunks))
    batch_texts = chunks[i:end]
    batch_ids = chunk_ids[i:end]
    batch_metas = chunk_metas[i:end]
    
    batch_embeddings = embed_model.encode(batch_texts).tolist()
    
    collection.add(
        documents=batch_texts,
        embeddings=batch_embeddings,
        ids=batch_ids,
        metadatas=batch_metas,
    )
    
    print(f"  {end}/{len(chunks)} 저장 완료 ({end/len(chunks)*100:.0f}%)")

print(f"\n✅ ChromaDB 임시 구축 완료! (총 {collection.count()}개 chunk)")

# ── 6. 드라이브로 영구 백업 (핵심 로직) ──
print("\n💾 구글 드라이브로 DB 폴더를 안전하게 백업합니다...")
drive_save_path = '/content/drive/MyDrive/medical_rag_db/chroma'

# 만약 드라이브에 예전 폴더가 남아있다면 덮어쓰기를 위해 삭제
if os.path.exists(drive_save_path):
    shutil.rmtree(drive_save_path)

# ChromaDB 전체 폴더 복사
shutil.copytree(local_db_path, drive_save_path)
print(f"✅ 백업 완료! 이제 코랩이 꺼져도 드라이브({drive_save_path})에서 불러올 수 있습니다.")

# ── 7. 검색 테스트 ──
print("\n── 🔍 검색 테스트 ──")
test_query = "대상포진의 치료 방법"
query_embedding = embed_model.encode([test_query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
)

for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n[결과 {i+1}] 출처: {meta['source']}")
    print(f"  {doc[:150]}...")