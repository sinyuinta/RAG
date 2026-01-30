"""
チャンクをEmbedding化してChromaDBに格納するスクリプト
"""

import json
import os
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# 設定
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "psychology_theories"
CHROMA_PERSIST_DIR = "./data/chromadb"

def load_chunks(filepath: Path) -> list[dict]:
    """JSONからチャンクを読み込み"""
    return json.loads(filepath.read_text(encoding='utf-8'))

def create_embedding(client: OpenAI, text: str) -> list[float]:
    """OpenAI APIでEmbedding生成"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def build_context_prefix(chunk: dict) -> str:
    """Contextual Retrieval用の文脈プレフィックスを生成"""
    return (
        f"[文脈] このチャンクは「{chunk['ancestor']}」学派の"
        f"「{chunk['source_author']}」による理論文書の"
        f"「{chunk['section_title']}」セクションです。"
        f"{chunk['family_line']}の{chunk['role']}として、"
        f"{chunk['chunk_type']}に関する内容を含みます。\n\n"
        f"[本文]\n"
    )

def build_vectordb(chunks_path: Path):
    """ChromaDBを構築"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment variables.")
        return

    # OpenAIクライアント初期化
    client = OpenAI(api_key=api_key)
    
    # ChromaDBクライアント初期化（永続化）
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # コレクション作成（既存なら削除して再作成）
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # コサイン類似度
    )
    
    # チャンク読み込み
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        return

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")
    
    # バッチ処理
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in batch:
            # 文脈プレフィックス + 本文でEmbedding
            context_prefix = build_context_prefix(chunk)
            full_text = context_prefix + chunk['content']
            
            try:
                embedding = create_embedding(client, full_text)
                
                ids.append(chunk['id'])
                embeddings.append(embedding)
                documents.append(chunk['content'])  # 保存は本文のみ
                metadatas.append({
                    'ancestor': chunk['ancestor'],
                    'family_line': chunk['family_line'],
                    'source_author': chunk['source_author'],
                    'role': chunk['role'],
                    'level': chunk['level'],
                    'chunk_type': chunk['chunk_type'],
                    'section_title': chunk['section_title'],
                    'source_file': chunk['source_file']
                })
            except Exception as e:
                print(f"Error generating embedding for chunk {chunk['id']}: {e}")
        
        if ids:
            # ChromaDBに追加
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)}")
    
    print(f"\nVectorDB built: {CHROMA_PERSIST_DIR}")
    print(f"Collection '{COLLECTION_NAME}' contains {collection.count()} vectors")

if __name__ == "__main__":
    chunks_path = Path("./data/chunks.json")
    build_vectordb(chunks_path)
