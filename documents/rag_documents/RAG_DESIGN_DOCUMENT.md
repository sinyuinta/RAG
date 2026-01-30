# 心理学理論RAGシステム 設計書

## 概要

本システムは、心理学理論文書（4学派・19文書）をベクトルデータベースに格納し、ユーザーの質問に対して関連する理論を検索・提示するRAG（Retrieval-Augmented Generation）システムである。

### 本フェーズのユースケース

| 項目 | 内容 |
|------|------|
| **ユーザー** | 臨床家（カウンセラー、心理士など） |
| **利用目的** | ケースに対してどの理論が適用できるかを検討する |
| **質問タイプ** | 「このケースにどの理論枠組みで理解できるか」「この介入の理論的根拠は」など |
| **期待する応答** | 複数の理論的視点の提示、実践上の注意点、理論の限界 |

### 将来のフェーズ（参考）

- 事業展開時に「思考アスレチック」のミニロボへの組み込みを想定
- そのため、本システムは拡張性を考慮した設計とする

---

## 1. アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────────┐
│                        ユーザー                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     フロントエンド                               │
│                   （React / Next.js）                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP API
┌─────────────────────────────────────────────────────────────────┐
│                      バックエンド                                │
│                  （Python / FastAPI）                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    RAG Pipeline                           │  │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐ │  │
│  │  │  Query  │───▶│Embedding│───▶│ Search  │───▶│ Prompt │ │  │
│  │  │ Process │    │ (OpenAI)│    │(ChromaDB)│   │ Build  │ │  │
│  │  └─────────┘    └─────────┘    └─────────┘    └────────┘ │  │
│  │                                                     │     │  │
│  │                                                     ▼     │  │
│  │                                              ┌──────────┐ │  │
│  │                                              │ GPT-4o   │ │  │
│  │                                              │ (OpenAI) │ │  │
│  │                                              └──────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────┐          ┌──────────────────────┐
│      ChromaDB        │          │     OpenAI API       │
│   （ベクトルDB）      │          │ ・Embedding          │
│   ・永続化ストレージ   │          │ ・Chat Completion    │
└──────────────────────┘          └──────────────────────┘
```

---

## 2. 技術スタック

| レイヤー | 技術 | 理由 |
|---------|------|------|
| **Embedding** | OpenAI `text-embedding-3-small` | 高品質、低コスト、日本語対応 |
| **Vector DB** | ChromaDB | 無料、シンプル、永続化対応、メタデータフィルタ可能 |
| **LLM** | GPT-4o | 指定要件 |
| **Backend** | Python 3.11 + FastAPI | 高速、型安全、OpenAPI自動生成 |
| **Frontend** | React / Next.js（任意） | 学生の得意技術で可 |
| **Deploy** | Railway / Render / Fly.io | 無料枠あり、簡易デプロイ |

### 依存パッケージ

```txt
# requirements.txt
fastapi==0.109.0
uvicorn==0.27.0
openai==1.12.0
chromadb==0.4.22
python-dotenv==1.0.0
pydantic==2.5.3
```

---

## 3. データパイプライン

### 3.1 全体フロー

```
Markdown文書（19本）
       │
       ▼
┌─────────────────┐
│ 1. チャンク分割  │  ← CHUNKING_GUIDE.md に従う
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ 2. メタデータ    │  ← 各チャンクに学派・著者情報を付与
│    抽出・付与    │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ 3. Embedding    │  ← OpenAI API
│    生成         │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ 4. ChromaDBに   │  ← ベクトル + メタデータを永続化
│    格納         │
└─────────────────┘
```

### 3.2 チャンク分割スクリプト

```python
# scripts/chunk_documents.py
"""
Markdown文書をチャンクに分割するスクリプト
CHUNKING_GUIDE.md のルールに従う
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Chunk:
    """1つのチャンクを表すデータクラス"""
    id: str                    # 一意のID（例: adler_core_001）
    content: str               # チャンク本文
    ancestor: str              # 始祖（Adler, Freud, Jung, Neisser）
    family_line: str           # 学派
    source_author: str         # 著者
    role: str                  # ancestor_core / descendant_extension
    level: str                 # foundation / applied
    chunk_type: str            # premise / core_concept / structure / application / boundary / caution / references
    section_title: str         # セクション名
    source_file: str           # 元ファイル名

def extract_metadata(content: str) -> dict:
    """Markdownファイルからメタデータセクションを抽出"""
    metadata = {}
    meta_match = re.search(r'## メタデータ\n(.*?)\n---', content, re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('- ', '')
                metadata[key] = value.strip()
    return metadata

def determine_chunk_type(section_title: str) -> str:
    """セクションタイトルからchunk_typeを決定"""
    title_lower = section_title.lower()
    if '基本前提' in section_title:
        return 'premise'
    elif '中核概念' in section_title:
        return 'core_concept'
    elif '理論構造' in section_title:
        return 'structure'
    elif '力点' in section_title or '説明' in section_title or '接続' in section_title:
        return 'application'
    elif '誤解' in section_title or '境界' in section_title:
        return 'boundary'
    elif '留意' in section_title or '三角コーン' in section_title:
        return 'caution'
    elif '文献' in section_title:
        return 'references'
    else:
        return 'core_concept'  # デフォルト

def split_by_headings(content: str, level: int = 2) -> list[tuple[str, str]]:
    """
    指定レベルの見出しで分割
    Returns: [(section_title, section_content), ...]
    """
    pattern = r'^(#{' + str(level) + r'})\s+(.+)$'
    sections = []
    current_title = ""
    current_content = []
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            if current_title or current_content:
                sections.append((current_title, '\n'.join(current_content)))
            current_title = match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    if current_title or current_content:
        sections.append((current_title, '\n'.join(current_content)))
    
    return sections

def chunk_document(filepath: Path) -> list[Chunk]:
    """
    1つのMarkdownファイルをチャンクに分割
    """
    content = filepath.read_text(encoding='utf-8')
    filename = filepath.stem
    metadata = extract_metadata(content)
    
    chunks = []
    chunk_counter = 0
    
    # レベル2見出し（##）で大分割
    level2_sections = split_by_headings(content, level=2)
    
    for section_title, section_content in level2_sections:
        # メタデータセクションはスキップ
        if 'メタデータ' in section_title:
            continue
        
        # 「理論の骨格」セクションはレベル3で細分割
        if '理論の骨格' in section_title:
            level3_sections = split_by_headings(section_content, level=3)
            for sub_title, sub_content in level3_sections:
                if not sub_title:
                    continue
                
                # 中核概念がさらに長い場合はレベル4で分割
                if '中核概念' in sub_title and len(sub_content) > 6000:
                    # **（1）見出し** パターンで分割
                    concept_pattern = r'\*\*（\d+）(.+?）\*\*'
                    concept_sections = re.split(concept_pattern, sub_content)
                    
                    # 分割結果を処理
                    i = 0
                    while i < len(concept_sections):
                        if i + 1 < len(concept_sections) and re.match(r'.+', concept_sections[i]):
                            concept_title = concept_sections[i]
                            concept_content = concept_sections[i + 1] if i + 1 < len(concept_sections) else ""
                            
                            chunk_counter += 1
                            full_title = f"{sub_title} - {concept_title}"
                            chunks.append(Chunk(
                                id=f"{filename}_{chunk_counter:03d}",
                                content=f"**（{concept_title}）**\n{concept_content}".strip(),
                                ancestor=metadata.get('ancestor', ''),
                                family_line=metadata.get('family_line', ''),
                                source_author=metadata.get('source_author', ''),
                                role=metadata.get('role', ''),
                                level=metadata.get('level', ''),
                                chunk_type=determine_chunk_type(full_title),
                                section_title=full_title,
                                source_file=filename
                            ))
                            i += 2
                        else:
                            i += 1
                else:
                    # 通常のレベル3セクション
                    chunk_counter += 1
                    chunks.append(Chunk(
                        id=f"{filename}_{chunk_counter:03d}",
                        content=sub_content.strip(),
                        ancestor=metadata.get('ancestor', ''),
                        family_line=metadata.get('family_line', ''),
                        source_author=metadata.get('source_author', ''),
                        role=metadata.get('role', ''),
                        level=metadata.get('level', ''),
                        chunk_type=determine_chunk_type(sub_title),
                        section_title=sub_title,
                        source_file=filename
                    ))
        else:
            # 他のレベル2セクション（力点、誤解、留意点など）
            # レベル3があれば分割、なければそのまま
            level3_sections = split_by_headings(section_content, level=3)
            
            if len(level3_sections) > 1:
                for sub_title, sub_content in level3_sections:
                    if not sub_title and not sub_content.strip():
                        continue
                    chunk_counter += 1
                    full_title = f"{section_title} - {sub_title}" if sub_title else section_title
                    chunks.append(Chunk(
                        id=f"{filename}_{chunk_counter:03d}",
                        content=sub_content.strip(),
                        ancestor=metadata.get('ancestor', ''),
                        family_line=metadata.get('family_line', ''),
                        source_author=metadata.get('source_author', ''),
                        role=metadata.get('role', ''),
                        level=metadata.get('level', ''),
                        chunk_type=determine_chunk_type(full_title),
                        section_title=full_title,
                        source_file=filename
                    ))
            else:
                chunk_counter += 1
                chunks.append(Chunk(
                    id=f"{filename}_{chunk_counter:03d}",
                    content=section_content.strip(),
                    ancestor=metadata.get('ancestor', ''),
                    family_line=metadata.get('family_line', ''),
                    source_author=metadata.get('source_author', ''),
                    role=metadata.get('role', ''),
                    level=metadata.get('level', ''),
                    chunk_type=determine_chunk_type(section_title),
                    section_title=section_title,
                    source_file=filename
                ))
    
    return chunks

def process_all_documents(input_dir: Path, output_path: Path):
    """全文書を処理してJSONに出力"""
    all_chunks = []
    
    for filepath in sorted(input_dir.glob('*.md')):
        if filepath.name.startswith('CHUNKING'):  # ガイドは除外
            continue
        print(f"Processing: {filepath.name}")
        chunks = chunk_document(filepath)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")
    
    # JSON出力
    output_data = [asdict(chunk) for chunk in all_chunks]
    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nTotal: {len(all_chunks)} chunks -> {output_path}")

if __name__ == "__main__":
    input_dir = Path("./rag_documents")
    output_path = Path("./data/chunks.json")
    output_path.parent.mkdir(exist_ok=True)
    process_all_documents(input_dir, output_path)
```

### 3.3 Embedding生成 & ChromaDB格納スクリプト

```python
# scripts/build_vectordb.py
"""
チャンクをEmbedding化してChromaDBに格納するスクリプト
"""

import json
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.config import Settings

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
    # OpenAIクライアント初期化
    client = OpenAI()  # OPENAI_API_KEY環境変数を使用
    
    # ChromaDBクライアント初期化（永続化）
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # コレクション作成（既存なら削除して再作成）
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # コサイン類似度
    )
    
    # チャンク読み込み
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
```

---

## 4. 検索・応答フロー

### 4.1 検索サービス

```python
# app/services/retrieval.py
"""
検索サービス：クエリに関連するチャンクを取得
"""

from openai import OpenAI
import chromadb
from typing import Optional

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "psychology_theories"
CHROMA_PERSIST_DIR = "./data/chromadb"

class RetrievalService:
    def __init__(self):
        self.openai_client = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
    
    def embed_query(self, query: str) -> list[float]:
        """クエリをEmbedding化"""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        ancestor_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None
    ) -> list[dict]:
        """
        ベクトル検索を実行
        
        Args:
            query: 検索クエリ
            n_results: 取得件数
            ancestor_filter: 学派フィルタ（Adler, Freud, Jung, Neisser）
            chunk_type_filter: チャンクタイプフィルタ
        
        Returns:
            検索結果のリスト
        """
        # クエリをEmbedding化
        query_embedding = self.embed_query(query)
        
        # メタデータフィルタ構築
        where_filter = None
        if ancestor_filter or chunk_type_filter:
            conditions = []
            if ancestor_filter:
                conditions.append({"ancestor": ancestor_filter})
            if chunk_type_filter:
                conditions.append({"chunk_type": chunk_type_filter})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # 検索実行
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # 結果を整形
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]  # 小さいほど類似
            })
        
        return formatted_results
```

### 4.2 応答生成サービス

```python
# app/services/generation.py
"""
応答生成サービス：検索結果をもとにGPTで応答を生成
"""

from openai import OpenAI
from typing import Optional

class GenerationService:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4o"
    
    def build_system_prompt(self) -> str:
        """システムプロンプトを構築"""
        return """あなたは心理学理論の専門家アシスタントです。
利用者は臨床家（カウンセラー、心理士など）であり、ケースに対してどの理論が適用できるかを検討するために使用しています。

以下のルールに従って回答してください：

1. **必ず提供された文書を根拠として回答する**
   - 文書にない情報で回答しない
   - 引用元を明示する（例：「アドラーの理論によると...」「Dreikursはこれを〜と呼んだ」）

2. **臨床的適用を意識した回答をする**
   - 「この理論ではこう理解できる」という形で理論と実践を接続する
   - 単なる理論説明ではなく、ケースへの適用可能性を示す

3. **複数の理論的視点を提示する**
   - 一つの現象を複数の学派から見るとどうなるかを並列で示す
   - 例：「アドラー的には〜と理解できる。一方フロイト的には〜」

4. **実践上の注意点を必ず含める**
   - 「誤解されやすい点」「実践での留意点」の内容を積極的に伝える
   - 理論の誤用・乱用のリスクを指摘する

5. **理論の限界を正直に示す**
   - 各理論が説明できないこと、適用困難な場面を伝える
   - 「この理論だけでは不十分な可能性がある」と言える

6. **文書に情報がない場合は正直に答える**
   - 「この文書には該当する情報がありません」
   - 推測や一般知識で補わない
"""
    
    def build_context(self, search_results: list[dict]) -> str:
        """検索結果からコンテキストを構築"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            meta = result['metadata']
            context_parts.append(
                f"【参考文献{i}】\n"
                f"学派: {meta['ancestor']} ({meta['family_line']})\n"
                f"著者: {meta['source_author']}\n"
                f"セクション: {meta['section_title']}\n"
                f"内容:\n{result['content']}\n"
            )
        return "\n---\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        search_results: list[dict],
        conversation_history: Optional[list[dict]] = None
    ) -> str:
        """
        応答を生成
        
        Args:
            query: ユーザーの質問
            search_results: 検索結果
            conversation_history: 会話履歴（オプション）
        
        Returns:
            生成された応答
        """
        # メッセージ構築
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        
        # 会話履歴があれば追加
        if conversation_history:
            messages.extend(conversation_history)
        
        # コンテキスト + クエリ
        context = self.build_context(search_results)
        user_message = f"""以下の参考文献をもとに質問に回答してください。

{context}

---
質問: {query}
"""
        messages.append({"role": "user", "content": user_message})
        
        # GPT呼び出し
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # 事実に基づく回答のため低め
            max_tokens=2000
        )
        
        return response.choices[0].message.content
```

### 4.3 FastAPI エンドポイント

```python
# app/main.py
"""
FastAPI アプリケーション
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.services.retrieval import RetrievalService
from app.services.generation import GenerationService

app = FastAPI(
    title="Psychology Theory RAG API",
    description="心理学理論に関する質問応答API",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# サービス初期化
retrieval_service = RetrievalService()
generation_service = GenerationService()


class QueryRequest(BaseModel):
    """質問リクエスト"""
    query: str
    n_results: int = 5
    ancestor_filter: Optional[str] = None  # Adler, Freud, Jung, Neisser
    chunk_type_filter: Optional[str] = None


class SearchResult(BaseModel):
    """検索結果"""
    id: str
    content: str
    metadata: dict
    distance: float


class QueryResponse(BaseModel):
    """質問応答レスポンス"""
    answer: str
    sources: list[SearchResult]


class SearchRequest(BaseModel):
    """検索のみリクエスト"""
    query: str
    n_results: int = 5
    ancestor_filter: Optional[str] = None
    chunk_type_filter: Optional[str] = None


@app.get("/")
async def root():
    """ヘルスチェック"""
    return {"status": "ok", "message": "Psychology Theory RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    質問応答エンドポイント
    
    検索 + GPT生成を一括で実行
    """
    try:
        # 1. 検索
        search_results = retrieval_service.search(
            query=request.query,
            n_results=request.n_results,
            ancestor_filter=request.ancestor_filter,
            chunk_type_filter=request.chunk_type_filter
        )
        
        if not search_results:
            return QueryResponse(
                answer="該当する情報が見つかりませんでした。",
                sources=[]
            )
        
        # 2. 応答生成
        answer = generation_service.generate(
            query=request.query,
            search_results=search_results
        )
        
        return QueryResponse(
            answer=answer,
            sources=[SearchResult(**r) for r in search_results]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    """
    検索のみエンドポイント
    
    デバッグ・評価用
    """
    try:
        results = retrieval_service.search(
            query=request.query,
            n_results=request.n_results,
            ancestor_filter=request.ancestor_filter,
            chunk_type_filter=request.chunk_type_filter
        )
        return [SearchResult(**r) for r in results]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """コレクション統計"""
    count = retrieval_service.collection.count()
    return {
        "total_chunks": count,
        "collection_name": retrieval_service.collection.name
    }
```

---

## 5. ディレクトリ構成

```
psychology-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPIアプリ
│   └── services/
│       ├── __init__.py
│       ├── retrieval.py        # 検索サービス
│       └── generation.py       # 応答生成サービス
├── scripts/
│   ├── chunk_documents.py      # チャンク分割
│   └── build_vectordb.py       # VectorDB構築
├── rag_documents/              # 心理学文書（19本 + ガイド）
│   ├── CHUNKING_GUIDE.md
│   ├── adler_core.md
│   ├── adler_dreikurs.md
│   └── ...
├── data/
│   ├── chunks.json             # 分割されたチャンク
│   └── chromadb/               # ChromaDB永続化ディレクトリ
├── requirements.txt
├── .env                        # 環境変数（OPENAI_API_KEY）
├── .gitignore
└── README.md
```

---

## 6. デプロイ構成

### 6.1 環境変数

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
```

### 6.2 起動コマンド

```bash
# 開発
uvicorn app.main:app --reload --port 8000

# 本番
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6.3 Railway / Render へのデプロイ

**Railway の場合：**

1. GitHubリポジトリを接続
2. 環境変数に `OPENAI_API_KEY` を設定
3. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**注意：** ChromaDBの永続化ディレクトリはデプロイごとにリセットされる可能性があるため、本番運用時は以下のいずれかを検討：

- デプロイ時に毎回 `build_vectordb.py` を実行する
- ChromaDB Cloud を使用する（有料）
- Pinecone などの外部Vector DBに切り替える

---

## 7. 評価・テスト方法

### 7.1 検索品質の評価

```python
# tests/test_retrieval.py
"""
検索品質のテスト
"""

TEST_QUERIES = [
    {
        "query": "劣等感とは何か",
        "expected_ancestor": "Adler",
        "expected_chunk_types": ["core_concept", "premise"]
    },
    {
        "query": "防衛機制の種類を教えて",
        "expected_ancestor": "Freud",
        "expected_chunk_types": ["core_concept"]
    },
    {
        "query": "ワーキングメモリの容量制限",
        "expected_ancestor": "Neisser",
        "expected_chunk_types": ["core_concept", "premise"]
    },
    {
        "query": "元型とは何か、スピリチュアルとの違いは",
        "expected_ancestor": "Jung",
        "expected_chunk_types": ["boundary", "core_concept"]
    },
]

def test_retrieval_accuracy(retrieval_service):
    """検索精度テスト"""
    for test_case in TEST_QUERIES:
        results = retrieval_service.search(test_case["query"], n_results=3)
        
        # 最上位結果の学派が期待通りか
        top_result = results[0]
        assert top_result['metadata']['ancestor'] == test_case['expected_ancestor'], \
            f"Query: {test_case['query']}, Expected: {test_case['expected_ancestor']}, Got: {top_result['metadata']['ancestor']}"
        
        print(f"✓ {test_case['query'][:30]}...")
```

### 7.2 応答品質の手動評価チェックリスト（臨床家向けRAG）

| 評価項目 | 確認内容 |
|----------|----------|
| **根拠性** | 回答が提供された文書に基づいているか |
| **正確性** | 理論の内容が正しく引用されているか |
| **引用明示** | どの学派・著者の見解かが明示されているか |
| **臨床接続性** | 理論がケースへの適用という形で説明されているか |
| **複数視点** | 複数の学派からの見方が提示されているか（該当する場合） |
| **注意点の提示** | 「誤解されやすい点」「実践での留意点」が反映されているか |
| **限界の明示** | 理論の適用限界や誤用リスクが示されているか |
| **過度な断定の回避** | 「〜と理解できる」「〜の観点からは」など、一つの見方として提示しているか |
| **不明時の対応** | 文書にない情報を求められた際に適切に断っているか |

### 7.3 テストクエリ例（臨床家向け）

```
【ケースの理論的理解】
- クライエントが他者の評価を過度に気にして行動できなくなっている。
  どの理論枠組みで理解できるか？
  → 複数学派からの視点が提示されるべき（アドラー：劣等コンプレックス、
    ホーナイ：基本的不安と迎合傾向、など）

- 40代男性が「自分の人生は何だったのか」と空虚感を訴えている。
  どの理論が参考になるか？
  → エリクソン（生殖性vs停滞）、ユング（人生後半の個性化）が検索されるべき

【アプローチの比較】
- 引きこもりのケースに対して、アドラー的アプローチとフロイト的アプローチでは
  どう見立てが異なるか？
  → アドラー（目的論、人生課題からの回避）vs フロイト（退行、防衛）の対比

- 完璧主義的なクライエントへのアプローチを学派ごとに教えてください
  → ホーナイ（べき思考）、アドラー（優越への努力の歪み）、
    認知心理学（メタ認知の問題）など

【介入の理論的根拠】
- 「勇気づけ」という介入の理論的背景と、実施上の注意点は？
  → アドラー/ドライカースからの検索、「褒めるとの違い」「留意点」が含まれるべき

- 早期回想を使ったアセスメントの根拠と限界は？
  → アドラー/モサックからの検索、「解釈を断定しない」注意点が含まれるべき

【理論の誤用防止】
- クライエントの症状に「目的がある」と伝えたいが、責任追及にならないか心配だ
  → 「実践での留意点」から「目的論を責任追及に使わない」が検索されるべき

- 「あなたは共同体感覚が足りない」とクライエントに伝えてよいか？
  → 「誤解されやすい点」「留意点」から、道徳的説教にしないという内容が出るべき

【理論の適用限界】
- 認知負荷理論はどのような臨床場面には使えないか？
  → 「理論の適用範囲」「限界」に関する内容が検索されるべき

- ユングの元型概念を使って解釈するとき、どこまでが心理学でどこからがスピリチュアルか？
  → 「誤解されやすい点」から境界に関する内容が検索されるべき

【文書外の質問（拒否確認）】
- ACT（アクセプタンス＆コミットメント・セラピー）について教えてください
  → 「この文書には該当する情報がありません」と回答すべき
```

---

## 8. 実装の順序（推奨）

```
Week 1:
  1. 環境構築（Python, 依存パッケージ）
  2. chunk_documents.py の実装・実行
  3. chunks.json の確認

Week 2:
  4. build_vectordb.py の実装・実行
  5. ChromaDB の動作確認

Week 3:
  6. retrieval.py の実装
  7. 検索のテスト（test_retrieval.py）

Week 4:
  8. generation.py の実装
  9. main.py（FastAPI）の実装
  10. ローカルでの動作確認

Week 5:
  11. デプロイ
  12. 評価・調整
```

---

## 9. トラブルシューティング

| 問題 | 原因 | 対処 |
|------|------|------|
| Embedding生成が遅い | APIレート制限 | バッチサイズを小さくする、待機を入れる |
| 検索結果が的外れ | チャンク分割が不適切 | CHUNKING_GUIDE.md を再確認、手動で分割調整 |
| GPT応答が文書を無視 | プロンプトが弱い | システムプロンプトを強化、temperature を下げる |
| ChromaDBエラー | 永続化ディレクトリの問題 | ディレクトリを削除して再構築 |

---

## 10. 将来の拡張ポイント

1. **ハイブリッド検索**：ベクトル検索 + BM25（キーワード検索）の組み合わせ
2. **リランキング**：Cohere Rerank などで検索結果を再順位付け
3. **会話履歴の保持**：マルチターン対話への対応
4. **フィードバック収集**：ユーザー評価をもとにした改善
5. **文書の追加**：新しい学派・理論の追加パイプライン
