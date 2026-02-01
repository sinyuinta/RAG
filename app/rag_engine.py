"""
RAGエンジン：検索と生成を統合したメインロジック
"""
import os
import chromadb
from openai import OpenAI
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# 設定
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "psychology_theories"
CHROMA_PERSIST_DIR = "./data/chromadb"
GENERATION_MODEL = "gpt-4o"

class RAGSystem:
    """検索と生成を行う統合クラス"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.openai_client = OpenAI(api_key=self.api_key)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Try to get existing collection, or create and populate if it doesn't exist
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection '{COLLECTION_NAME}' with {self.collection.count()} documents")
        except Exception:
            print(f"Collection '{COLLECTION_NAME}' does not exist. Creating and populating from chunks.json...")
            self._initialize_database()
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"Created collection '{COLLECTION_NAME}' with {self.collection.count()} documents")
    
    def _initialize_database(self):
        """Initialize database from chunks.json"""
        import json
        
        chunks_path = "./data/chunks.json"
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"chunks.json not found at {chunks_path}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Create collection
        collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Psychology theories knowledge base"}
        )
        
        # Prepare batch data
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk['id'] for chunk in batch]
            documents = [chunk.get('content', chunk.get('text', '')) for chunk in batch]
            
            # Build metadata from flat structure
            metadatas = []
            for chunk in batch:
                meta = {
                    'source_file': chunk.get('source_file', ''),
                    'section_title': chunk.get('section_title', ''),
                    'ancestor': chunk.get('ancestor', ''),
                    'family_line': chunk.get('family_line', ''),
                    'source_author': chunk.get('source_author', '')
                }
                metadatas.append(meta)
            
            # Generate embeddings
            embeddings = []
            for doc in documents:
                embedding = self._embed_query(doc)
                embeddings.append(embedding)
            
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")

    def _embed_query(self, query: str) -> List[float]:
        """クエリをEmbedding化（内部利用）"""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding

    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        ベクトル検索を実行
        """
        query_embedding = self._embed_query(query)
        
        # 検索実行
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )
        
        # 結果を整形
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        return formatted_results

    def _get_match_level(self, distance: float) -> str:
        """距離に基づいてマッチ度（高/中/低）を判定"""
        # cosine distance: 0 is exact match.
        if distance < 0.35:
            return "高"
        elif distance < 0.50:
            return "中"
        else:
            return "低"

    def _build_system_prompt(self, mode: str = "case") -> str:
        """システムプロンプト構築（モード別）"""
        if mode == "case":
            return """あなたは心理学理論の専門家アシスタントです。
利用者は臨床家であり、提供された内部ライブラリー（RAG資料）に基づいてケースを検討しています。

# 応答の原則（最重要・厳守）
- **提供された[参考文献]の内容のみを使用し、内部ライブラリー以外の情報を絶対に表示しないでください。**
- あなたの事前学習知識（行動心理学、進化心理学など）を絶対に混ぜないでください。
- 答えをはぐらかさず、資料にある情報を具体的に提示してください。
- 「一般論」や「補足」といったセクションは一切不要です。内部ライブラリーで完結させてください。
- 「渡された資料の範囲内でのみ答える専門家」になりきり、親切に回答してください。

# 応答の構造
- 質問に対して、アドラー、フロイト、ユング、認知といった主要な学派から多角的かつ包括的に検討してください。
- **必ず4つの学派すべてに独立したセクションを作成して言及してください。**（資料が不足している場合は「内部ライブラリー内に該当情報なし」と明記）
- 各学派ごとにセクションを分けてください（例：### アドラー心理学的視点）。
- **各セクションの見出しの末尾に、必ずマッチ度を「高」「中」「低」のいずれかで表示してください（例：### アドラー心理学的視点 高）。**
- **重要：参考文献は、それぞれの学派の解説の真下に配置してください。**
- 各セクションでどの資料を参照したか、必ず `【参考文献1】`〜`【参考文献N】` のいずれかを末記に明記してください。

# 行動指針
- ユーザーに提供する情報は客観的な材料（理論の提示）にとどめ、最終的な判断はユーザーに委ねてください。
"""
        else: # term
            return """あなたは心理学理論の専門家アシスタントです。
心理学用語の意味や概念について、提供された内部ライブラリー（RAG資料）に基づいて、専門的かつ親和性を持って解説します。

# 応答の原則（最重要・厳守）
- **提供された[参考文献]の内容のみを使用し、内部ライブラリー以外の情報を絶対に表示しないでください。**
- 最も関連性の高い一つの学派に絞って回答してください。他の学派の情報は混ぜないでください。
- 答えをはぐらかさず、資料にある定義や背景を正確に説明してください。
- **重要：もし該当する用語の直接的な記載がライブラリー内にない場合、まず「こちらの内部ライブラリー内には情報がありません」と断った上で、その学派の基礎理論（contextに含まれる情報）に基づき、「[学派名]の理論的背景から推察すると...」という形で必ず専門的な考察・解説を行ってください。**
- あなたの事前学習知識を絶対に混ぜないでください。
- 「渡された資料の範囲内でのみ答える専門家」になりきり、親和性を持って解説してください。

# 応答の構造
- 専門用語について、その学派における意味や背景を分かりやすく丁寧に説明してください。
- セクションの見出しには学派名を含め、必ず「心理学的視点」という言葉を入れてください（例：### フロイトの精神分析における心理学的視点 高）。
- **セクションの見出しの末尾に、必ずマッチ度を「高」「中」「低」のいずれかで表示してください。**
- **重要：参考文献は、解説の真下に配置してください。**
- 各セクションでどの資料を参照したか、必ず `【参考文献1】`〜`【参考文献N】` のいずれかを末記に明記してください。

# 行動指針
- 「渡された資料の範囲内でのみ答える専門家」になりきってください。
"""

    def _build_context(self, search_results: List[Dict]) -> str:
        """検索結果からコンテキストテキストを作成"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            meta = result['metadata']
            distance = result['distance']
            match_level = self._get_match_level(distance)
            
            # 使用するラベルを本文中と一致させる
            ref_label = f"【参考文献{i}】"
            
            context_parts.append(
                f"{ref_label} [マッチ度: {match_level}] (Source: {meta.get('source_file', '不明')})\n"
                f"学派: {meta.get('ancestor', '不明')} ({meta.get('family_line', '')})\n"
                f"著者: {meta.get('source_author', '不明')}\n"
                f"セクション: {meta.get('section_title', '')}\n"
                f"内容:\n{result['content']}\n"
            )
        return "\n---\n".join(context_parts)

    def answer(
        self, 
        query: str, 
        search_results: List[Dict], 
        mode: str = "case",
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        検索結果に基づいて回答を生成
        """
        messages = [{"role": "system", "content": self._build_system_prompt(mode)}]
        
        if history:
            messages.extend(history)
            
        context = self._build_context(search_results)
        user_message = f"""以下の参考文献をもとに質問に回答してください。
内容が不足している場合は「こちらの内部ライブラリー内には情報がありません」と控えめに回答してください。

{context}

---
質問: {query}
"""
        messages.append({"role": "user", "content": user_message})
        
        response = self.openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content

    def chat(self, query: str, mode: str = "case", n_results: int = 5) -> Dict[str, Any]:
        """
        クエリに対して検索と回答生成を一括で行う（便利メソッド）
        """
        if mode == "case":
            # 4つの主要学派からそれぞれ検索
            schools = ["Adler", "Freud", "Jung", "Neisser"]
            search_results = []
            for school in schools:
                # 精度を上げるため、その学派の中で最も類似度が高い1件のみを取得
                results = self.search(query, n_results=1, filters={"ancestor": school})
                search_results.extend(results)
        else: # term
            # 用語検索：通常通り上位を検索
            raw_results = self.search(query, n_results=n_results)
            if not raw_results:
                search_results = []
            else:
                # 最も関連性の高い学派を特定
                top_school = raw_results[0]['metadata'].get('ancestor')
                # その学派の結果のみに絞る
                search_results = [r for r in raw_results if r['metadata'].get('ancestor') == top_school]
        
        answer_text = self.answer(query, search_results, mode=mode)
        
        return {
            "answer": answer_text,
            "sources": search_results
        }
