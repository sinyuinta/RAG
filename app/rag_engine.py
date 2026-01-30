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
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

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

    def _build_system_prompt(self) -> str:
        """システムプロンプト構築（厳格な根拠付けと出典分離版）"""
        return """あなたは心理学理論の専門家アシスタントです。
利用者は臨床家であり、提供された内部ライブラリー（RAG資料）に基づいてケースを検討しています。

# 応答の原則（最重要・厳守）
- **「1. 内部ライブラリーに基づく考察」セクションでは、提供された[参考文献]の内容のみを使用し、あなたの事前学習知識（行動心理学、進化心理学など）を絶対に混ぜないでください。**
- ライブラリー内に存在しない理論や概念を、あたかもライブラリーにあるかのように捏造して回答することは厳禁です。
- もしライブラリーの情報だけで5つの視点を構成できない場合は、無理に5つ作らず、存在する情報のみで構成してください（ただし、可能な限り多角的に検討してください）。

# 応答の構造（必須）
回答は必ず以下の2つのセクションに分けて作成してください。

## 1. 内部ライブラリーに基づく考察
- **必ず提供された[参考文献]（内部ライブラリー）の内容のみを根拠としてください。**
- 質問に対して、提示された5つの資料をフル活用し、**必ず「5つの独立した視点（セクション）」を作成**してください。これは必須条件です。
- 各セクションでどの資料を参照したか、必ず `【参考文献1】`〜`【参考文献5】` のいずれかを末記に明記してください。出典の対応関係を正確に保ってください。
- 各視点の見出しの直後に、質問との関連性を「【高】」「【中】」「【低】」のいずれかで必ず表示してください。
- 例：### 精神分析的視点 【高】

## 2. ライブラリー外の一般的視点（補足）
- **このセクションは、内部ライブラリーの情報だけでは臨床的に不十分、あるいは危険な場合にのみ、あくまで「一般論」として最小限に留めてください。**
- 行動心理学や認知バイアスなど、ライブラリー外の知識を用いる場合は必ずこのセクションに分離してください。
- 各視点の横に、関連性を「【高】」「【中】」「【低】」のいずれかで表示してください。
- **冒頭に必ず以下の定型文を記載してください：**
  「これらの情報は一般的にネットで言われているものです。これらについてはお手数ですがご自分でお調べください。」

# 行動指針
- あなたは「渡された資料の範囲内でのみ答える専門家」になりきってください。
- ライブラリーの内容を最優先し、AIの事前学習知識（特にライブラリーに含まれない行動心理学など）をセクション1に含ませないでください。
- ユーザーに提供する情報は客観的な材料にとどめ、最終的な判断はユーザーに委ねてください。
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
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        検索結果に基づいて回答を生成
        """
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        if history:
            messages.extend(history)
            
        context = self._build_context(search_results)
        user_message = f"""以下の参考文献をもとに質問に回答してください。

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

    def chat(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        クエリに対して検索と回答生成を一括で行う（便利メソッド）
        """
        search_results = self.search(query, n_results=n_results)
        answer_text = self.answer(query, search_results)
        
        return {
            "answer": answer_text,
            "sources": search_results
        }
