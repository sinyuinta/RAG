"""
FastAPI アプリケーション
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Any
from dotenv import load_dotenv

# サービスは起動時に初期化せず、初回リクエスト時またはイベントで初期化するなど工夫もできるが
# ここではシンプルにモジュールレベルインポート時に初期化を試みる
# ただし、環境変数がロードされていないとエラーになるため注意
load_dotenv()

# 新しい統合エンジンをインポート
from app.rag_engine import RAGSystem

app = FastAPI(
    title="Psychology Theory RAG API",
    description="心理学理論に関する質問応答API",
    version="1.0.0"
)

# 静的ファイルの提供設定（フロントエンドを表示するために必要）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 文書ファイルの提供設定（出典リンクをクリックして開けるようにする）
# 直接 rag_documents フォルダを /docs として公開することでパスを短縮
app.mount("/docs", StaticFiles(directory="documents/rag_documents"), name="docs")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAGシステムの初期化
try:
    rag_system = RAGSystem()
except Exception as e:
    print(f"Warning: Failed to initialize RAG system. {e}")
    rag_system = None

# --- Models ---
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict
    distance: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: Optional[dict] = None  # ancestorなどのフィルタ用

# --- Endpoints ---

@app.get("/")
async def root():
    """ルートアクセスはフロントエンドへリダイレクト"""
    return RedirectResponse(url="/static/index.html")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    質問に対して回答とソースを返す
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG System not initialized")
    
    try:
        # 統合メソッドを使用
        result = rag_system.chat(request.query)
        return result
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    検索のみを実行して結果を返す（デバッグや確認用）
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG System not initialized")
    
    try:
        results = rag_system.search(
            query=request.query, 
            n_results=request.n_results,
            filters=request.filters
        )
        return results
    except Exception as e:
        print(f"Error processing search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    """統計情報の取得"""
    if not rag_system:
        return {"error": "RAG System not initialized"}
        
    try:
        count = rag_system.collection.count()
        return {
            "total_documents": count,
            "collection_name": rag_system.collection.name
        }
    except Exception as e:
        return {"error": str(e)}
