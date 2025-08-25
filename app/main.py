# app/main.py
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse
from functools import lru_cache
import os

from .config import settings, Settings
from .vector_search import VectorSearchService
from .models import SearchResponse, ShowResponse

app = FastAPI(
    title="Vector Search API",
    description="A FastAPI application for semantic search on documents.",
    version="1.0.0"
)

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_search_service(settings: Settings = Depends(get_settings)):
    """
    VectorSearchServiceのインスタンスを生成し、キャッシュします。
    これにより、アプリケーション全体で単一のインスタンスが再利用され、
    重いモデルの読み込みが初回リクエスト時に一度だけ行われます。
    データベース接続に失敗した場合、この関数が呼び出されたときにエラーが発生します。
    """
    try:
        return VectorSearchService(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize VectorSearchService: {e}")
@app.get("/", tags=["General"])
def read_root():
    """ルートエンドポイント"""
    return {"message": "Welcome to the Vector Search API"}

@app.get("/search", response_model=SearchResponse, tags=["Search"])
def perform_search(
    q: str = Query(..., min_length=1, description="検索クエリ"),
    search_service: VectorSearchService = Depends(get_search_service)
):
    """
    クエリに基づいてセマンティック検索を実行し、関連するドキュメントを返す
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        search_results = search_service.search(q)
        return search_results
    except Exception as e:
        # エラーログを記録することが望ましい
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/visualize", tags=["Visualization"])
def get_visualization(settings: Settings = Depends(get_settings)):
    """
    t-SNEで2次元に削減されたチャンクベクトルの可視化画像を返す
    画像はインデックス作成時に事前に生成される
    """
    if not os.path.exists(settings.VISUALIZATION_IMAGE_PATH):
        raise HTTPException(status_code=404, detail="Visualization image not found. It might not have been generated yet.")
    return FileResponse(settings.VISUALIZATION_IMAGE_PATH)

@app.get("/show", response_model=ShowResponse, tags=["Debug"])
def show_all_documents(
    search_service: VectorSearchService = Depends(get_search_service)
):
    """
    （デバッグ用）ロードされている全てのドキュメントを返す
    """
    return {"documents": search_service.documents}