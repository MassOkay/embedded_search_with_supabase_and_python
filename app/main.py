# app/main.py
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from .config import settings, Settings
from .vector_search import VectorSearchService
from .models import SearchResponse, ShowResponse

app = FastAPI(
    title="Vector Search API",
    description="A FastAPI application for semantic search on documents.",
    version="1.0.0"
)

# アプリケーションの生存期間中、VectorSearchServiceのインスタンスを一つだけ作成する
# これにより、重いモデルやインデックスの読み込みが起動時に一度だけ行われる
search_service = VectorSearchService(settings)

@app.get("/", tags=["General"])
def read_root():
    """ルートエンドポイント"""
    return {"message": "Welcome to the Vector Search API"}

@app.get("/search", response_model=SearchResponse, tags=["Search"])
def perform_search(q: str = Query(..., min_length=1, description="検索クエリ")):
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
def get_visualization():
    """
    t-SNEで2次元に削減されたチャンクベクトルの可視化画像を返す
    画像はインデックス作成時に事前に生成される
    """
    if not os.path.exists(settings.VISUALIZATION_IMAGE_PATH):
        raise HTTPException(status_code=404, detail="Visualization image not found. It might not have been generated yet.")
    return FileResponse(settings.VISUALIZATION_IMAGE_PATH, media_type="image/png")

@app.get("/show", response_model=ShowResponse, tags=["Debug"])
def show_all_documents():
    """
    （デバッグ用）ロードされている全てのドキュメントを返す
    """
    return {"documents": search_service.documents}