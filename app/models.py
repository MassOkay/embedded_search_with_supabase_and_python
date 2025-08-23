# app/models.py
from pydantic import BaseModel
from typing import List, Optional

class SearchResultItem(BaseModel):
    """検索結果の各アイテムのスキーマ"""
    title: str
    overview_snippet: str
    score: float
    hit_chunk: str

class SearchResponse(BaseModel):
    """/search エンドポイントのレスポンススキーマ"""
    query: str
    results: List[SearchResultItem]

class DocumentItem(BaseModel):
    """/show エンドポイントで返されるドキュメントのスキーマ"""
    # 元のJSONの構造に合わせてフィールドを定義
    # 例: title, 講義概要, ...
    # ここではAnyで受け入れる
    data: dict

class ShowResponse(BaseModel):
    """/show エンドポイントのレスポンススキーマ"""
    documents: List[dict] # 簡単のため辞書のリストとしておく