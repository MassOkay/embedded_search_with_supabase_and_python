# app/vector_search.py
import json
import os
import numpy as np
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from contextlib import contextmanager

from .config import Settings

class VectorSearchService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = SentenceTransformer(self.settings.MODEL_NAME)
        self.documents = self._load_source_documents()
        self.conn_pool = psycopg.ConnectionPool(settings.get_db_connection_string())
        # プール内の全接続でNumpy配列を扱えるようにする
        register_vector(self.conn_pool)

    @contextmanager
    def get_connection(self):
        """コネクションプールから接続を取得し、自動で返却する"""
        conn = self.conn_pool.getconn()
        try:
            yield conn
        finally:
            self.conn_pool.putconn(conn)

    def _load_source_documents(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.settings.SOURCE_JSON_PATH):
            raise FileNotFoundError(f"Source JSON file not found at {self.settings.SOURCE_JSON_PATH}")
        with open(self.settings.SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def search(self, query: str) -> Dict[str, Any]:
        """クエリに基づいてSupabaseでセマンティック検索を実行する"""
        query_vector = self.model.encode(query)

        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                # <=> はコサイン距離(1-類似度)を計算する演算子。小さいほど類似
                cur.execute(
                    """
                    SELECT content, original_doc_id, 1 - (embedding <=> %s) AS score
                    FROM documents
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, self.settings.SEARCH_K)
                )
                hits = cur.fetchall()

        # 検索結果を元のドキュメントに集約
        doc_scores: Dict[int, Dict[str, Any]] = {}
        for hit in hits:
            original_doc_id = hit["original_doc_id"]
            score = hit["score"]
            
            if original_doc_id not in doc_scores or score > doc_scores[original_doc_id]["score"]:
                doc_scores[original_doc_id] = {
                    "score": float(score),
                    "hit_chunk": hit["content"]
                }
        
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda doc_id: doc_scores[doc_id]["score"], reverse=True)
        
        results = []
        for doc_id in sorted_doc_ids:
            doc = self.documents[doc_id]
            score_info = doc_scores[doc_id]
            results.append({
                "title": doc.get("title", "No Title"),
                "overview_snippet": (doc.get("講義概要", "") or "")[:100] + "...",
                "score": score_info["score"],
                "hit_chunk": score_info["hit_chunk"]
            })
            
        return {"query": query, "results": results}