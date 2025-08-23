# ingest_data.py
import json
import re
import numpy as np
import os
import psycopg
from psycopg.rows import dict_row
from psycopg.types.vector import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# app/config.py から設定をインポート
from app.config import settings

def split_into_sentences(text: str) -> list[str]:
    """テキストを文単位に分割する"""
    sentences = re.split(r'(?<=[。！？.])\s*|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def main():
    """ドキュメントを処理し、ベクトル化してSupabaseに登録する"""
    
    # --- 1. モデルの初期化 ---
    print(f"Loading model: {settings.MODEL_NAME}")
    model = SentenceTransformer(settings.MODEL_NAME)

    # --- 2. チャンクの作成 ---
    print(f"Loading source documents from {settings.SOURCE_JSON_PATH}")
    with open(settings.SOURCE_JSON_PATH, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    chunks_to_embed = []
    for i, doc in enumerate(documents):
        title = doc.get("title", "")
        full_text = f"{doc.get('講義概要', '')}\n{doc.get('授業科目の内容・目的・方法・到達目標', '')}".strip()
        chunks = split_into_sentences(full_text)
        for chunk_text in chunks:
            if chunk_text:
                chunks_to_embed.append({
                    "original_doc_id": i,
                    "content": f"{title}: {chunk_text}"
                })
    print(f"Created {len(chunks_to_embed)} chunks.")

    # --- 3. ベクトルのエンコード ---
    print("Encoding chunks into vectors...")
    embeddings = model.encode(
        [chunk["content"] for chunk in chunks_to_embed],
        show_progress_bar=True
    )

    # --- 4. Supabaseへの接続とテーブル作成 ---
    print("Connecting to Supabase database...")
    with psycopg.connect(settings.get_db_connection_string()) as conn:
        with conn.cursor() as cur:
            # pgvector拡張機能はSupabase UIで有効化済みと想定
            # print("Ensuring vector extension is enabled...")
            # cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            print("Creating table 'documents'...")
            cur.execute("DROP TABLE IF EXISTS documents;")
            cur.execute(f"""
                CREATE TABLE documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    original_doc_id INTEGER,
                    embedding VECTOR({settings.DIMENSION})
                );
            """)
            
            # --- 5. データのバッチ登録 ---
            print(f"Inserting {len(chunks_to_embed)} records into database...")
            # psycopgでNumpy配列をvector型として扱えるように登録
            register_vector(conn)
            
            with cur.copy("COPY documents (content, original_doc_id, embedding) FROM STDIN") as copy:
                for i, chunk in enumerate(tqdm(chunks_to_embed)):
                    copy.write_row((chunk['content'], chunk['original_doc_id'], embeddings[i]))

            # --- 6. インデックスの作成 (検索高速化のため) ---
            print("Creating index on embeddings...")
            # IVFFlatインデックスを作成。リスト数はデータの1/1000かsqrt(N)が良いとされる
            num_lists = max(1, int(len(chunks_to_embed) / 100))
            cur.execute(f"CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});")
            
    print("Data ingestion completed successfully.")
    # 可視化のロジックは省略（必要であればingest_data.pyの前バージョンから追加可能）

if __name__ == "__main__":
    main()