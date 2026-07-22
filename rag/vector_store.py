"""向量存储模块 - 支持 ChromaDB / Numpy 双后端"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class VectorStore:
    def __init__(self, backend: str = "numpy", persist_dir: str = "./data/vector_db", embedding_model=None, embedding_dim: int = 384):
        self.backend = backend
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._collections: Dict[str, dict] = {}
        self._bm25_index = None
        self._bm25_texts = []
        self._bm25_metadatas = []
        self._initialize()

    def _initialize(self):
        self._ensure_embedding_model()
        self._load_persisted()

    def _ensure_embedding_model(self):
        if self.embedding_model is not None:
            return
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        model_found = False
        if os.path.exists(cache_dir):
            for entry in os.listdir(cache_dir):
                if "paraphrase-multilingual" in entry.lower() or "bge" in entry.lower():
                    model_found = True
                    break
        if model_found:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
                return
            except Exception:
                pass
        self.embedding_model = _FallbackEmbedder(self.embedding_dim)

    def get_or_create_collection(self, name: str) -> dict:
        if name not in self._collections:
            self._collections[name] = {"ids": [], "embeddings": np.array([]).reshape(0, self.embedding_dim), "documents": [], "metadatas": []}
        return self._collections[name]

    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = None):
        if collection_name is None:
            collection_name = "ecommerce_knowledge"
        if not documents:
            return
        col = self.get_or_create_collection(collection_name)
        texts = [d["text"] if isinstance(d, dict) else str(d) for d in documents]
        ids = [d.get("id", f"doc_{i}") if isinstance(d, dict) else f"doc_{i}" for i, d in enumerate(documents)]
        metadatas = [d.get("metadata", {}) if isinstance(d, dict) else {} for d in documents]
        embeddings = self.embedding_model.encode(texts)

        col["ids"].extend(ids)
        col["documents"].extend(texts)
        col["metadatas"].extend(metadatas)
        if len(col["embeddings"]) == 0:
            col["embeddings"] = embeddings
        else:
            col["embeddings"] = np.vstack([col["embeddings"], embeddings])

        self._build_bm25(texts, metadatas)

    def _build_bm25(self, texts: List[str], metadatas: List[Dict]):
        try:
            import jieba
            from rank_bm25 import BM25Okapi
            all_texts = list(self._bm25_texts) + list(texts)
            all_metadatas = list(self._bm25_metadatas) + list(metadatas)
            tokenized = [list(jieba.cut(t)) for t in all_texts]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_texts = all_texts
            self._bm25_metadatas = all_metadatas
        except ImportError:
            self._bm25_index = None

    def similarity_search(self, query: str, k: int = 5, filter: Dict = None) -> List[Dict]:
        col = self._collections.get("ecommerce_knowledge")
        if col is None or len(col["embeddings"]) == 0:
            return self._bm25_fallback(query, k)

        # Dense: Cosine top-20
        query_emb = self.embedding_model.encode([query])[0]
        embeddings = col["embeddings"]
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(emb_norms, query_norm)
        n_dense = min(20, len(similarities))
        dense_idx = np.argsort(similarities)[::-1][:n_dense]

        # Sparse: BM25 top-20
        sparse_results = self._bm25_fallback(query, 20)

        # RRF 融合
        rrf_k = 60
        scores = {}
        doc_map = {}
        for rank, idx in enumerate(dense_idx, 1):
            doc_id = col["ids"][idx]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
            doc_map[doc_id] = {"text": col["documents"][idx], "metadata": col["metadatas"][idx], "score": float(similarities[idx]), "id": doc_id}

        for rank, doc in enumerate(sparse_results, 1):
            did = doc["id"]
            scores[did] = scores.get(did, 0) + 1 / (rrf_k + rank)
            if did not in doc_map:
                doc_map[did] = doc

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, rrf_score in sorted_ids[:k]:
            doc = doc_map[doc_id]
            doc["rrf_score"] = rrf_score
            results.append(doc)

        if not results:
            return self._bm25_fallback(query, k)
        return results

    def _bm25_fallback(self, query: str, k: int = 5) -> List[Dict]:
        if self._bm25_index is None:
            return []
        try:
            import jieba
            tokens = list(jieba.cut(query))
            scores = self._bm25_index.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({"text": self._bm25_texts[idx], "metadata": self._bm25_metadatas[idx], "score": float(scores[idx]), "id": f"bm25_{idx}"})
            return results
        except Exception:
            return []

    def get_collection_size(self) -> int:
        col = self._collections.get("ecommerce_knowledge")
        return len(col["documents"]) if col else 0

    def _load_persisted(self):
        p = Path(self.persist_dir) / "ecommerce_knowledge"
        if p.exists():
            emb_path = p / "embeddings.npy"
            data_path = p / "data.json"
            if emb_path.exists() and data_path.exists():
                try:
                    embeddings = np.load(emb_path)
                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._collections["ecommerce_knowledge"] = {"ids": data["ids"], "embeddings": embeddings, "documents": data["documents"], "metadatas": data["metadatas"]}
                    self._build_bm25(data["documents"], data["metadatas"])
                except Exception:
                    pass

    def save_all(self):
        col = self._collections.get("ecommerce_knowledge")
        if col is None:
            return
        save_dir = Path(self.persist_dir) / "ecommerce_knowledge"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "embeddings.npy", col["embeddings"])
        with open(save_dir / "data.json", "w", encoding="utf-8") as f:
            json.dump({"ids": col["ids"], "documents": col["documents"], "metadatas": col["metadatas"]}, f, ensure_ascii=False, default=str)

    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        return self.similarity_search(query, k=k)


class _FallbackEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            chars = list(text)
            for j, char in enumerate(chars[:self.dim * 3]):
                code = ord(char)
                vectors[i, j % self.dim] += code / 65536.0
                vectors[i, (j * 7 + 13) % self.dim] += (code % 256) / 256.0
            for j in range(len(chars) - 1):
                bg = ord(chars[j]) * 256 + ord(chars[j + 1])
                vectors[i, bg % self.dim] += 0.1
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors
