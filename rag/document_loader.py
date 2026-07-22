"""文档加载与分块模块"""
import re
from pathlib import Path
from typing import List, Dict


class DocumentLoader:
    """加载知识库文档并切分为语义块"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = [r"\n## ", r"\n# ", r"\n\n", r"\n", r"。", r"\. ", r"！", r"？", r"；", r" ", r""]

    def load_directory(self, dir_path: str) -> List[Dict]:
        """加载目录下所有文档"""
        dir_path = Path(dir_path)
        all_docs = []
        for file_path in sorted(dir_path.glob("*")):
            if file_path.suffix.lower() in [".txt", ".md", ".html"]:
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)
        return all_docs

    def load_file(self, file_path: str) -> List[Dict]:
        """加载单个文件并分块"""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = self._split_text(text)
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{path.stem}_{i}",
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "filename": path.stem,
                    "chunk_index": i,
                }
            })
        return docs

    def _split_text(self, text: str) -> List[str]:
        """递归语义分块"""
        if not text or not isinstance(text, str):
            return []
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            # 找最佳切分点
            best = end
            for sep in self._separators:
                sub = text[start:end + 50]
                match = list(re.finditer(sep, sub))
                if match:
                    best = start + match[-1].end()
                    break
            chunk = text[start:best].strip()
            if chunk:
                chunks.append(chunk)
            start = best - self.chunk_overlap if best - self.chunk_overlap > start else best
            if start >= len(text):
                break
        return chunks
