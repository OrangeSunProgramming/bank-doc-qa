"""hybrid retrieval (vector + BM25 keyword search, merged with reciprocal rank fusion)."""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

INDEX_DIR = Path("index")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.vector_store = Chroma(
            collection_name="bank_10k",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=str(INDEX_DIR / "chroma"),
        )

    def search(self, question: str, k: int = 5, pool: int = 20) -> list[dict]:
        # 1. Vector search: chunks whose meaning is closest to the question.
        vector_hits = self.vector_store.similarity_search(question, k=pool)
        vector_ids = [d.metadata["chunk_id"] for d in vector_hits]

        # 2. Keyword search: chunks sharing the question's exact words (names, numbers, terms).
        scores = self.bm25.get_scores(tokenize(question))
        keyword_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]

        # 3. Reciprocal rank fusion: reward chunks ranked high by either method.
        fused: dict[int, float] = {}
        for ranking in (vector_ids, keyword_ids):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)

        top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.chunks[i] for i in top_ids]


def main() -> None:
    retriever = HybridRetriever()
    print("Ask a question about the 10-Ks (press Enter on an empty line to quit).")
    while question := input("\nQuestion: ").strip():
        for rank, chunk in enumerate(retriever.search(question), start=1):
            meta = chunk["metadata"]
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"\n[{rank}] {meta['source']}, page {meta['page']}\n    {preview}...")


if __name__ == "__main__":
    main()