"""load the 10-K PDFs, split them into chunks, and build the search indexes."""
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
CHUNKS_FILE = INDEX_DIR / "chunks.json"
CHUNK_SIZE = 2000      #characters, roughly 500 tokens
CHUNK_OVERLAP = 200    #overlap so sentences cut at a boundary appear in both chunks


def load_pdfs() -> list[Document]:
    """Read every PDF page and keep the file name and page number as metadata."""
    pages = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(Document(page_content=text,
                                      metadata={"source": pdf_path.name, "page": page_number}))
        print(f"Loaded {pdf_path.name}: {len(reader.pages)} pages")
    return pages


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks; each chunk keeps its source and page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return chunks


def main() -> None:
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)  #rebuild from scratch so re-running never duplicates chunks
    INDEX_DIR.mkdir()

    chunks = split_into_chunks(load_pdfs())
    print(f"Created {len(chunks)} chunks")

    #Keyword index input: save the raw chunks so retrieve.py can build BM25 from them.
    CHUNKS_FILE.write_text(json.dumps(
        [{"text": c.page_content, "metadata": c.metadata} for c in chunks]))

    #Vector index: embed every chunk and store it in Chroma on disk.
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="bank_10k",
        persist_directory=str(INDEX_DIR / "chroma"),
    )
    print("Vector index built in index/chroma")


if __name__ == "__main__":
    main()